#!/usr/bin/env python3

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree


@dataclass(frozen=True)
class JUnitSummary:
    tests: int
    failures: int
    errors: int
    skipped: int

    @property
    def passed(self) -> int:
        return self.tests - self.failures - self.errors - self.skipped


def _declared_count(root: ElementTree.Element, key: str) -> int | None:
    raw = root.get(key)
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"root {key} count is not an integer: {raw!r}") from exc
    if value < 0:
        raise ValueError(f"root {key} count is negative: {value}")
    return value


def _summarize_cases(element: ElementTree.Element) -> JUnitSummary:
    cases = list(element.iter("testcase"))
    return JUnitSummary(
        tests=len(cases),
        failures=sum(case.find("failure") is not None for case in cases),
        errors=sum(case.find("error") is not None for case in cases),
        skipped=sum(case.find("skipped") is not None for case in cases),
    )


def _validate_declared_counts(
    element: ElementTree.Element, summary: JUnitSummary, label: str
) -> None:
    for key in ("tests", "failures", "errors", "skipped"):
        declared = _declared_count(element, key)
        actual = getattr(summary, key)
        if declared is not None and declared != actual:
            raise ValueError(
                f"{label} {key} count is {declared}, but test cases contain {actual}"
            )


def summarize_junit(root: ElementTree.Element) -> JUnitSummary:
    if root.tag not in {"testsuite", "testsuites"}:
        raise ValueError(f"unexpected root element: {root.tag!r}")

    summary = _summarize_cases(root)
    _validate_declared_counts(root, summary, "root")
    for suite in root.iter("testsuite"):
        if suite is root:
            continue
        suite_summary = _summarize_cases(suite)
        _validate_declared_counts(
            suite, suite_summary, f"testsuite {suite.get('name', '<unnamed>')!r}"
        )

    if summary.tests == 0:
        raise ValueError("report contains no test cases")
    if summary.failures or summary.errors:
        raise ValueError(
            f"report contains {summary.failures} failures and {summary.errors} errors"
        )
    return summary


def write_synthetic_error(path: Path, reason: str) -> None:
    root = ElementTree.Element(
        "testsuites", tests="1", failures="0", errors="1", skipped="0"
    )
    suite = ElementTree.SubElement(
        root,
        "testsuite",
        name="ci-report-finalization",
        tests="1",
        failures="0",
        errors="1",
        skipped="0",
    )
    case = ElementTree.SubElement(
        suite,
        "testcase",
        classname="ci.report",
        name=f"{path.name} was not produced",
    )
    error = ElementTree.SubElement(case, "error", message=reason)
    error.text = reason
    ElementTree.indent(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    ElementTree.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def validate_path(path: Path, synthesize_invalid: bool) -> JUnitSummary:
    try:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError("report is missing or empty")
        root = ElementTree.parse(path).getroot()
    except (OSError, ElementTree.ParseError, ValueError) as exc:
        if synthesize_invalid:
            write_synthetic_error(path, str(exc))
        raise ValueError(str(exc)) from exc
    return summarize_junit(root)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--synthesize-invalid",
        action="store_true",
        help="replace missing, empty, or malformed reports with a one-error JUnit report",
    )
    parser.add_argument("reports", nargs="+", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    problems: list[str] = []
    for report in args.reports:
        try:
            summary = validate_path(report, args.synthesize_invalid)
        except ValueError as exc:
            problems.append(f"{report}: {exc}")
            continue
        print(
            f"{report}: {summary.tests} tests, {summary.passed} passed, "
            f"{summary.skipped} skipped"
        )

    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
