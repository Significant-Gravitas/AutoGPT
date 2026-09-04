#!/usr/bin/env python3

import argparse
import json
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


def validate_path(
    path: Path, synthesize_invalid: bool, allowed_skips: set[str] | None = None
) -> JUnitSummary:
    try:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError("report is missing or empty")
        root = ElementTree.parse(path).getroot()
    except (OSError, ElementTree.ParseError, ValueError) as exc:
        if synthesize_invalid:
            try:
                write_synthetic_error(path, str(exc))
            except OSError as write_exc:
                raise ValueError(
                    f"{exc}; could not write synthetic error report: {write_exc}"
                ) from write_exc
        raise ValueError(str(exc)) from exc
    summary = summarize_junit(root)
    if allowed_skips is not None:
        unexpected = [
            f"{case.get('classname', '')}.{case.get('name', '')}"
            for case in root.iter("testcase")
            if case.find("skipped") is not None
            and f"{case.get('classname', '')}.{case.get('name', '')}"
            not in allowed_skips
        ]
        if unexpected:
            raise ValueError(f"unapproved skipped test IDs: {', '.join(unexpected)}")
        if summary.passed == 0:
            raise ValueError("report contains no executed passing tests")
    return summary


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--synthesize-invalid",
        action="store_true",
        help="replace missing, empty, or malformed reports with a one-error JUnit report",
    )
    skip_policy = parser.add_mutually_exclusive_group()
    skip_policy.add_argument(
        "--require-no-skips",
        action="store_true",
        help="reject reports containing skipped test cases",
    )
    skip_policy.add_argument(
        "--allow-skips-from",
        type=Path,
        help="reject skips not listed as exact classname.name IDs in this JSON file",
    )
    parser.add_argument("reports", nargs="+", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    problems: list[str] = []
    allowed_skips = None
    if args.allow_skips_from:
        try:
            entries = json.loads(args.allow_skips_from.read_text(encoding="utf-8"))
            if (
                not isinstance(entries, list)
                or any(
                    not isinstance(entry, str)
                    or "." not in entry
                    or not entry.split(".", 1)[0].strip()
                    or not entry.rsplit(".", 1)[-1].strip()
                    for entry in entries
                )
                or len(set(entries)) != len(entries)
            ):
                raise ValueError(
                    "skip allowlist must contain unique nonempty classname.name IDs"
                )
            allowed_skips = set(entries)
        except (OSError, ValueError) as exc:
            print(f"Invalid skip allowlist: {exc}", file=sys.stderr)
            return 1
    for report in args.reports:
        try:
            summary = validate_path(report, args.synthesize_invalid, allowed_skips)
        except ValueError as exc:
            problems.append(f"{report}: {exc}")
            if args.allow_skips_from and "unapproved skipped test IDs" in str(exc):
                problems.append(
                    f"Review the skip policy in {args.allow_skips_from} before "
                    "approving an intentional new skip; do not allowlist a regression."
                )
            continue
        if args.require_no_skips and summary.skipped:
            problems.append(
                f"{report}: report contains {summary.skipped} skipped tests"
            )
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
