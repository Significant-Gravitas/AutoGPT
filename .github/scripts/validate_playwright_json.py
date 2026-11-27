#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path
from typing import Iterator

# What pinKnownFailure() (src/playwright/journeys/known-failure.ts) records:
# the one check a known failure is allowed to fail on.
KNOWN_FAILURE_ANNOTATION = "known-failure"
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def _require_mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _require_list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def _require_count(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _iter_tests(
    suites: list[object], location: str = "suites"
) -> Iterator[dict[str, object]]:
    for suite_index, raw_suite in enumerate(suites):
        suite_name = f"{location}[{suite_index}]"
        suite = _require_mapping(raw_suite, suite_name)
        for spec_index, raw_spec in enumerate(
            _require_list(suite.get("specs", []), f"{suite_name}.specs")
        ):
            spec_name = f"{suite_name}.specs[{spec_index}]"
            spec = _require_mapping(raw_spec, spec_name)
            for test_index, raw_test in enumerate(
                _require_list(spec.get("tests"), f"{spec_name}.tests")
            ):
                yield _require_mapping(raw_test, f"{spec_name}.tests[{test_index}]")
        yield from _iter_tests(
            _require_list(suite.get("suites", []), f"{suite_name}.suites"),
            f"{suite_name}.suites",
        )


def _pinned_check(test: dict[str, object], test_name: str) -> str:
    """The check a known failure is pinned to fail on: its one
    ``known-failure`` annotation, which pinKnownFailure() adds."""
    pinned: list[str] = []
    for index, raw in enumerate(
        _require_list(test.get("annotations", []), f"{test_name}.annotations")
    ):
        annotation = _require_mapping(raw, f"{test_name}.annotations[{index}]")
        if annotation.get("type") == KNOWN_FAILURE_ANNOTATION:
            pinned.append(str(annotation.get("description") or "").strip())
    if len(pinned) != 1 or not pinned[0]:
        raise ValueError(
            f"{test_name} is a known failure with {len(pinned)} pinned checks, "
            "not one: pin it once, with pinKnownFailure() right before the "
            "check it fails"
        )
    return pinned[0]


def _failed_check(error: dict[str, object]) -> str:
    """The check an error failed: the first line of its message, which is the
    message `expect(value, message)` was given."""
    lines = _ANSI_ESCAPE.sub("", str(error.get("message") or "")).strip().splitlines()
    return lines[0].removeprefix("Error: ").strip() if lines else ""


def validate_report(
    report: object, min_tests: int = 1, allow_known_failures: bool = False
) -> int:
    root = _require_mapping(report, "report")
    errors = _require_list(root.get("errors"), "errors")
    if errors:
        raise ValueError(f"report contains {len(errors)} top-level errors")

    config = _require_mapping(root.get("config"), "config")
    projects = _require_list(config.get("projects"), "config.projects")
    if not projects:
        raise ValueError("config.projects must contain at least one project")
    for index, raw_project in enumerate(projects):
        project = _require_mapping(raw_project, f"config.projects[{index}]")
        retries = _require_count(
            project.get("retries"), f"config.projects[{index}].retries"
        )
        if retries != 0:
            raise ValueError(
                f"config.projects[{index}].retries must be 0, got {retries}"
            )

    stats = _require_mapping(root.get("stats"), "stats")
    counts = {
        key: _require_count(stats.get(key), f"stats.{key}")
        for key in ("expected", "skipped", "unexpected", "flaky")
    }
    for key in ("skipped", "unexpected", "flaky"):
        if counts[key] != 0:
            raise ValueError(f"stats.{key} must be 0, got {counts[key]}")
    if counts["expected"] == 0:
        raise ValueError("report contains zero expected tests")

    tests = list(_iter_tests(_require_list(root.get("suites"), "suites")))
    if len(tests) != counts["expected"]:
        raise ValueError(
            f"discovered {len(tests)} tests but stats.expected is {counts['expected']}"
        )
    if len(tests) < min_tests:
        raise ValueError(
            f"report contains {len(tests)} tests, below required minimum {min_tests}"
        )

    for index, test in enumerate(tests):
        test_name = f"test[{index}]"
        if test.get("status") != "expected":
            raise ValueError(
                f"{test_name}.status must be expected, got {test.get('status')!r}"
            )
        # test.fail(): a known bug the test is pinned to. It still has to fail.
        known_failure = allow_known_failures and test.get("expectedStatus") == "failed"
        if test.get("expectedStatus") != "passed" and not known_failure:
            raise ValueError(
                f"{test_name}.expectedStatus must be passed, got "
                f"{test.get('expectedStatus')!r}"
            )
        results = _require_list(test.get("results"), f"{test_name}.results")
        if len(results) != 1:
            raise ValueError(
                f"{test_name}.results must contain exactly one attempt, got {len(results)}"
            )
        result = _require_mapping(results[0], f"{test_name}.results[0]")
        retry = _require_count(result.get("retry"), f"{test_name}.results[0].retry")
        if retry != 0:
            raise ValueError(f"{test_name}.results[0].retry must be 0, got {retry}")
        if known_failure:
            if result.get("status") != "failed":
                raise ValueError(
                    f"{test_name}.results[0].status must be failed for a known "
                    f"failure, got {result.get('status')!r}"
                )
            # Only on the pinned check: anything else that failed (a cleanup,
            # the journey guard at teardown, an earlier soft check) is a new
            # failure, not the known bug.
            pinned = _pinned_check(test, test_name)
            result_errors = _require_list(
                result.get("errors"), f"{test_name}.results[0].errors"
            )
            if not result_errors:
                raise ValueError(
                    f"{test_name} is a known failure that reported no error"
                )
            for error_index, raw_error in enumerate(result_errors):
                failed = _failed_check(
                    _require_mapping(
                        raw_error, f"{test_name}.results[0].errors[{error_index}]"
                    )
                )
                if failed != pinned:
                    raise ValueError(
                        f"{test_name} is pinned to fail on {pinned!r}, but it "
                        f"also failed on {failed!r}"
                    )
            if len(result_errors) != 1:
                raise ValueError(
                    f"{test_name} failed {len(result_errors)} times on its "
                    f"pinned check {pinned!r}; a known failure fails once"
                )
            continue
        if result.get("status") != "passed":
            raise ValueError(
                f"{test_name}.results[0].status must be passed, got "
                f"{result.get('status')!r}"
            )
        result_errors = _require_list(
            result.get("errors"), f"{test_name}.results[0].errors"
        )
        if result_errors:
            raise ValueError(
                f"{test_name}.results[0] contains {len(result_errors)} errors"
            )

    return len(tests)


def _write_invalid_report(path: Path, error: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "config": {"projects": []},
                "suites": [],
                "errors": [{"message": error}],
                "stats": {
                    "startTime": "",
                    "duration": 0,
                    "expected": 0,
                    "skipped": 0,
                    "unexpected": 1,
                    "flaky": 0,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail unless a Playwright JSON report is complete and clean."
    )
    parser.add_argument(
        "--synthesize-invalid",
        action="store_true",
        help="replace a missing, empty, or malformed report with a machine-readable error",
    )
    parser.add_argument(
        "--min-tests",
        type=int,
        default=1,
        help="minimum number of discovered passing tests required",
    )
    parser.add_argument(
        "--allow-known-failures",
        action="store_true",
        help="accept tests marked test.fail() that failed as expected",
    )
    parser.add_argument("report", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.min_tests < 1:
        print("--min-tests must be at least 1")
        return 2
    try:
        raw_report = args.report.read_text(encoding="utf-8")
        if not raw_report.strip():
            raise ValueError("report is empty")
        report = json.loads(raw_report)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        problem = f"{args.report}: {exc}"
        if args.synthesize_invalid:
            _write_invalid_report(args.report, problem)
        print(problem)
        return 1

    try:
        test_count = validate_report(
            report,
            min_tests=args.min_tests,
            allow_known_failures=args.allow_known_failures,
        )
    except ValueError as exc:
        print(f"{args.report}: {exc}")
        return 1

    print(f"{args.report}: {test_count} tests passed once without skips or retries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
