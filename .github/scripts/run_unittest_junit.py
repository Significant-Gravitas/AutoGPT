#!/usr/bin/env python3

import argparse
import sys
import time
import unittest
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TestRecord:
    classname: str
    name: str
    duration: float
    outcome: str
    detail: str = ""


class JUnitResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records: list[TestRecord] = []
        self._started_at: dict[int, float] = {}
        self._outcomes: dict[int, tuple[str, str]] = {}
        self._recorded: set[int] = set()

    def _append_record(self, test, duration: float, outcome: str, detail: str):
        test_id = test.id()
        classname, _, name = test_id.rpartition(".")
        self.records.append(
            TestRecord(
                classname=classname,
                name=name or test_id,
                duration=duration,
                outcome=outcome,
                detail=detail,
            )
        )
        self._recorded.add(id(test))

    def startTest(self, test):
        self._started_at[id(test)] = time.monotonic()
        self._outcomes[id(test)] = ("success", "")
        super().startTest(test)

    def addFailure(self, test, err):
        self._outcomes[id(test)] = ("failure", self._exc_info_to_string(err, test))
        super().addFailure(test, err)

    def addError(self, test, err):
        detail = self._exc_info_to_string(err, test)
        if id(test) in self._started_at:
            self._outcomes[id(test)] = ("error", detail)
        else:
            self._append_record(test, 0.0, "error", detail)
        super().addError(test, err)

    def addSkip(self, test, reason):
        self._outcomes[id(test)] = ("skipped", reason)
        super().addSkip(test, reason)

    def addExpectedFailure(self, test, err):
        self._outcomes[id(test)] = (
            "skipped",
            f"expected failure\n{self._exc_info_to_string(err, test)}",
        )
        super().addExpectedFailure(test, err)

    def addUnexpectedSuccess(self, test):
        self._outcomes[id(test)] = ("failure", "unexpected success")
        super().addUnexpectedSuccess(test)

    def addSubTest(self, test, subtest, err):
        if err is not None:
            outcome = (
                "failure" if issubclass(err[0], test.failureException) else "error"
            )
            detail = self._exc_info_to_string(err, subtest)
            self._outcomes[id(test)] = (outcome, detail)
        super().addSubTest(test, subtest, err)

    def stopTest(self, test):
        started_at = self._started_at.pop(id(test), time.monotonic())
        outcome, detail = self._outcomes.pop(id(test), ("success", ""))
        if id(test) not in self._recorded:
            self._append_record(
                test,
                time.monotonic() - started_at,
                outcome,
                detail,
            )
        self._recorded.discard(id(test))
        super().stopTest(test)


def write_junit(
    path: Path,
    suite_name: str,
    records: list[TestRecord],
    duration: float,
) -> None:
    failures = sum(record.outcome == "failure" for record in records)
    errors = sum(record.outcome == "error" for record in records)
    skipped = sum(record.outcome == "skipped" for record in records)
    suite = ET.Element(
        "testsuite",
        {
            "name": suite_name,
            "tests": str(len(records)),
            "failures": str(failures),
            "errors": str(errors),
            "skipped": str(skipped),
            "time": f"{duration:.6f}",
        },
    )
    for record in records:
        case = ET.SubElement(
            suite,
            "testcase",
            {
                "classname": record.classname,
                "name": record.name,
                "time": f"{record.duration:.6f}",
            },
        )
        if record.outcome != "success":
            tag = "skipped" if record.outcome == "skipped" else record.outcome
            message = record.detail.splitlines()[0] if record.detail else record.outcome
            child = ET.SubElement(case, tag, {"message": message})
            child.text = record.detail

    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-directory", required=True)
    parser.add_argument("--pattern", default="test*.py")
    parser.add_argument("--suite-name", default="single-container-runtime-helpers")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    suite = unittest.defaultTestLoader.discover(
        args.start_directory,
        pattern=args.pattern,
    )
    started_at = time.monotonic()
    runner = unittest.TextTestRunner(
        stream=sys.stderr,
        verbosity=2,
        resultclass=JUnitResult,
    )
    result = runner.run(suite)
    write_junit(
        args.output,
        args.suite_name,
        result.records,
        time.monotonic() - started_at,
    )
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
