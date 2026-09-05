#!/usr/bin/env python3

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path


JUNIT_REPORTS = (
    "runtime-helpers.xml",
    "publication-policy.xml",
    "smoke.xml",
)
TRIVY_REPORTS = (
    "trivy-critical.json",
    "trivy-secrets.json",
)
TRIVY_FINDING_KEYS = (
    "Vulnerabilities",
    "Misconfigurations",
    "Secrets",
    "Licenses",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--expected-image", required=True)
    return parser.parse_args()


def nonnegative_count(element: ET.Element, attribute: str, report: str) -> int:
    value = element.get(attribute)
    if value is None:
        raise ValueError(f"{report} is missing {attribute}")
    try:
        count = int(value)
    except ValueError as error:
        raise ValueError(f"{report} has invalid {attribute}={value!r}") from error
    if count < 0:
        raise ValueError(f"{report} has negative {attribute}={count}")
    return count


def verify_junit(path: Path) -> None:
    root = ET.parse(path).getroot()
    if root.tag not in {"testsuite", "testsuites"}:
        raise ValueError(f"{path.name} is not a JUnit XML report")
    cases = list(root.iter("testcase"))
    summaries = [(root, cases)]
    if root.tag == "testsuites":
        summaries.extend(
            (suite, list(suite.iter("testcase"))) for suite in root.iter("testsuite")
        )
    if not cases:
        raise ValueError(f"{path.name} contains no test cases")

    for summary, summary_cases in summaries:
        actual = {
            "tests": len(summary_cases),
            "failures": sum(case.find("failure") is not None for case in summary_cases),
            "errors": sum(case.find("error") is not None for case in summary_cases),
            "skipped": sum(case.find("skipped") is not None for case in summary_cases),
        }
        for attribute, actual_count in actual.items():
            declared_count = nonnegative_count(summary, attribute, path.name)
            if declared_count != actual_count:
                raise ValueError(
                    f"{path.name} declares {attribute}={declared_count}, "
                    f"found {actual_count}"
                )

    failures = sum(case.find("failure") is not None for case in cases)
    errors = sum(case.find("error") is not None for case in cases)
    skipped = sum(case.find("skipped") is not None for case in cases)
    if failures:
        raise ValueError(f"{path.name} reports {failures} failures")
    if errors:
        raise ValueError(f"{path.name} reports {errors} errors")
    if len(cases) == skipped:
        raise ValueError(f"{path.name} contains no successful test cases")


def verify_trivy(path: Path, expected_image: str) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or "Results" not in document:
        raise ValueError(f"{path.name} is not a Trivy JSON report")
    if document.get("SchemaVersion") != 2:
        raise ValueError(f"{path.name} has an unexpected schema version")
    if document.get("ArtifactType") != "container_image":
        raise ValueError(f"{path.name} did not scan a container image")
    if document.get("ArtifactName") != expected_image:
        raise ValueError(
            f"{path.name} scanned {document.get('ArtifactName')!r}, "
            f"expected {expected_image!r}"
        )

    results = document["Results"]
    if not isinstance(results, list) or not results:
        raise ValueError(f"{path.name} has an invalid Results value")

    findings: dict[str, int] = {}
    for result in results:
        if not isinstance(result, dict):
            raise ValueError(f"{path.name} has an invalid result entry")
        for key in ("Target", "Class", "Type"):
            if not isinstance(result.get(key), str) or not result[key]:
                raise ValueError(f"{path.name} result is missing {key}")
        for key in TRIVY_FINDING_KEYS:
            if key not in result:
                continue
            value = result[key]
            if not isinstance(value, list):
                raise ValueError(f"{path.name} has an invalid {key} value")
            if value:
                findings[key] = findings.get(key, 0) + len(value)

    if findings:
        summary = ", ".join(f"{key}={count}" for key, count in sorted(findings.items()))
        raise ValueError(f"{path.name} contains findings: {summary}")


def main() -> int:
    args = parse_args()
    try:
        for report in JUNIT_REPORTS:
            verify_junit(args.report_dir / report)
        for report in TRIVY_REPORTS:
            verify_trivy(args.report_dir / report, args.expected_image)
    except (OSError, ET.ParseError, json.JSONDecodeError, ValueError) as error:
        print(f"single-container report verification failed: {error}")
        return 1
    print("single-container reports contain no failures or findings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
