import argparse
import json
from pathlib import Path

from validate_playwright_json import validate_report


def identities(report):
    found = []
    if any(project["repeatEach"] != 1 for project in report["config"]["projects"]):
        raise ValueError("The comparison requires exactly one repetition")

    def visit(suites):
        for suite in suites:
            for spec in suite.get("specs", []):
                for test in spec["tests"]:
                    key = (spec["id"], test["projectName"])
                    if not key[0] or not key[1]:
                        raise ValueError("Empty test identity")
                    found.append(key)
            visit(suite.get("suites", []))

    visit(report["suites"])
    if len(found) != len(set(found)):
        raise ValueError("Duplicate test identity within a report")
    return set(found)


def verify(discovery, baseline, shards):
    if discovery["config"].get("shard") is not None:
        raise ValueError("Discovery must cover the unsharded suite")
    expected = identities(discovery)
    if len(expected) < 28:
        raise ValueError("Discovery is below the existing smoke-suite floor")
    validate_report(baseline, min_tests=28)
    if baseline["config"].get("shard") is not None:
        raise ValueError("Baseline must run the complete suite")
    if identities(baseline) != expected:
        raise ValueError("Baseline differs from the discovered tests")
    if set(shards) != {1, 2, 3, 4}:
        raise ValueError("All four shard reports are required")
    actual = set()
    for number, report in shards.items():
        validate_report(report)
        if report["config"].get("shard") != {"current": number, "total": 4}:
            raise ValueError(f"Wrong shard configuration in report {number}")
        selected = identities(report)
        if actual & selected:
            raise ValueError("Test identities overlap across shards")
        actual.update(selected)
    if actual != expected:
        raise ValueError("Shard union differs from the independently discovered suite")
    return len(expected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", type=Path)
    args = parser.parse_args()
    reports = {}
    discovery = None
    for number in range(5):
        directory = args.reports / f"playwright-test-results-smoke-{number}"
        manifests = list(directory.rglob("smoke-inventory.json"))
        results = list(directory.rglob("results.json"))
        if len(manifests) != 1 or len(results) != 1:
            raise ValueError(f"Missing or ambiguous reports for shard {number}")
        current = json.loads(manifests[0].read_text())
        if discovery is not None and identities(current) != identities(discovery):
            raise ValueError("Runners discovered different test inventories")
        if current["config"].get("shard") is not None:
            raise ValueError("Discovery was itself sharded")
        discovery = current
        reports[number] = json.loads(results[0].read_text())
    count = verify(discovery, reports.pop(0), reports)
    print(f"Baseline and four shards each cover the same {count} tests once, without skips or retries")


if __name__ == "__main__":
    main()
