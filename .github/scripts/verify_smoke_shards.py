import argparse
import json
from pathlib import Path

from validate_playwright_json import validate_report

SHARD_COUNT = 4
SMOKE_TEST_FLOOR = 28


def identities(report):
    found = []
    projects = report["config"]["projects"]
    if not projects or any(project["repeatEach"] != 1 for project in projects):
        raise ValueError("Smoke tests must run exactly once")

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


def discovery_identities(report):
    if report["errors"] or report["config"].get("shard") is not None:
        raise ValueError("Discovery must cover the complete suite without errors")
    expected = identities(report)
    if len(expected) < SMOKE_TEST_FLOOR:
        raise ValueError("Discovery is below the existing smoke-suite floor")
    return expected


def verify(discovery, shards):
    expected = discovery_identities(discovery)
    if set(shards) != set(range(1, SHARD_COUNT + 1)):
        raise ValueError("Every smoke shard report is required")
    actual = set()
    for number, report in shards.items():
        validate_report(report)
        if report["config"].get("shard") != {
            "current": number,
            "total": SHARD_COUNT,
        }:
            raise ValueError(f"Wrong shard configuration in report {number}")
        selected = identities(report)
        if actual & selected:
            raise ValueError("Test identities overlap across shards")
        actual.update(selected)
    if actual != expected:
        raise ValueError("Shard union differs from the independently discovered suite")
    return len(expected)


def verify_seed(discovery, report):
    expected = discovery_identities(discovery)
    validate_report(report, min_tests=SMOKE_TEST_FLOOR)
    if report["config"].get("shard") is not None or identities(report) != expected:
        raise ValueError("Fresh seed must pass the complete unsharded smoke suite")
    return len(expected)


def verify_directory(directory):
    expected_directories = {
        f"playwright-test-results-smoke-{number}"
        for number in range(1, SHARD_COUNT + 1)
    }
    if {path.name for path in directory.iterdir()} != expected_directories:
        raise ValueError("Missing or unexpected smoke shard artifacts")
    reports = {}
    discovery = None
    for number in range(1, SHARD_COUNT + 1):
        shard_dir = directory / f"playwright-test-results-smoke-{number}"
        manifests = list(shard_dir.rglob("smoke-inventory.json"))
        results = list(shard_dir.rglob("results.json"))
        if len(manifests) != 1 or len(results) != 1:
            raise ValueError(f"Missing or ambiguous reports for shard {number}")
        current = json.loads(manifests[0].read_text(encoding="utf-8"))
        current_ids = discovery_identities(current)
        if discovery is not None and current_ids != discovery_identities(discovery):
            raise ValueError("Runners discovered different test inventories")
        discovery = current
        reports[number] = json.loads(results[0].read_text(encoding="utf-8"))
    return verify(discovery, reports)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("shards").add_argument("reports", type=Path)
    seed = commands.add_parser("seed")
    seed.add_argument("inventory", type=Path)
    seed.add_argument("report", type=Path)
    args = parser.parse_args()
    if args.command == "seed":
        count = verify_seed(
            json.loads(args.inventory.read_text(encoding="utf-8")),
            json.loads(args.report.read_text(encoding="utf-8")),
        )
    else:
        count = verify_directory(args.reports)
    print(
        f"Smoke reports cover all {count} discovered tests once, without skips or retries"
    )


if __name__ == "__main__":
    main()
