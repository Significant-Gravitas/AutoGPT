#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
from xml.etree import ElementTree

from backend_test_shard import SHARDS
from validate_junit import load_skip_policy, validate_path


def validate_shard_reports(
    directory: Path, version: str, allowed_skips: set[str]
) -> set[str]:
    observed = set()
    for shard in (*SHARDS, "remainder"):
        report = (
            directory
            / f"backend-test-reports-py{version}-{shard}"
            / f"junit-{shard}.xml"
        )
        try:
            validate_path(report, False, allowed_skips)
            root = ElementTree.parse(report).getroot()
        except (OSError, ElementTree.ParseError, ValueError) as exc:
            raise ValueError(f"Python {version}, shard {shard}: {exc}") from exc
        observed.update(
            f"{case.get('classname', '')}.{case.get('name', '')}"
            for case in root.iter("testcase")
            if case.find("skipped") is not None
        )
    return observed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-skips-from", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        policy = load_skip_policy(args.allow_skips_from)
        if policy.python_versions is None:
            raise ValueError("aggregate validation requires explicit python_versions")
        expected = {
            f"backend-test-reports-py{version}-{shard}"
            for version in policy.python_versions
            for shard in (*SHARDS, "remainder")
        }
        actual = {
            path.name for path in args.reports_dir.glob("backend-test-reports-py*")
        }
        if expected != actual:
            raise ValueError(
                f"incomplete shard artifacts; missing: {sorted(expected - actual)}; "
                f"unexpected: {sorted(actual - expected)}"
            )
        problems = []
        # No secrets in this job, so secret-gated IDs are accepted and left out
        # of the staleness sweep; the per-shard step is the strict check.
        for version in policy.python_versions:
            allowed = policy.for_python_version(version)
            try:
                observed = validate_shard_reports(
                    args.reports_dir, version, allowed | policy.secret_gated_ids()
                )
                if stale := allowed - observed:
                    raise ValueError(
                        f"Python {version} has stale skip IDs in {args.allow_skips_from}: "
                        f"{', '.join(sorted(stale))}. Remove obsolete allowances or "
                        "configure legitimate Python-version-specific skips explicitly."
                    )
            except ValueError as exc:
                problems.append(str(exc))
                continue
            print(
                f"Python {version}: all four shard reports match {len(allowed)} skip IDs"
            )
        if problems:
            raise ValueError("\n".join(problems))
    except (OSError, ValueError) as exc:
        print(f"Invalid backend skip reports: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
