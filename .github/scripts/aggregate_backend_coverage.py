#!/usr/bin/env python3

import argparse
import json
import sys
import time
from pathlib import Path
from xml.etree import ElementTree

from backend_test_shard import report_shards
from coverage import Coverage, CoverageData
from coverage.exceptions import CoverageException
from report_backend_coverage import parallel_xml
from validate_cobertura import validate_report
from validate_junit import load_skip_policy


def coverage_inputs(
    reports: Path, versions: tuple[str, ...], source_sha: str
) -> list[Path]:
    expected = {
        f"backend-test-reports-py{version}-{shard}": (version, shard)
        for version in versions
        for shard in report_shards(partitioned=True)
    }
    actual = {path.name for path in reports.glob("backend-test-reports-py*")}
    if actual != set(expected):
        raise ValueError(
            f"incomplete coverage artifacts; missing: {sorted(set(expected) - actual)}; "
            f"unexpected: {sorted(actual - set(expected))}"
        )
    inputs = []
    for name, (version, shard) in sorted(expected.items()):
        directory = reports / name
        metrics = json.loads(
            (directory / f"shard-metrics-{shard}.json").read_text(encoding="utf-8")
        )
        if metrics.get("source_sha") != source_sha:
            raise ValueError(
                f"{name}: coverage source revision does not match checkout"
            )
        if ".".join(metrics.get("python", "").split(".")[:2]) != version:
            raise ValueError(f"{name}: coverage Python version does not match artifact")
        if metrics.get("exit_code") != 0:
            raise ValueError(f"{name}: test execution did not succeed")
        data_file = directory / f".coverage-{shard}"
        if set(directory.glob(".coverage-*")) != {data_file}:
            raise ValueError(f"{name}: expected exactly one finalized coverage file")
        if not data_file.is_file() or data_file.stat().st_size == 0:
            raise ValueError(f"{name}: coverage data is missing or empty")
        data = CoverageData(basename=str(data_file))
        data.read()
        if not data.has_arcs() or not data.measured_files():
            raise ValueError(f"{name}: backend branch coverage data is missing")
        inputs.append(data_file.resolve())
    return inputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--allow-skips-from", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        started = time.perf_counter()
        policy = load_skip_policy(args.allow_skips_from)
        if not policy.python_versions:
            raise ValueError("coverage aggregation requires explicit Python versions")
        if args.data_file.exists() or args.output.exists():
            raise ValueError("coverage output already exists; refusing stale data")
        inputs = coverage_inputs(
            args.reports_dir, tuple(policy.python_versions), args.source_sha
        )
        coverage = Coverage(
            data_file=str(args.data_file), source=["backend"], branch=True
        )
        coverage.combine(
            data_paths=[str(path) for path in inputs], strict=True, keep=True
        )
        coverage.save()
        combined = time.perf_counter()
        parallel_xml(
            str(args.data_file),
            list(coverage.get_data().measured_files()),
            str(args.output),
        )
        root = ElementTree.parse(args.output).getroot()
        validate_report(root)
        print(
            json.dumps(
                {
                    "source_sha": args.source_sha,
                    "reports": len(inputs),
                    "combine_seconds": combined - started,
                    "xml_seconds": time.perf_counter() - combined,
                    "measured_files": len(coverage.get_data().measured_files()),
                    "lines_valid": int(root.attrib["lines-valid"]),
                    "lines_covered": int(root.attrib["lines-covered"]),
                },
                sort_keys=True,
            )
        )
    except (OSError, ValueError, CoverageException, ElementTree.ParseError) as error:
        print(f"Invalid backend coverage: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
