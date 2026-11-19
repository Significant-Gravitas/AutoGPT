import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from xml.etree import ElementTree

from coverage import CoverageData


def run_tests(paths, label, output, allow_skips=False):
    report = output / f"junit-{label}.xml"
    resource = output / f"resources-{label}.txt"
    log_path = output / f"console-{label}.log"
    env = {
        **os.environ,
        "BENCHMARK_DISABLE_STARTUP_GC": "0" if label == "baseline" else "1",
        "BACKEND_TEST_STARTUP_GC_METRICS": str(output / f"gc-{label}.json"),
        "COVERAGE_FILE": str(output / f".coverage-{label}"),
        "BACKEND_TEST_SHARD": "util-executor",
        "BACKEND_TEST_LAYOUT": "paths",
        "BACKEND_TEST_PARTITION": "0",
        "BACKEND_TEST_METRICS": str(output / f"metrics-{label}.json"),
        "BACKEND_TEST_LAUNCH_TIME": str(time.perf_counter()),
        "PYTHONPATH": os.pathsep.join(
            (
                str(Path("../../.github/benchmarks").resolve()),
                str(Path("../../.github/scripts").resolve()),
                os.environ.get("PYTHONPATH", ""),
            )
        ),
    }
    started = time.perf_counter()
    with log_path.open("w") as log:
        result = subprocess.run(
            [
                "/usr/bin/time",
                "-v",
                "-o",
                str(resource),
                "poetry",
                "run",
                "pytest",
                "-p",
                "no:cacheprovider",
                "-p",
                "startup_gc_probe",
                "-p",
                "backend_shard_plugin",
                "-c",
                "pyproject.toml",
                *(str(path) for path in paths),
                "--cov=backend",
                "--cov-branch",
                "--cov-report=",
                f"--junitxml={report}",
                "--durations=30",
                "-q",
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    seconds = time.perf_counter() - started
    if result.returncode:
        print(log_path.read_text())
        raise SystemExit(result.returncode)
    cases = list(ElementTree.parse(report).iter("testcase"))
    allowed = (
        set(
            json.loads(
                Path("../../.github/scripts/backend-allowed-skips.json").read_text()
            )["common"]
        )
        if allow_skips
        else set()
    )
    skipped = {
        f"{case.get('classname')}.{case.get('name')}"
        for case in cases
        if case.find("skipped") is not None
    }
    if (
        not cases
        or skipped - allowed
        or any(
            case.find(status) is not None
            for case in cases
            for status in ("failure", "error")
        )
    ):
        raise RuntimeError("Missing, failed or unexpectedly skipped cases")
    memory = next(
        int(line.rsplit(":", 1)[1].strip())
        for line in resource.read_text().splitlines()
        if "Maximum resident set size" in line
    )
    return {
        "seconds": seconds,
        "peak_rss_kib": memory,
        "cases": sorted((case.get("classname"), case.get("name")) for case in cases),
        "skipped": sorted(skipped),
        "startup_gc": json.loads((output / f"gc-{label}.json").read_text()),
    }


def compare_coverage(output):
    coverage = {}
    for label in ("baseline", "candidate"):
        data = CoverageData(basename=str(output / f".coverage-{label}"))
        data.read()
        if not data.has_arcs():
            raise RuntimeError("Missing branch coverage")
        coverage[label] = {
            name: set(data.arcs(name) or []) for name in data.measured_files()
        }
    differences = []
    for name in sorted(coverage["baseline"].keys() | coverage["candidate"].keys()):
        before = coverage["baseline"].get(name, set())
        after = coverage["candidate"].get(name, set())
        if before != after or any(name not in value for value in coverage.values()):
            differences.append(
                {
                    "file": name,
                    "lost_arcs": sorted(before - after),
                    "added_arcs": sorted(after - before),
                }
            )
    (output / "coverage-diff.json").write_text(json.dumps(differences, indent=2))
    return differences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--order", choices=["baseline-first", "candidate-first"], required=True
    )
    args = parser.parse_args()
    output = Path(os.environ["RUNNER_TEMP"]) / "startup-gc"
    output.mkdir()
    labels = ["baseline", "candidate"]
    if args.order == "candidate-first":
        labels.reverse()
    results = {
        label: run_tests([Path("backend/util/bounded_yaml_test.py")], label, output)
        for label in labels
    }
    if any(len(value["cases"]) != 17 for value in results.values()):
        raise RuntimeError("Expected all 17 bounded YAML cases")
    if results["baseline"]["cases"] != results["candidate"]["cases"]:
        raise RuntimeError("Test inventories differ")
    differences = compare_coverage(output)
    results["coverage_changed_files"] = len(differences)
    (output / "comparison.json").write_text(json.dumps(results, indent=2))
    print(
        "::notice title=startup-gc-comparison::"
        + json.dumps(
            {
                label: {
                    "seconds": results[label]["seconds"],
                    "peak_rss_kib": results[label]["peak_rss_kib"],
                    "startup_seconds": results[label]["startup_gc"]["seconds"],
                    "startup_gc_seconds": results[label]["startup_gc"]["gc_seconds"],
                }
                for label in labels
            }
        )
    )
    validation = run_tests(
        [
            *sorted(Path("backend/util").glob("architecture*_test.py")),
            Path("backend/blocks/test/test_block.py"),
        ],
        "candidate-full",
        output,
        allow_skips=True,
    )
    (output / "validation.json").write_text(json.dumps(validation, indent=2))
    if differences:
        raise RuntimeError("Branch coverage parity needs investigation")


if __name__ == "__main__":
    main()
