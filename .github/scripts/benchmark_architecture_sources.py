import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from xml.etree import ElementTree


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--order", choices=["baseline-first", "candidate-first"], required=True
    )
    args = parser.parse_args()
    backend = Path.cwd()
    scripts = backend / "../../.github/scripts"
    candidate = backend / "../../.github/benchmarks/architecture-cached.py"
    target = backend / "backend/util/architecture_test.py"
    original = target.read_bytes()
    output = Path(os.environ["RUNNER_TEMP"]) / "architecture-comparison"
    output.mkdir()
    labels = ["baseline", "candidate"]
    if args.order == "candidate-first":
        labels.reverse()
    results = {}
    try:
        for label in labels:
            target.write_bytes(
                original if label == "baseline" else candidate.read_bytes()
            )
            report = output / f"junit-{label}.xml"
            env = {
                **os.environ,
                "BACKEND_TEST_SHARD": "util-executor",
                "BACKEND_TEST_LAYOUT": "paths",
                "BACKEND_TEST_PARTITION": "0",
                "BACKEND_TEST_LAUNCH_TIME": str(time.perf_counter()),
                "BACKEND_TEST_METRICS": str(output / f"metrics-{label}.json"),
                "COVERAGE_FILE": str(output / f".coverage-{label}"),
                "PYTHONPATH": str(scripts),
            }
            started = time.perf_counter()
            with (output / f"console-{label}.log").open("w") as log:
                result = subprocess.run(
                    [
                        "/usr/bin/time",
                        "-v",
                        "-o",
                        str(output / f"resources-{label}.txt"),
                        "poetry",
                        "run",
                        "pytest",
                        "-p",
                        "backend_shard_plugin",
                        "-p",
                        "no:cacheprovider",
                        "-c",
                        "pyproject.toml",
                        str(target),
                        "--cov=backend",
                        "--cov-branch",
                        "--cov-report=",
                        f"--junitxml={report}",
                        "--durations=15",
                        "-q",
                    ],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if result.returncode:
                print((output / f"console-{label}.log").read_text())
                raise SystemExit(result.returncode)
            tree = ElementTree.parse(report)
            cases = list(tree.iter("testcase"))
            if not cases or any(list(case) for case in cases):
                raise RuntimeError("Missing, failed or skipped architecture case")
            results[label] = {
                "seconds": time.perf_counter() - started,
                "cases": sorted(
                    (case.get("classname"), case.get("name")) for case in cases
                ),
            }
            probe = """
import tempfile
from pathlib import Path
from backend.util.architecture_test import _dataclass_imports
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / 'edited.py'
    path.write_text('value = 1\\n')
    assert _dataclass_imports(path) == []
    path.write_text('import dataclasses\\n')
    assert _dataclass_imports(path), 'new violation was hidden'
    path.write_text('value = 2\\n')
    assert _dataclass_imports(path) == [], 'removed violation was retained'
    path.write_text('invalid syntax :=\\n')
    assert _dataclass_imports(path) == []
    path.unlink()
    try:
        _dataclass_imports(path)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError('deleted source was reused')
"""
            subprocess.run(
                ["poetry", "run", "python", "-c", probe], env=env, check=True
            )
        if results["baseline"]["cases"] != results["candidate"]["cases"]:
            raise RuntimeError("Architecture case inventories differ")
        (output / "comparison.json").write_text(json.dumps(results, indent=2))
        print(
            json.dumps(
                {
                    label: {"seconds": value["seconds"], "cases": len(value["cases"])}
                    for label, value in results.items()
                }
            )
        )
    finally:
        target.write_bytes(original)


if __name__ == "__main__":
    main()
