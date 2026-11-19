import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from xml.etree import ElementTree


def run_tests(
    paths: list[Path], label: str, output: Path, allow_block_skips: bool = False
) -> dict:
    report = output / f"junit-{label}.xml"
    env = {
        **os.environ,
        "COVERAGE_FILE": str(output / f".coverage-{label}"),
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
                "no:cacheprovider",
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
        print((output / f"console-{label}.log").read_text())
        raise SystemExit(result.returncode)
    cases = list(ElementTree.parse(report).iter("testcase"))
    allowed = (
        set(
            json.loads(
                Path("../../.github/scripts/backend-allowed-skips.json").read_text()
            )["common"]
        )
        if allow_block_skips
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
        raise RuntimeError("Missing, failed or skipped architecture case")
    return {
        "seconds": seconds,
        "cases": sorted((case.get("classname"), case.get("name")) for case in cases),
        "skipped": sorted(skipped),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--order", choices=["baseline-first", "candidate-first"], required=True
    )
    args = parser.parse_args()
    backend = Path.cwd()
    target = backend / "backend/util/architecture_test.py"
    candidate = backend / "../../.github/benchmarks/architecture-prefilter-20261119.py"
    regression = backend / "backend/util/architecture_prefilter_regression_test.py"
    assert not regression.exists()
    original = target.read_bytes()
    regression.write_bytes(
        (
            backend
            / "../../.github/benchmarks/architecture_prefilter_regression_test.py"
        ).read_bytes()
    )
    output = Path(os.environ["RUNNER_TEMP"]) / "architecture-prefilter"
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
            results[label] = run_tests(
                [
                    target,
                    backend / "backend/util/architecture_scan_test.py",
                    regression,
                ],
                label,
                output,
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
        target.write_bytes(candidate.read_bytes())
        validation = run_tests(
            [
                *sorted((backend / "backend/util").glob("architecture*_test.py")),
                backend / "backend/blocks/test/test_block.py",
            ],
            "candidate-full",
            output,
            allow_block_skips=True,
        )
        (output / "validation.json").write_text(json.dumps(validation, indent=2))
    finally:
        target.write_bytes(original)
        regression.unlink()


if __name__ == "__main__":
    main()
