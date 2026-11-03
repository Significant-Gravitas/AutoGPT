import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree

from backend_shard_weights import (
    assign_files,
    load_weights,
    partition_files,
    path_shard,
)
from backend_test_shard import PARTITIONS, SHARDS, shard_args
from update_backend_test_durations import weights_from_reports


class WeightedShardTests(unittest.TestCase):
    def test_partition_ownership_is_stable_and_includes_unmeasured_files(self):
        files = [f"backend/copilot/{i}_test.py" for i in range(8)]
        weights = {files[0]: 20, files[1]: 10}
        result = partition_files(files, weights, 4)
        self.assertEqual(result, partition_files(list(reversed(files)), weights, 4))
        self.assertEqual(set(result), set(files))
        self.assertEqual(set(result.values()), {1, 2, 3, 4})
        self.assertNotEqual(result[files[0]], result[files[1]])
        self.assertEqual(path_shard("backend/copilot/sdk/new_test.py"), "sdk-billing")
        self.assertEqual(
            path_shard("backend/copilot/sdk_sibling/new_test.py"), "copilot"
        )
        self.assertEqual(path_shard("backend/new_area/new_test.py"), "remainder")

    def test_partitioned_pytest_executes_new_and_nested_cases_exactly_once(self):
        scripts = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = root / "autogpt_platform/backend"
            backend.mkdir(parents=True)
            (backend / "pyproject.toml").write_text(
                '[tool.pytest.ini_options]\npython_files = ["*_check.py", "*_test.py"]\n'
            )
            (backend / "conftest.py").write_text(
                'collect_ignore = ["ignored_check.py"]\n'
            )
            (backend / "ignored_check.py").write_text('raise RuntimeError("ignored")\n')
            for paths in SHARDS.values():
                for name in paths:
                    path = backend / name
                    (path.parent if path.suffix else path).mkdir(
                        parents=True, exist_ok=True
                    )
            files = {}
            for shard in SHARDS:
                if shard == "billing-threads":
                    names = [SHARDS[shard][0]] + [
                        f"backend/executor/billing_e2e/case_{i}_check.py"
                        for i in range(2)
                    ]
                else:
                    names = [
                        f"{SHARDS[shard][0]}/new_{i}/case_check.py" for i in range(4)
                    ]
                files.update({name: shard for name in names})
            files.update(
                {
                    "../../.github/scripts/helper_check.py": "remainder",
                    "new_top_level/root_check.py": "remainder",
                    "backend/new_area/new_check.py": "remainder",
                    "root_check.py": "remainder",
                    "new_top_level/another_check.py": "remainder",
                    "backend/another_area/new_check.py": "remainder",
                    "new_top_level/third_check.py": "remainder",
                    "backend/another_area/second_check.py": "remainder",
                }
            )
            for index, name in enumerate(files):
                target = backend / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(
                    "import pytest\n"
                    '@pytest.mark.parametrize("value", [1, 2])\n'
                    f"def test_case_{index}(value): assert value > 0\n"
                )
            weights = root / "weights.json"
            weights.write_text(
                json.dumps({"version": 1, "seconds": {next(iter(files)): 10}})
            )
            cases = []
            for shard, count in PARTITIONS.items():
                for part in range(1, count + 1):
                    report = root / f"{shard}-{part}.json"
                    result = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "pytest",
                            "-p",
                            "backend_shard_plugin",
                            "--import-mode=importlib",
                            *shard_args(shard),
                            "-q",
                        ],
                        cwd=backend,
                        env={
                            **os.environ,
                            "PYTHONPATH": str(scripts),
                            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                            "BACKEND_TEST_SHARD": shard,
                            "BACKEND_TEST_LAYOUT": "paths",
                            "BACKEND_TEST_PARTITION": str(part),
                            "BACKEND_TEST_WEIGHTS": str(weights),
                            "BACKEND_TEST_METRICS": str(report),
                        },
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(
                        result.returncode,
                        0,
                        f"{shard}-{part}: {result.stdout}{result.stderr}",
                    )
                    metrics = json.loads(report.read_text())
                    cases.extend(
                        node.rsplit("::", 1)[-1] for node in metrics["nodeids"]
                    )
                    for path, values in metrics["files"].items():
                        if values["cases"]:
                            self.assertEqual(files[path], shard)
            self.assertCountEqual(
                cases,
                [
                    f"test_case_{i}[{value}]"
                    for i in range(len(files))
                    for value in (1, 2)
                ],
            )

    def test_refresh_counts_collection_and_rejects_incomplete_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, shard in enumerate((*SHARDS, "remainder")):
                path = Path(directory) / f"shard-metrics-{shard}.json"
                path.write_text(
                    json.dumps(
                        {
                            "source_sha": "fixed-source",
                            "layout": "paths",
                            "exit_code": 0,
                            "python": "3.13.17",
                            "shard": shard,
                            "files": {
                                f"backend/{index}_test.py": {
                                    "cases": 1,
                                    "collection_seconds": 2,
                                    "test_seconds": 3,
                                }
                            },
                        }
                    )
                )
                paths.append(path)
            weights = weights_from_reports(paths)
            self.assertEqual(len(weights["seconds"]), 8)
            self.assertEqual(set(weights["seconds"].values()), {5})
            with self.assertRaisesRegex(ValueError, "every shard"):
                weights_from_reports(paths[:-1])
            with self.assertRaisesRegex(ValueError, "Duplicate shard"):
                weights_from_reports(paths + paths[:1])

    def test_balances_duration_without_moving_stateful_data_tests(self):
        weights = {
            "backend/data/redis_test.py": 200,
            "backend/copilot/heavy_test.py": 100,
            "backend/copilot/other_test.py": 95,
            "backend/api/fast_test.py": 5,
            "backend/api/tiny_test.py": 4,
        }
        owners = assign_files(list(reversed(weights)), weights, ("data", "a", "b"))
        self.assertEqual(owners["backend/data/redis_test.py"], "data")
        self.assertEqual(set(owners), set(weights))
        loads = {
            shard: sum(
                weights[path] for path, owner in owners.items() if owner == shard
            )
            for shard in ("a", "b")
        }
        self.assertLessEqual(max(loads.values()), 105)
        self.assertEqual(
            owners, assign_files(list(weights), weights, ("data", "a", "b"))
        )

    def test_refresh_accepts_complete_partition_reports_and_rejects_a_missing_part(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for shard, count in PARTITIONS.items():
                for part in range(1, count + 1):
                    path = Path(directory) / f"{shard}-{part}.json"
                    path.write_text(
                        json.dumps(
                            {
                                "source_sha": "fixed-source",
                                "layout": "paths",
                                "exit_code": 0,
                                "python": "3.13.17",
                                "shard": shard,
                                "partition": part,
                                "files": {
                                    f"backend/{shard}/{part}_test.py": {
                                        "cases": 2,
                                        "collection_seconds": 1,
                                        "test_seconds": 3,
                                    }
                                },
                            }
                        )
                    )
                    paths.append(path)
            weights = weights_from_reports(paths)
            self.assertEqual(len(weights["seconds"]), sum(PARTITIONS.values()))
            self.assertEqual(set(weights["seconds"].values()), {4})
            with self.assertRaisesRegex(ValueError, "every shard"):
                weights_from_reports(paths[:-1])

    def test_invalid_weights_fail_instead_of_silently_changing_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.json"
            for value in (-1, float("nan"), float("inf"), "1", True):
                with self.subTest(value=value):
                    path.write_text(
                        json.dumps({"version": 1, "seconds": {"a.py": value}})
                    )
                    with self.assertRaises(ValueError):
                        load_weights(path)

    def test_real_pytest_runs_every_case_once_and_preserves_collection_rules(self):
        scripts = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = root / "autogpt_platform/backend"
            backend.mkdir(parents=True)
            (backend / "pyproject.toml").write_text(
                '[tool.pytest.ini_options]\npython_files = ["*_check.py"]\n'
            )
            (backend / "conftest.py").write_text(
                'collect_ignore = ["ignored_check.py"]\n'
            )
            (backend / "ignored_check.py").write_text('raise RuntimeError("ignored")\n')
            files = (
                ["backend/data/redis_check.py"]
                + [f"backend/area_{i % 3}/case_{i}_check.py" for i in range(21)]
                + [
                    "../../.github/scripts/helper_check.py",
                    "new_top_level/root_check.py",
                ]
            )
            for index, name in enumerate(files):
                target = backend / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(
                    "import pytest\n"
                    '@pytest.mark.parametrize("value", [1, 2])\n'
                    f"def test_case_{index}(value): assert value > 0\n"
                )
            weights = root / "weights.json"
            weights.write_text(json.dumps({"version": 1, "seconds": {files[1]: 50}}))
            cases = []
            inventories = []
            for shard in (*SHARDS, "remainder"):
                junit = root / f"{shard}.xml"
                report = root / f"{shard}.json"
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        "-p",
                        "backend_shard_plugin",
                        "--import-mode=importlib",
                        "-c",
                        "pyproject.toml",
                        ".",
                        "../../.github/scripts",
                        f"--junitxml={junit}",
                        "-q",
                    ],
                    cwd=backend,
                    env={
                        **os.environ,
                        "PYTHONPATH": str(scripts),
                        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                        "BACKEND_TEST_SHARD": shard,
                        "BACKEND_TEST_LAYOUT": "weighted",
                        "BACKEND_TEST_PARTITION": "0",
                        "BACKEND_TEST_WEIGHTS": str(weights),
                        "BACKEND_TEST_METRICS": str(report),
                    },
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                rows = list(ElementTree.parse(junit).iter("testcase"))
                cases.extend(row.get("name") for row in rows)
                metrics = json.loads(report.read_text())
                inventories.extend(metrics["nodeids"])
                self.assertEqual(metrics["layout"], "weighted")
                self.assertEqual(metrics["exit_code"], 0)
                self.assertEqual(len(metrics["nodeids"]), len(rows))
                self.assertEqual(
                    sum(file["cases"] for file in metrics["files"].values()), len(rows)
                )
                if shard == "data":
                    self.assertEqual(len(rows), 2)
            expected = [
                f"test_case_{i}[{value}]" for i in range(len(files)) for value in (1, 2)
            ]
            self.assertCountEqual(cases, expected)
            self.assertEqual(len(inventories), len(set(inventories)))


if __name__ == "__main__":
    unittest.main()
