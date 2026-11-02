import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree

from backend_shard_weights import assign_files, load_weights
from backend_test_shard import SHARDS
from update_backend_test_durations import weights_from_reports


class WeightedShardTests(unittest.TestCase):
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
