import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree

import backend_test_shard
from backend_test_shard import SHARDS, shard_args


class BackendShardTests(unittest.TestCase):
    def test_all_test_locations_are_assigned_once_with_backend_config(self):
        paths = [
            "backend/data/sample_unit_test.py",
            "backend/copilot/sample_unit_test.py",
            "backend/util/sample_unit_test.py",
            "backend/executor/sample_unit_test.py",
            "backend/blocks/sample_unit_test.py",
            "backend/new_folder/sample_unit_test.py",
            "backend/new_unit_test.py",
            "new_top_level/sample_unit_test.py",
            "root_unit_test.py",
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = root / "autogpt_platform/backend"
            backend.mkdir(parents=True)
            (backend / "pyproject.toml").write_text(
                '[tool.pytest.ini_options]\npython_files = ["*_unit_test.py"]\n'
            )
            helpers = root / ".github/scripts"
            helpers.mkdir(parents=True)
            (helpers / "helper_unit_test.py").write_text("def test_helper(): pass\n")
            for index, path in enumerate(paths):
                target = backend / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(f"def test_case_{index}(): pass\n")
            names = []
            for shard in ("data", "copilot", "util-executor", "remainder"):
                report = root / f"{shard}.xml"
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        "--import-mode=importlib",
                        *shard_args(shard),
                        f"--junitxml={report}",
                        "-q",
                    ],
                    cwd=backend,
                    env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                names.extend(
                    case.get("name")
                    for case in ElementTree.parse(report).iter("testcase")
                )
            self.assertCountEqual(
                names, [f"test_case_{i}" for i in range(len(paths))] + ["test_helper"]
            )

    def test_unknown_shard_is_rejected(self):
        with self.assertRaises(ValueError):
            shard_args("typo")

    def test_workflow_matrix_matches_script_shards(self):
        workflow = (
            Path(__file__).resolve().parents[1] / "workflows/platform-backend-ci.yml"
        )
        line = next(
            line
            for line in workflow.read_text().splitlines()
            if line.strip().startswith("test-shard: [")
        )
        matrix = [
            shard.strip() for shard in line.split("[", 1)[1].rstrip("] ").split(",")
        ]
        self.assertCountEqual(matrix, [*SHARDS, "remainder"])

    def test_main_propagates_pytest_status_and_scopes_docker_tests_to_their_shard(
        self,
    ):
        flags = ("E2E_REDIS_CLUSTER_RESTART", "E2E_CLAMD_CONTAINER")
        for shard, enabled in (
            ("data", {"E2E_REDIS_CLUSTER_RESTART"}),
            ("util-executor", {"E2E_CLAMD_CONTAINER"}),
            ("copilot", set()),
            ("remainder", set()),
        ):
            with (
                self.subTest(shard=shard),
                patch.object(
                    backend_test_shard.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess([], 3),
                ) as run,
                patch.dict(os.environ),
                patch.object(sys, "argv", ["backend_test_shard.py", shard, "--", "-q"]),
            ):
                for flag in flags:
                    os.environ.pop(flag, None)
                self.assertEqual(backend_test_shard.main(), 3)
                self.assertEqual(
                    {f for f in flags if os.environ.get(f) == "1"}, enabled
                )
                run.assert_called_once_with(
                    ["poetry", "run", "pytest", *shard_args(shard), "-q"], check=False
                )
