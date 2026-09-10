import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree

from backend_test_shard import shard_args


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
