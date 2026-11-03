import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class BenchmarkReportTests(unittest.TestCase):
    def test_cli_rejects_version_source_editability_and_package_changes(self):
        script = Path(__file__).with_name("benchmark_backend_tools.py")
        package = {
            "version": "1.0",
            "url": "git+https://example.test/project",
            "commit": "abc",
            "editable": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            baseline, candidate = (
                Path(directory) / name for name in ("before.json", "after.json")
            )
            baseline.write_text(
                json.dumps({"package": package, "pip": {"version": "25"}})
            )
            variants = [
                ({"package": package}, 0),
                ({"package": {**package, "version": "2.0"}}, 1),
                ({"package": {**package, "commit": "def"}}, 1),
                ({"package": {**package, "url": "https://example.test/other"}}, 1),
                ({"package": {**package, "editable": True}}, 1),
                ({}, 1),
                ({"package": package, "unexpected": package}, 1),
            ]
            for packages, expected in variants:
                with self.subTest(packages=packages):
                    candidate.write_text(json.dumps(packages))
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(script),
                            "compare",
                            "--baseline",
                            str(baseline),
                            "--candidate",
                            str(candidate),
                        ],
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(result.returncode, expected, result.stderr)
                    self.assertEqual(bool(json.loads(result.stdout)), bool(expected))

    def test_refuses_local_dependency_installation(self):
        environment = {
            key: value for key, value in os.environ.items() if key != "GITHUB_ACTIONS"
        }
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("benchmark_backend_tools.py")),
                "run",
            ],
            env=environment,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("hosted Linux CI", result.stderr)


if __name__ == "__main__":
    unittest.main()
