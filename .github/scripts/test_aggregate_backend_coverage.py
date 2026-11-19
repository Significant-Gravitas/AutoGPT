import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree

from backend_test_shard import report_shards
from coverage import CoverageData


class AggregateBackendCoverageTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.reports = self.root / "reports"
        backend = self.root / "backend"
        backend.mkdir()
        (backend / "__init__.py").touch()
        self.sample = backend / "sample.py"
        self.sample.write_text(
            "def choose(value):\n"
            "    if value:\n"
            "        return 1\n"
            "    return 0\n"
        )
        self.unexecuted = backend / "unexecuted.py"
        self.unexecuted.write_text("value = 5\n")
        self.policy = self.root / "policy.json"
        self.policy.write_text(
            json.dumps({"common": [], "python_versions": {"3.12": [], "3.13": []}})
        )
        for version in ("3.12", "3.13"):
            for shard in report_shards(partitioned=True):
                report = self.reports / f"backend-test-reports-py{version}-{shard}"
                report.mkdir(parents=True)
                (report / f"shard-metrics-{shard}.json").write_text(
                    json.dumps(
                        {"source_sha": "tested-sha", "python": version, "exit_code": 0}
                    )
                )
                data = CoverageData(basename=str(report / f".coverage-{shard}"))
                branch = 3 if version == "3.12" else 4
                data.add_arcs({str(self.sample): [(-1, 2), (2, branch), (branch, -1)]})
                data.touch_file(str(self.unexecuted))
                data.write()
        self.first = sorted(self.reports.iterdir())[0]
        self.first_data = next(self.first.glob(".coverage-*"))
        self.first_metrics = next(self.first.glob("shard-metrics-*.json"))

    def run_aggregate(self):
        env = dict(os.environ)
        for key in ("COVERAGE_FILE", "COVERAGE_PROCESS_START", "COVERAGE_RCFILE"):
            env.pop(key, None)
        return subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("aggregate_backend_coverage.py")),
                "--reports-dir",
                str(self.reports),
                "--allow-skips-from",
                str(self.policy),
                "--source-sha",
                "tested-sha",
                "--data-file",
                ".coverage-complete",
                "--output",
                "coverage.xml",
            ],
            cwd=self.root,
            env=env,
            capture_output=True,
            text=True,
        )

    def assert_rejected(self, message):
        result = self.run_aggregate()
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn(message, result.stderr)
        self.assertFalse((self.root / "coverage.xml").exists())

    def test_complete_matrix_combines_complementary_branches_and_unexecuted_files(self):
        result = self.run_aggregate()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads(result.stdout.splitlines()[-1])
        self.assertEqual(summary["reports"], len(report_shards(True)) * 2)
        classes = {
            item.attrib["filename"]: item
            for item in ElementTree.parse(self.root / "coverage.xml").iter("class")
        }
        branch = classes["sample.py"].find("lines/line[@number='2']")
        self.assertEqual(branch.attrib["condition-coverage"], "100% (2/2)")
        self.assertEqual(classes["unexecuted.py"].attrib["line-rate"], "0")
        self.assertTrue(self.first_data.is_file())

    def test_rejects_missing_partition(self):
        shutil.rmtree(self.first)
        self.assert_rejected("incomplete coverage artifacts")

    def test_rejects_extra_partition(self):
        (self.reports / "backend-test-reports-py3.13-unknown-1").mkdir()
        self.assert_rejected("incomplete coverage artifacts")

    def test_rejects_missing_raw_data_in_otherwise_complete_reports(self):
        self.first_data.unlink()
        self.assert_rejected("expected exactly one finalized coverage file")

    def test_rejects_uncombined_worker_data(self):
        shutil.copyfile(self.first_data, str(self.first_data) + ".worker")
        self.assert_rejected("expected exactly one finalized coverage file")

    def test_rejects_corrupt_raw_data(self):
        self.first_data.write_bytes(b"invalid SQLite data")
        self.assert_rejected("Couldn't use data file")

    def test_rejects_line_only_data(self):
        self.first_data.unlink()
        data = CoverageData(basename=str(self.first_data))
        data.add_lines({str(self.sample): [1, 2, 3]})
        data.write()
        self.assert_rejected("branch coverage data is missing")

    def test_rejects_mismatched_revision_python_and_failed_execution(self):
        original = json.loads(self.first_metrics.read_text())
        for key, value, message in (
            ("source_sha", "old-sha", "source revision"),
            ("python", "3.11", "Python version"),
            ("exit_code", 1, "test execution did not succeed"),
        ):
            with self.subTest(key=key):
                self.first_metrics.write_text(json.dumps({**original, key: value}))
                self.assert_rejected(message)

    def test_rejects_stale_output(self):
        (self.root / ".coverage-complete").write_bytes(b"stale data")
        self.assert_rejected("refusing stale data")


if __name__ == "__main__":
    unittest.main()
