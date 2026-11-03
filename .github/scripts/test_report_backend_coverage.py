import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree


class BackendCoverageReportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.reporter = Path(__file__).with_name("report_backend_coverage.py")
        backend = self.root / "backend"
        backend.mkdir()
        (backend / "__init__.py").touch()
        (backend / "sample.py").write_text(
            "def branch(value):\n"
            "    if value:\n"
            "        return 1\n"
            "    return 0\n"
            "\n"
            "def excluded():  # pragma: no cover\n"
            "    return 5\n"
        )
        (backend / "unexecuted.py").write_text("value = 2\n")
        (backend / "omitted.py").write_text("value = 3\n")
        (backend / "syntax_fixture.py").write_text("invalid syntax :=\n")
        (self.root / "test_sample.py").write_text(
            "from backend.sample import branch\n"
            "def test_branch():\n"
            "    assert branch(True) == 1\n"
        )
        (self.root / "pyproject.toml").write_text(
            '[tool.coverage.run]\nomit = ["backend/omitted.py"]\n'
        )
        self.env = {
            **os.environ,
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "COVERAGE_FILE": str(self.root / ".coverage"),
        }
        for key in ("PYTEST_ADDOPTS", "COVERAGE_PROCESS_START", "COVERAGE_RCFILE"):
            self.env.pop(key, None)

    def run_command(self, *args):
        return subprocess.run(
            [sys.executable, *map(str, args)],
            cwd=self.root,
            env=self.env,
            capture_output=True,
            text=True,
        )

    def run_tests(self, report, branch=True):
        result = self.run_command(
            "-m",
            "pytest",
            "-p",
            "pytest_cov",
            "--cov=backend",
            *(["--cov-branch"] if branch else []),
            f"--cov-report={report}",
            "-q",
            "test_sample.py",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def report(self, data_file=".coverage"):
        return self.run_command(
            self.reporter, "--data-file", data_file, "--output", "actual.xml"
        )

    def test_deferred_xml_preserves_all_lines_branches_and_unexecuted_files(self):
        self.run_tests("xml:expected.xml")
        result = self.run_tests("")
        self.assertNotIn("coverage:", result.stdout)
        report = self.report()
        self.assertEqual(report.returncode, 0, report.stdout + report.stderr)
        expected = ElementTree.canonicalize(
            from_file=self.root / "expected.xml",
            strip_text=True,
            exclude_attrs={"timestamp"},
        )
        actual = ElementTree.canonicalize(
            from_file=self.root / "actual.xml",
            strip_text=True,
            exclude_attrs={"timestamp"},
        )
        self.assertEqual(actual, expected)
        classes = {
            item.get("filename"): item
            for item in ElementTree.parse(self.root / "actual.xml").iter("class")
        }
        self.assertEqual(classes["unexecuted.py"].get("line-rate"), "0")
        self.assertNotIn("omitted.py", classes)
        self.assertNotIn("syntax_fixture.py", classes)
        self.assertIn('condition-coverage="50% (1/2)"', actual)

    def test_combines_suffixed_data_from_pytest_cov(self):
        self.run_tests("")
        (self.root / ".coverage").rename(self.root / ".coverage.worker.combine")
        result = self.report()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((self.root / ".coverage").is_file())

    def test_rejects_missing_branch_data(self):
        result = self.report()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("branch coverage data is missing", result.stderr)
        self.assertFalse((self.root / "actual.xml").exists())

    def test_rejects_line_only_data(self):
        self.run_tests("", branch=False)
        result = self.report()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("branch coverage data is missing", result.stderr)
        self.assertFalse((self.root / "actual.xml").exists())


if __name__ == "__main__":
    unittest.main()
