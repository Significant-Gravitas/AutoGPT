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
        nested = backend / "nested"
        nested.mkdir()
        (nested / "__init__.py").touch()
        (nested / "executed.py").write_text(
            "def branch(value):\n"
            "    if value:\n"
            "        return 7\n"
            "    return 9\n"
        )
        (nested / "unexecuted.py").write_text("value = 4\n")
        (nested / "excluded.py").write_text("value = 5  # pragma: no cover\n")
        (self.root / "test_sample.py").write_text(
            "from backend.sample import branch\n"
            "from backend.nested.executed import branch as nested_branch\n"
            "def test_branch():\n"
            "    assert branch(True) == 1\n"
            "    assert nested_branch(False) == 9\n"
        )
        (self.root / "pyproject.toml").write_text(
            '[tool.coverage.run]\nsource = ["backend"]\n'
            'omit = ["backend/omitted.py"]\n'
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

    def test_parallel_xml_preserves_all_lines_branches_packages_and_unexecuted_files(
        self,
    ):
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
        self.assertEqual(classes["nested/unexecuted.py"].get("line-rate"), "0")
        self.assertEqual(classes["nested/excluded.py"].get("line-rate"), "1")
        self.assertNotIn("omitted.py", classes)
        self.assertNotIn("syntax_fixture.py", classes)
        self.assertIn('condition-coverage="50% (1/2)"', actual)

    def test_combined_workers_preserve_complementary_branches(self):
        self.run_tests("xml:expected.xml")
        (self.root / ".coverage").rename(self.root / ".coverage.first")
        (self.root / "test_sample.py").write_text(
            "from backend.sample import branch\n"
            "from backend.nested.executed import branch as nested_branch\n"
            "def test_branch():\n"
            "    assert branch(False) == 0\n"
            "    assert nested_branch(True) == 7\n"
        )
        self.run_tests("")
        (self.root / ".coverage").rename(self.root / ".coverage.second")
        combined = self.run_command("-m", "coverage", "combine", "--keep")
        self.assertEqual(combined.returncode, 0, combined.stdout + combined.stderr)
        expected = self.run_command("-m", "coverage", "xml", "-i", "-o", "expected.xml")
        self.assertEqual(expected.returncode, 0, expected.stdout + expected.stderr)
        (self.root / ".coverage").unlink()
        result = self.report()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            ElementTree.canonicalize(
                from_file=self.root / "actual.xml",
                strip_text=True,
                exclude_attrs={"timestamp"},
            ),
            ElementTree.canonicalize(
                from_file=self.root / "expected.xml",
                strip_text=True,
                exclude_attrs={"timestamp"},
            ),
        )

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
