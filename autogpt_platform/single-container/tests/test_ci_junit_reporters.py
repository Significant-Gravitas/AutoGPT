import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
COMMAND_REPORTER = REPOSITORY_ROOT / ".github/scripts/run_command_junit.py"
UNITTEST_REPORTER = REPOSITORY_ROOT / ".github/scripts/run_unittest_junit.py"


class CommandReporterTests(unittest.TestCase):
    def test_success_report_preserves_name_and_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "command.xml"
            result = subprocess.run(
                [
                    sys.executable,
                    str(COMMAND_REPORTER),
                    "--name",
                    "safe <command>",
                    "--classname",
                    "ci.command",
                    "--output",
                    str(output),
                    "--",
                    sys.executable,
                    "-c",
                    "print('ok')",
                ],
                check=False,
            )

            self.assertEqual(result.returncode, 0)
            suite = ET.parse(output).getroot()
            self.assertEqual(suite.attrib["failures"], "0")
            self.assertEqual(suite.find("testcase").attrib["name"], "safe <command>")

    def test_failure_report_preserves_child_exit_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "command.xml"
            result = subprocess.run(
                [
                    sys.executable,
                    str(COMMAND_REPORTER),
                    "--name",
                    "failure",
                    "--classname",
                    "ci.command",
                    "--output",
                    str(output),
                    "--",
                    sys.executable,
                    "-c",
                    "raise SystemExit(7)",
                ],
                check=False,
            )

            self.assertEqual(result.returncode, 7)
            suite = ET.parse(output).getroot()
            self.assertEqual(suite.attrib["failures"], "1")
            self.assertIsNotNone(suite.find("testcase/failure"))


class UnittestReporterTests(unittest.TestCase):
    def test_discovery_report_contains_every_test_id(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            test_directory = Path(temp_dir) / "tests"
            test_directory.mkdir()
            (test_directory / "test_sample.py").write_text(
                "import unittest\n"
                "class SampleTests(unittest.TestCase):\n"
                "    def test_first(self):\n"
                "        self.assertTrue(True)\n"
                "    def test_second(self):\n"
                "        self.assertEqual(2 + 2, 4)\n",
                encoding="utf-8",
            )
            output = Path(temp_dir) / "unittest.xml"

            result = subprocess.run(
                [
                    sys.executable,
                    str(UNITTEST_REPORTER),
                    "--start-directory",
                    str(test_directory),
                    "--pattern",
                    "test_*.py",
                    "--output",
                    str(output),
                ],
                check=False,
            )

            self.assertEqual(result.returncode, 0)
            suite = ET.parse(output).getroot()
            self.assertEqual(suite.attrib["tests"], "2")
            names = {case.attrib["name"] for case in suite.findall("testcase")}
            self.assertEqual(names, {"test_first", "test_second"})

    def test_failure_report_preserves_test_exit_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            test_directory = Path(temp_dir) / "tests"
            test_directory.mkdir()
            (test_directory / "test_failure.py").write_text(
                "import unittest\n"
                "class FailureTests(unittest.TestCase):\n"
                "    def test_failure(self):\n"
                "        self.fail('expected failure')\n",
                encoding="utf-8",
            )
            output = Path(temp_dir) / "unittest.xml"

            result = subprocess.run(
                [
                    sys.executable,
                    str(UNITTEST_REPORTER),
                    "--start-directory",
                    str(test_directory),
                    "--output",
                    str(output),
                ],
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            suite = ET.parse(output).getroot()
            self.assertEqual(suite.attrib["failures"], "1")
            self.assertIsNotNone(suite.find("testcase/failure"))

    def test_class_fixture_error_is_recorded(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            test_directory = Path(temp_dir) / "tests"
            test_directory.mkdir()
            (test_directory / "test_fixture_error.py").write_text(
                "import unittest\n"
                "class FixtureErrorTests(unittest.TestCase):\n"
                "    @classmethod\n"
                "    def setUpClass(cls):\n"
                "        raise RuntimeError('fixture failed')\n"
                "    def test_never_runs(self):\n"
                "        pass\n",
                encoding="utf-8",
            )
            output = Path(temp_dir) / "unittest.xml"

            result = subprocess.run(
                [
                    sys.executable,
                    str(UNITTEST_REPORTER),
                    "--start-directory",
                    str(test_directory),
                    "--output",
                    str(output),
                ],
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            suite = ET.parse(output).getroot()
            self.assertEqual(suite.attrib["tests"], "1")
            self.assertEqual(suite.attrib["errors"], "1")
            self.assertIsNotNone(suite.find("testcase/error"))


if __name__ == "__main__":
    unittest.main()
