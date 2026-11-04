import runpy
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
COMMAND_REPORTER = REPOSITORY_ROOT / ".github/scripts/run_command_junit.py"
UNITTEST_REPORTER = REPOSITORY_ROOT / ".github/scripts/run_unittest_junit.py"


class XMLSafeTests(unittest.TestCase):
    def test_xml_character_range_boundaries(self):
        xml_safe = runpy.run_path(str(UNITTEST_REPORTER))["xml_safe"]
        allowed = "\t\n\r\x20\ud7ff\ue000\ufffd\U00010000\U0010ffff"
        forbidden = "\x00\x08\x0b\x0c\x0e\x1f\ud800\udfff\ufffe\uffff"
        self.assertEqual(xml_safe(allowed + forbidden), allowed)
        root = ET.Element("failure")
        root.text = xml_safe(allowed + forbidden)
        parsed = ET.fromstring(ET.tostring(root))
        self.assertEqual(parsed.text, allowed.replace("\r", "\n"))


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

    def test_spawn_error_is_recorded_as_machine_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "command.xml"
            result = subprocess.run(
                [
                    sys.executable,
                    str(COMMAND_REPORTER),
                    "--name",
                    "spawn-error",
                    "--classname",
                    "ci.command",
                    "--output",
                    str(output),
                    "--",
                    str(Path(temp_dir) / "missing-command"),
                ],
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            suite = ET.parse(output).getroot()
            self.assertEqual(suite.attrib["failures"], "0")
            self.assertEqual(suite.attrib["errors"], "1")
            self.assertIsNotNone(suite.find("testcase/error"))


class ReportWriteFailureTests(unittest.TestCase):
    def test_report_write_errors_are_diagnostic_and_preserve_failure_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tests = root / "tests"
            tests.mkdir()
            (tests / "test_healthy.py").write_text(
                "import unittest\n"
                "class Healthy(unittest.TestCase):\n"
                "    def test_passes(self): pass\n",
                encoding="utf-8",
            )
            existing_directory = root / "directory.xml"
            existing_directory.mkdir()
            existing_file = root / "not-a-directory"
            existing_file.write_text("occupied", encoding="utf-8")
            cases = [
                (
                    "command-success",
                    COMMAND_REPORTER,
                    ["--name", "command", "--classname", "ci"],
                    ["--", sys.executable, "-c", "pass"],
                    1,
                ),
                (
                    "command-failure",
                    COMMAND_REPORTER,
                    ["--name", "command", "--classname", "ci"],
                    ["--", sys.executable, "-c", "raise SystemExit(7)"],
                    7,
                ),
                (
                    "command-spawn-error",
                    COMMAND_REPORTER,
                    ["--name", "command", "--classname", "ci"],
                    ["--", str(root / "missing-command")],
                    127,
                ),
                (
                    "unittest-success",
                    UNITTEST_REPORTER,
                    ["--start-directory", str(tests)],
                    [],
                    1,
                ),
                (
                    "unittest-discovery-error",
                    UNITTEST_REPORTER,
                    ["--start-directory", str(root / "missing-tests")],
                    [],
                    1,
                ),
            ]
            for name, reporter, args, command, status in cases:
                for output in [existing_directory, existing_file / "report.xml"]:
                    with self.subTest(name=name, output=str(output)):
                        result = subprocess.run(
                            [
                                sys.executable,
                                str(reporter),
                                *args,
                                "--output",
                                str(output),
                                *command,
                            ],
                            capture_output=True,
                            text=True,
                            check=False,
                        )
                        self.assertEqual(result.returncode, status, result.stderr)
                        self.assertIn("Failed to write JUnit report", result.stderr)
                        self.assertIn(str(output), result.stderr)
                        self.assertNotIn("Traceback", result.stderr)


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
                "        pass\n"
                "class Healthy(unittest.TestCase):\n"
                "    def test_passes(self):\n"
                "        self.assertTrue(True)\n",
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
            self.assertEqual(suite.attrib["tests"], "2")
            self.assertEqual(suite.attrib["errors"], "1")
            self.assertIn("fixture failed", suite.find("testcase/error").text)
            self.assertIn(
                "test_passes",
                {case.attrib["name"] for case in suite.findall("testcase")},
            )

    def test_zero_discovered_tests_is_a_machine_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            test_directory = Path(temp_dir) / "tests"
            test_directory.mkdir()
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

    def test_discovery_exception_is_a_machine_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "unittest.xml"
            result = subprocess.run(
                [
                    sys.executable,
                    str(UNITTEST_REPORTER),
                    "--start-directory",
                    str(Path(temp_dir) / "missing"),
                    "--output",
                    str(output),
                ],
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            suite = ET.parse(output).getroot()
            self.assertEqual(suite.attrib["errors"], "1")
            case = suite.find("testcase")
            self.assertEqual(case.attrib["name"], "test_discovery")
            self.assertIn("ImportError", case.find("error").text)

    def test_skip_is_failure_unless_exactly_allowlisted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            test_directory = Path(temp_dir) / "tests"
            test_directory.mkdir()
            (test_directory / "test_skip.py").write_text(
                "import unittest\n"
                "class SkipTests(unittest.TestCase):\n"
                "    @unittest.skip('missing optional tool')\n"
                "    def test_optional(self):\n"
                "        pass\n",
                encoding="utf-8",
            )
            rejected = Path(temp_dir) / "rejected.xml"
            allowed = Path(temp_dir) / "allowed.xml"
            command = [
                sys.executable,
                str(UNITTEST_REPORTER),
                "--start-directory",
                str(test_directory),
                "--output",
            ]

            rejected_result = subprocess.run(command + [str(rejected)], check=False)
            allowed_result = subprocess.run(
                command
                + [
                    str(allowed),
                    "--allow-skip",
                    "test_skip.SkipTests.test_optional",
                ],
                check=False,
            )

            self.assertNotEqual(rejected_result.returncode, 0)
            rejected_suite = ET.parse(rejected).getroot()
            self.assertEqual(rejected_suite.attrib["failures"], "1")
            self.assertEqual(allowed_result.returncode, 0)
            allowed_suite = ET.parse(allowed).getroot()
            self.assertEqual(allowed_suite.attrib["skipped"], "1")

            other = Path(temp_dir) / "other.xml"
            other_result = subprocess.run(
                command
                + [str(other), "--allow-skip", "test_skip.SkipTests.test_other"],
                check=False,
            )
            self.assertNotEqual(other_result.returncode, 0)
            other_suite = ET.parse(other).getroot()
            self.assertEqual(other_suite.attrib["failures"], "1")
            self.assertEqual(other_suite.attrib["skipped"], "0")
            self.assertIn(
                "test_skip.SkipTests.test_optional",
                other_suite.find("testcase/failure").text,
            )

    def test_expected_failure_is_never_allowlisted_as_a_skip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            test_directory = Path(temp_dir) / "tests"
            test_directory.mkdir()
            (test_directory / "test_expected.py").write_text(
                "import unittest\n"
                "class ExpectedTests(unittest.TestCase):\n"
                "    @unittest.expectedFailure\n"
                "    def test_broken(self):\n"
                "        self.fail('still broken')\n",
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
                    "--allow-skip",
                    "test_expected.ExpectedTests.test_broken",
                ],
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            suite = ET.parse(output).getroot()
            self.assertEqual(suite.attrib["failures"], "1")
            self.assertEqual(suite.attrib["skipped"], "0")


if __name__ == "__main__":
    unittest.main()
