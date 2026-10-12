import subprocess
import sys
import tempfile
import textwrap
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


REPORTER = Path(__file__).resolve().parents[3] / ".github/scripts/run_unittest_junit.py"


class UnittestSkipReportingTests(unittest.TestCase):
    def run_reporter(self, source):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tests = root / "tests"
            tests.mkdir()
            (tests / "test_healthy.py").write_text(
                "import unittest\n"
                "class Healthy(unittest.TestCase):\n"
                "    def test_passes(self): pass\n",
                encoding="utf-8",
            )
            (tests / "test_skipped.py").write_text(
                textwrap.dedent(source), encoding="utf-8"
            )
            output = root / "junit.xml"
            result = subprocess.run(
                [
                    sys.executable,
                    str(REPORTER),
                    "--start-directory",
                    str(tests),
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            return result, ET.parse(output).getroot()

    def assert_skip_rejected(self, source, expected_name):
        result, report = self.run_reporter(source)
        self.assertNotEqual(result.returncode, 0, result.stderr)
        rejected = [
            case
            for case in report.findall("testcase")
            if case.find("failure") is not None
        ]
        self.assertEqual(len(rejected), 1)
        self.assertIn(expected_name, rejected[0].get("name"))
        self.assertIn("disallowed skip", rejected[0].find("failure").text)
        self.assertIn("required service unavailable", rejected[0].find("failure").text)
        self.assertEqual(report.get("failures"), "1")

    def test_class_fixture_skip_is_reported_and_rejected(self):
        self.assert_skip_rejected(
            """
            import unittest
            class Required(unittest.TestCase):
                @classmethod
                def setUpClass(cls):
                    raise unittest.SkipTest('required service unavailable')
                def test_required(self):
                    self.fail('must not execute after skipped setup')
            """,
            "Required",
        )

    def test_module_fixture_skip_is_reported_and_rejected(self):
        self.assert_skip_rejected(
            """
            import unittest
            def setUpModule():
                raise unittest.SkipTest('required service unavailable')
            class Required(unittest.TestCase):
                def test_required(self):
                    self.fail('must not execute after skipped setup')
            """,
            "setUpModule",
        )

    def test_subtest_skip_is_reported_and_rejected(self):
        self.assert_skip_rejected(
            """
            import unittest
            class Required(unittest.TestCase):
                def test_required(self):
                    with self.subTest(service='database'):
                        self.skipTest('required service unavailable')
                    with self.subTest(service='queue'):
                        self.assertTrue(True)
            """,
            "test_required (service='database')",
        )

    def test_subtest_skip_cannot_replace_an_assertion_failure(self):
        result, report = self.run_reporter(
            """
            import unittest
            class Required(unittest.TestCase):
                def test_required(self):
                    with self.subTest(service='database'):
                        self.fail('database assertion failed')
                    with self.subTest(service='queue'):
                        self.skipTest('required service unavailable')
            """
        )
        self.assertNotEqual(result.returncode, 0)
        failures = report.findall("testcase/failure")
        self.assertEqual(len(failures), 2)
        self.assertTrue(
            any("database assertion failed" in case.text for case in failures)
        )
        self.assertTrue(any("disallowed skip" in case.text for case in failures))

    def test_repeated_subtest_failures_and_errors_are_all_reported(self):
        result, report = self.run_reporter(
            """
            import unittest
            class Required(unittest.TestCase):
                def test_required(self):
                    with self.subTest(service='database'):
                        self.fail('database assertion failed')
                    with self.subTest(service='queue'):
                        self.fail('queue assertion failed')
                    with self.subTest(service='storage'):
                        raise RuntimeError('storage unavailable')
                    with self.subTest(service='optional'):
                        self.skipTest('required service unavailable')
            """
        )
        self.assertNotEqual(result.returncode, 0)
        cases = report.findall("testcase")
        self.assertEqual(report.get("tests"), str(len(cases)))
        self.assertEqual(report.get("failures"), "3")
        self.assertEqual(report.get("errors"), "1")
        failures = report.findall("testcase/failure")
        self.assertEqual(len(failures), 3)
        for message in [
            "database assertion failed",
            "queue assertion failed",
            "disallowed skip",
        ]:
            self.assertTrue(any(message in failure.text for failure in failures))
        self.assertIn("storage unavailable", report.find("testcase/error").text)
        self.assertEqual(len({case.get("name") for case in cases}), len(cases))

    def test_cleanup_error_preserves_subtest_failure(self):
        result, report = self.run_reporter(
            """
            import unittest
            class Required(unittest.TestCase):
                def cleanup(self):
                    raise RuntimeError('cleanup failed')
                def test_required(self):
                    self.addCleanup(self.cleanup)
                    with self.subTest(service='database'):
                        self.fail('database assertion failed')
            """
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(report.get("failures"), "1")
        self.assertEqual(report.get("errors"), "1")
        self.assertIn("database assertion failed", report.find("testcase/failure").text)
        self.assertIn("cleanup failed", report.find("testcase/error").text)


if __name__ == "__main__":
    unittest.main()
