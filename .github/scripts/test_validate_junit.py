import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree

from validate_junit import main, summarize_junit


def report_xml(
    *, failures: int = 0, errors: int = 0, skipped: int = 0
) -> ElementTree.Element:
    tests = 1 + failures + errors + skipped
    root = ElementTree.Element(
        "testsuites",
        tests=str(tests),
        failures=str(failures),
        errors=str(errors),
        skipped=str(skipped),
    )
    suite = ElementTree.SubElement(root, "testsuite", name="suite")
    ElementTree.SubElement(suite, "testcase", name="passes")
    for index in range(failures):
        case = ElementTree.SubElement(suite, "testcase", name=f"failure-{index}")
        ElementTree.SubElement(case, "failure")
    for index in range(errors):
        case = ElementTree.SubElement(suite, "testcase", name=f"error-{index}")
        ElementTree.SubElement(case, "error")
    for index in range(skipped):
        case = ElementTree.SubElement(suite, "testcase", name=f"skipped-{index}")
        ElementTree.SubElement(case, "skipped")
    return root


class ValidateJUnitTests(unittest.TestCase):
    def test_accepts_passing_report_with_accounted_skips(self):
        summary = summarize_junit(report_xml(skipped=2))

        self.assertEqual(summary.tests, 3)
        self.assertEqual(summary.passed, 1)
        self.assertEqual(summary.skipped, 2)

    def test_rejects_failure_and_error_cases(self):
        for root in (report_xml(failures=1), report_xml(errors=1)):
            with (
                self.subTest(root=root.attrib),
                self.assertRaisesRegex(ValueError, "failures and .* errors"),
            ):
                summarize_junit(root)

    def test_rejects_declared_count_mismatch(self):
        root = report_xml()
        root.set("failures", "1")

        with self.assertRaisesRegex(ValueError, "root failures count"):
            summarize_junit(root)

    def test_rejects_nested_suite_count_mismatch_without_root_counts(self):
        root = report_xml()
        root.attrib.clear()
        suite = next(root.iter("testsuite"))
        suite.set("tests", "2")

        with self.assertRaisesRegex(ValueError, "testsuite 'suite' tests count"):
            summarize_junit(root)

    def test_accepts_skipped_only_report(self):
        root = report_xml(skipped=1)
        passing_case = next(root.iter("testcase"))
        passing_case.append(ElementTree.Element("skipped"))
        root.set("skipped", "2")

        summary = summarize_junit(root)

        self.assertEqual(summary.tests, 2)
        self.assertEqual(summary.passed, 0)

    def test_synthesizes_machine_readable_error_for_missing_report(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "missing.xml"

            status = main(["--synthesize-invalid", str(report)])

            self.assertEqual(status, 1)
            root = ElementTree.parse(report).getroot()
            self.assertEqual(root.get("tests"), "1")
            self.assertEqual(root.get("errors"), "1")
            self.assertEqual(len(list(root.iter("error"))), 1)

    def test_synthesizes_machine_readable_error_for_malformed_report(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "malformed.xml"
            report.write_text("<testsuites>", encoding="utf-8")

            status = main(["--synthesize-invalid", str(report)])

            self.assertEqual(status, 1)
            root = ElementTree.parse(report).getroot()
            self.assertEqual(root.get("errors"), "1")


if __name__ == "__main__":
    unittest.main()
