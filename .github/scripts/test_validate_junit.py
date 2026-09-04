import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
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
    def test_failed_synthetic_write_returns_failure_without_losing_cause(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "missing.xml"
            with (
                patch(
                    "validate_junit.write_synthetic_error",
                    side_effect=OSError("read only"),
                ),
                patch("sys.stderr") as stderr,
            ):
                self.assertEqual(main(["--synthesize-invalid", str(report)]), 1)
            output = "".join(call.args[0] for call in stderr.write.call_args_list)
            self.assertIn("report is missing or empty", output)
            self.assertIn("could not write synthetic error report: read only", output)

    def test_skips_without_classname_cannot_match_fully_qualified_allowlist(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "skipped.xml"
            allowlist = Path(directory) / "allowed.json"
            ElementTree.ElementTree(report_xml(skipped=1)).write(report)
            allowlist.write_text(json.dumps(["backend.example.skipped-0"]))
            self.assertEqual(
                main(["--allow-skips-from", str(allowlist), str(report)]), 1
            )

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

    def test_require_no_skips_rejects_a_skipped_case(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "skipped.xml"
            ElementTree.ElementTree(report_xml(skipped=1)).write(report)

            self.assertEqual(main([str(report)]), 0)
            self.assertEqual(main(["--require-no-skips", str(report)]), 1)

    def test_skip_allowlist_rejects_new_skips_and_accepts_existing_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "skipped.xml"
            allowlist = Path(directory) / "allowed.json"
            root = report_xml(skipped=1)
            for case in root.iter("testcase"):
                case.set("classname", "backend.example")
            ElementTree.ElementTree(root).write(report)
            allowlist.write_text(json.dumps(["backend.example.skipped-0"]))
            args = ["--allow-skips-from", str(allowlist), str(report)]
            self.assertEqual(main(args), 0)
            allowlist.write_text(json.dumps(["backend.example.other"]))
            with patch("sys.stderr") as stderr:
                self.assertEqual(main(args), 1)
            output = "".join(call.args[0] for call in stderr.write.call_args_list)
            self.assertIn(str(allowlist), output)
            self.assertIn("do not allowlist a regression", output)

    def test_skip_allowlist_must_be_a_valid_list_of_exact_ids(self):
        self._check_invalid_allowlists()

    def test_allowlisted_all_skipped_report_still_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "skipped.xml"
            allowlist = Path(directory) / "allowed.json"
            report.write_text(
                '<testsuite><testcase classname="backend.example" name="only"><skipped/></testcase></testsuite>'
            )
            allowlist.write_text(json.dumps(["backend.example.only"]))
            self.assertEqual(
                main(["--allow-skips-from", str(allowlist), str(report)]), 1
            )

    def _check_invalid_allowlists(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "passed.xml"
            allowlist = Path(directory) / "allowed.json"
            ElementTree.ElementTree(report_xml()).write(report)
            args = ["--allow-skips-from", str(allowlist), str(report)]
            for content in (
                "{",
                "{}",
                "[1]",
                '[""]',
                '["a", "a"]',
                '[".skipped-0"]',
                '["backend.example."]',
                '["missing-separator"]',
            ):
                with self.subTest(content=content):
                    allowlist.write_text(content)
                    self.assertEqual(main(args), 1)
            allowlist.unlink()
            self.assertEqual(main(args), 1)

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
