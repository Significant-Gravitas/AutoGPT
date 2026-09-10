import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from validate_cobertura import main, validate_report


def valid_report() -> ET.Element:
    return ET.fromstring(
        '<coverage lines-valid="1" lines-covered="0" line-rate="0">'
        '<packages><package name="src"><classes>'
        '<class name="example" filename="src/example.ts">'
        '<lines><line number="1" hits="0"/></lines>'
        "</class></classes></package></packages></coverage>"
    )


class ValidateCoberturaTests(unittest.TestCase):
    def test_accepts_real_uncovered_lines_without_imposing_a_threshold(self):
        validate_report(valid_report())

    def test_rejects_arbitrary_xml_and_empty_coverage(self):
        for report in ("<report/>", "<coverage/>"):
            with self.subTest(report=report), self.assertRaises(ValueError):
                validate_report(ET.fromstring(report))

    def test_rejects_invalid_summary(self):
        for attribute, value in (
            ("lines-valid", "0"),
            ("lines-valid", "-1"),
            ("lines-covered", "2"),
            ("lines-covered", "0.5"),
            ("line-rate", "nan"),
            ("line-rate", "inf"),
            ("line-rate", "-0.1"),
            ("line-rate", "1.1"),
        ):
            with self.subTest(attribute=attribute, value=value):
                report = valid_report()
                report.set(attribute, value)
                with self.assertRaises(ValueError):
                    validate_report(report)

    def test_requires_source_classes_and_lines(self):
        for path in ("packages", "packages/package/classes/class/lines"):
            with self.subTest(path=path):
                report = valid_report()
                report.find(path).clear()
                with self.assertRaises(ValueError):
                    validate_report(report)

    def test_rejects_invalid_line_data(self):
        for attribute, value in (("number", "0"), ("hits", "-1")):
            with self.subTest(attribute=attribute):
                report = valid_report()
                report.find(".//line").set(attribute, value)
                with self.assertRaises(ValueError):
                    validate_report(report)

    def test_rejects_missing_filename(self):
        report = valid_report()
        del report.find(".//class").attrib["filename"]
        with self.assertRaises(ValueError):
            validate_report(report)

    def test_rejects_summary_counts_that_disagree_with_source_lines(self):
        for attribute, value in (("lines-valid", "2"), ("lines-covered", "1")):
            with self.subTest(attribute=attribute):
                report = valid_report()
                report.set(attribute, value)
                with self.assertRaises(ValueError):
                    validate_report(report)

    def test_rejects_uncovered_summary_with_a_covered_source_line(self):
        report = valid_report()
        report.find(".//class/lines/line").set("hits", "1")
        with self.assertRaises(ValueError):
            validate_report(report)

    def test_rejects_rate_that_disagrees_with_source_line_counts(self):
        report = valid_report()
        report.set("line-rate", "0.5")
        with self.assertRaises(ValueError):
            validate_report(report)

    def test_accepts_rounded_reporter_rate_and_ignores_method_line_duplicates(self):
        report = valid_report()
        report.set("lines-valid", "3")
        report.set("lines-covered", "1")
        lines = report.find(".//class/lines")
        ET.SubElement(lines, "line", number="2", hits="1")
        ET.SubElement(lines, "line", number="3", hits="0")
        methods = ET.SubElement(report.find(".//class"), "methods")
        method_lines = ET.SubElement(ET.SubElement(methods, "method"), "lines")
        ET.SubElement(method_lines, "line", number="2", hits="1")
        for rate in ("0.3333", "0.33329999999999999", str(1 / 3)):
            with self.subTest(rate=rate):
                report.set("line-rate", rate)
                validate_report(report)

    def test_counts_nested_packages_in_e2e_reports(self):
        report = valid_report()
        classes = report.find("packages/package/classes")
        source = classes.find("class")
        classes.remove(source)
        nested = ET.SubElement(classes, "package", name="nested")
        ET.SubElement(nested, "classes").append(source)
        validate_report(report)

    def test_cli_fails_for_missing_empty_malformed_or_noncoverage_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "coverage.xml"
            self.assertEqual(main([str(path)]), 1)
            for text in ("", "<coverage", "<report/>"):
                with self.subTest(text=text):
                    path.write_text(text, encoding="utf-8")
                    self.assertEqual(main([str(path)]), 1)
                    self.assertEqual(path.read_text(encoding="utf-8"), text)

    def test_cli_accepts_valid_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "coverage.xml"
            ET.ElementTree(valid_report()).write(path, encoding="utf-8")
            self.assertEqual(main([str(path)]), 0)


if __name__ == "__main__":
    unittest.main()
