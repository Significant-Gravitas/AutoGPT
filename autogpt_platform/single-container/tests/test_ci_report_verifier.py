import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
REPORT_VERIFIER = REPOSITORY_ROOT / ".github/scripts/verify_single_container_reports.py"


class ReportVerifierTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.report_dir = Path(self.temporary_directory.name)
        passing_junit = (
            '<testsuite tests="1" failures="0" errors="0" skipped="0">'
            '<testcase classname="ci" name="passes" />'
            "</testsuite>"
        )
        for name in ("runtime-helpers.xml", "publication-policy.xml", "smoke.xml"):
            (self.report_dir / name).write_text(passing_junit, encoding="utf-8")
        passing_trivy = {
            "SchemaVersion": 2,
            "ArtifactName": "single-container:test",
            "ArtifactType": "container_image",
            "Results": [
                {
                    "Target": "single-container:test (debian)",
                    "Class": "os-pkgs",
                    "Type": "debian",
                    "Vulnerabilities": [],
                    "Secrets": [],
                }
            ],
        }
        for name in ("trivy-critical.json", "trivy-secrets.json"):
            (self.report_dir / name).write_text(
                json.dumps(passing_trivy), encoding="utf-8"
            )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def run_verifier(self):
        return subprocess.run(
            [
                sys.executable,
                str(REPORT_VERIFIER),
                "--report-dir",
                str(self.report_dir),
                "--expected-image",
                "single-container:test",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_passing_reports_succeed(self):
        result = self.run_verifier()

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_junit_failure_is_rejected_even_when_summary_claims_success(self):
        (self.report_dir / "smoke.xml").write_text(
            '<testsuite tests="1" failures="0" errors="0" skipped="0">'
            '<testcase classname="ci" name="fails"><failure /></testcase>'
            "</testsuite>",
            encoding="utf-8",
        )

        result = self.run_verifier()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("declares failures=0, found 1", result.stdout)

    def test_junit_error_count_is_rejected(self):
        (self.report_dir / "runtime-helpers.xml").write_text(
            '<testsuite tests="1" failures="0" errors="1" skipped="0">'
            '<testcase classname="ci" name="errors"><error /></testcase>'
            "</testsuite>",
            encoding="utf-8",
        )

        result = self.run_verifier()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("reports 1 errors", result.stdout)

    def test_junit_aggregate_failure_is_rejected(self):
        (self.report_dir / "smoke.xml").write_text(
            '<testsuites tests="1" failures="1" errors="0" skipped="0">'
            '<testsuite tests="1" failures="0" errors="0" skipped="0">'
            '<testcase classname="ci" name="misreported" />'
            "</testsuite>"
            "</testsuites>",
            encoding="utf-8",
        )

        result = self.run_verifier()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("declares failures=1, found 0", result.stdout)

    def test_nested_junit_suites_are_accepted(self):
        (self.report_dir / "smoke.xml").write_text(
            '<testsuites tests="1" failures="0" errors="0" skipped="0">'
            '<testsuite tests="1" failures="0" errors="0" skipped="0">'
            '<testsuite tests="1" failures="0" errors="0" skipped="0">'
            '<testcase classname="ci" name="nested" />'
            "</testsuite>"
            "</testsuite>"
            "</testsuites>",
            encoding="utf-8",
        )

        result = self.run_verifier()

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_junit_count_mismatch_is_rejected(self):
        (self.report_dir / "publication-policy.xml").write_text(
            '<testsuite tests="99" failures="0" errors="0" skipped="0">'
            '<testcase classname="ci" name="only-one" />'
            "</testsuite>",
            encoding="utf-8",
        )

        result = self.run_verifier()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("declares tests=99, found 1", result.stdout)

    def test_junit_skipped_only_report_is_rejected(self):
        (self.report_dir / "smoke.xml").write_text(
            '<testsuite tests="1" failures="0" errors="0" skipped="1">'
            '<testcase classname="ci" name="skipped"><skipped /></testcase>'
            "</testsuite>",
            encoding="utf-8",
        )

        result = self.run_verifier()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("contains no successful test cases", result.stdout)

    def test_trivy_findings_are_rejected(self):
        report = {
            "SchemaVersion": 2,
            "ArtifactName": "single-container:test",
            "ArtifactType": "container_image",
            "Results": [
                {
                    "Target": "single-container:test (debian)",
                    "Class": "os-pkgs",
                    "Type": "debian",
                    "Vulnerabilities": [{"ID": "CVE"}],
                }
            ],
        }
        (self.report_dir / "trivy-critical.json").write_text(
            json.dumps(report), encoding="utf-8"
        )

        result = self.run_verifier()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Vulnerabilities=1", result.stdout)

    def test_non_trivy_json_is_rejected(self):
        (self.report_dir / "trivy-secrets.json").write_text("{}", encoding="utf-8")

        result = self.run_verifier()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("is not a Trivy JSON report", result.stdout)

    def test_trivy_report_without_scanned_results_is_rejected(self):
        (self.report_dir / "trivy-secrets.json").write_text(
            json.dumps(
                {
                    "SchemaVersion": 2,
                    "ArtifactName": "single-container:test",
                    "ArtifactType": "container_image",
                    "Results": None,
                }
            ),
            encoding="utf-8",
        )

        result = self.run_verifier()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid Results value", result.stdout)

    def test_trivy_result_without_scan_identity_is_rejected(self):
        (self.report_dir / "trivy-secrets.json").write_text(
            json.dumps(
                {
                    "SchemaVersion": 2,
                    "ArtifactName": "single-container:test",
                    "ArtifactType": "container_image",
                    "Results": [{}],
                }
            ),
            encoding="utf-8",
        )

        result = self.run_verifier()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("result is missing Target", result.stdout)

    def test_trivy_null_finding_list_is_rejected(self):
        report = json.loads((self.report_dir / "trivy-critical.json").read_text())
        report["Results"][0]["Vulnerabilities"] = None
        (self.report_dir / "trivy-critical.json").write_text(
            json.dumps(report), encoding="utf-8"
        )

        result = self.run_verifier()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid Vulnerabilities value", result.stdout)

    def test_trivy_wrong_image_is_rejected(self):
        report = json.loads((self.report_dir / "trivy-critical.json").read_text())
        report["ArtifactName"] = "other:image"
        (self.report_dir / "trivy-critical.json").write_text(
            json.dumps(report), encoding="utf-8"
        )

        result = self.run_verifier()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("scanned 'other:image'", result.stdout)


if __name__ == "__main__":
    unittest.main()
