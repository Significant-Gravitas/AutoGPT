import copy
import json
import tempfile
import unittest
from pathlib import Path

from validate_playwright_json import main, validate_report


def valid_report() -> dict[str, object]:
    return {
        "config": {"projects": [{"name": "chromium", "retries": 0}]},
        "errors": [],
        "stats": {
            "expected": 1,
            "skipped": 0,
            "unexpected": 0,
            "flaky": 0,
        },
        "suites": [
            {
                "specs": [
                    {
                        "tests": [
                            {
                                "status": "expected",
                                "expectedStatus": "passed",
                                "results": [
                                    {
                                        "retry": 0,
                                        "status": "passed",
                                        "errors": [],
                                    }
                                ],
                            }
                        ]
                    }
                ],
                "suites": [],
            }
        ],
    }


class ValidatePlaywrightJSONTests(unittest.TestCase):
    def test_accepts_clean_single_attempt_report(self):
        self.assertEqual(validate_report(valid_report()), 1)

    def test_rejects_configured_retries(self):
        report = valid_report()
        report["config"]["projects"][0]["retries"] = 2

        with self.assertRaisesRegex(ValueError, "retries must be 0"):
            validate_report(report)

    def test_rejects_flaky_test_that_eventually_passed(self):
        report = valid_report()
        report["stats"]["expected"] = 0
        report["stats"]["flaky"] = 1
        test = report["suites"][0]["specs"][0]["tests"][0]
        test["status"] = "flaky"
        test["results"].insert(
            0, {"retry": 0, "status": "failed", "errors": [{"message": "boom"}]}
        )
        test["results"][1]["retry"] = 1

        with self.assertRaisesRegex(ValueError, "stats.flaky must be 0"):
            validate_report(report)

    def test_rejects_skipped_test(self):
        report = valid_report()
        report["stats"]["expected"] = 0
        report["stats"]["skipped"] = 1

        with self.assertRaisesRegex(ValueError, "stats.skipped must be 0"):
            validate_report(report)

    def test_rejects_multiple_attempts_even_when_stats_claim_clean(self):
        report = valid_report()
        test = report["suites"][0]["specs"][0]["tests"][0]
        test["results"].append(copy.deepcopy(test["results"][0]))

        with self.assertRaisesRegex(ValueError, "exactly one attempt"):
            validate_report(report)

    def test_rejects_zero_test_report(self):
        report = valid_report()
        report["stats"]["expected"] = 0
        report["suites"] = []

        with self.assertRaisesRegex(ValueError, "zero expected tests"):
            validate_report(report)

    def test_rejects_report_below_required_inventory(self):
        with self.assertRaisesRegex(ValueError, "below required minimum 2"):
            validate_report(valid_report(), min_tests=2)

    def test_rejects_boolean_retry_value(self):
        report = valid_report()
        report["suites"][0]["specs"][0]["tests"][0]["results"][0]["retry"] = False

        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            validate_report(report)

    def test_synthesizes_machine_readable_error_for_missing_report(self):
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "results.json"

            self.assertEqual(
                main(["--synthesize-invalid", str(report_path)]),
                1,
            )
            synthesized = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(synthesized["stats"]["unexpected"], 1)
            self.assertEqual(len(synthesized["errors"]), 1)

    def test_preserves_semantically_failed_report(self):
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "results.json"
            report = valid_report()
            report["stats"]["flaky"] = 1
            report_path.write_text(json.dumps(report), encoding="utf-8")
            original = report_path.read_text(encoding="utf-8")

            self.assertEqual(
                main(["--synthesize-invalid", str(report_path)]),
                1,
            )
            self.assertEqual(report_path.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
