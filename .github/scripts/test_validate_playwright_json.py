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


PINNED = "a schedule the user paused should stay paused"


def known_failure_report(*messages: str) -> dict[str, object]:
    """A test pinned with pinKnownFailure() that failed with *messages*, as
    Playwright reports `expect(value, message)` failures."""
    report = valid_report()
    test = report["suites"][0]["specs"][0]["tests"][0]
    test["expectedStatus"] = "failed"
    test["annotations"] = [
        {"type": "issue", "description": "https://example.invalid/issues/1"},
        {"type": "fail", "description": "#1: re-hiring resumes it"},
        {"type": "known-failure", "description": PINNED},
    ]
    test["results"][0]["status"] = "failed"
    test["results"][0]["errors"] = [{"message": message} for message in messages]
    return report


def failure(check: str) -> str:
    return (
        f"\x1b[31mError: {check}\x1b[39m\n\n"
        "expect(received).toBe(expected) // Object.is equality\n\n"
        'Expected: "paused"\nReceived: "active"'
    )


class ValidatePlaywrightJSONTests(unittest.TestCase):
    def test_accepts_clean_single_attempt_report(self):
        self.assertEqual(validate_report(valid_report()), 1)

    def test_known_failures_are_only_accepted_when_allowed(self):
        report = known_failure_report(failure(PINNED))

        with self.assertRaisesRegex(ValueError, "expectedStatus must be passed"):
            validate_report(report)
        self.assertEqual(validate_report(report, allow_known_failures=True), 1)

    def test_a_known_failure_must_fail_only_on_its_pinned_check(self):
        # A cleanup's soft check, or the journey guard at teardown, failing as
        # well is a new failure the known bug must not hide.
        for extra in (
            "deleting schedule 42: 500",
            "the journey hit 1 unexpected error(s). Declare deliberate ones with "
            "test.use({ expectedResponses }):",
        ):
            with self.subTest(extra=extra):
                with self.assertRaisesRegex(ValueError, "but it also failed on"):
                    validate_report(
                        known_failure_report(failure(PINNED), failure(extra)),
                        allow_known_failures=True,
                    )
        # The whole first line, not a prefix: a longer message is another check.
        for other in ("another check", f"{PINNED} (after reload)"):
            with self.subTest(other=other):
                with self.assertRaisesRegex(ValueError, "but it also failed on"):
                    validate_report(
                        known_failure_report(failure(other)),
                        allow_known_failures=True,
                    )

    def test_a_known_failure_fails_once(self):
        with self.assertRaisesRegex(ValueError, "failed 2 times on its pinned check"):
            validate_report(
                known_failure_report(failure(PINNED), failure(PINNED)),
                allow_known_failures=True,
            )

    def test_a_plain_error_message_matches_its_pinned_check(self):
        # No ANSI colour and no "Error: " prefix, as some reporters write it.
        report = known_failure_report(f"{PINNED}\n\nExpected: paused")
        self.assertEqual(validate_report(report, allow_known_failures=True), 1)

    def test_a_known_failure_needs_one_pinned_check_and_an_error(self):
        report = known_failure_report(failure(PINNED))
        test = report["suites"][0]["specs"][0]["tests"][0]
        test["annotations"] = test["annotations"][:2]
        with self.assertRaisesRegex(ValueError, "with 0 pinned checks, not one"):
            validate_report(report, allow_known_failures=True)

        report = known_failure_report(failure(PINNED))
        test = report["suites"][0]["specs"][0]["tests"][0]
        test["annotations"].append({"type": "known-failure", "description": PINNED})
        with self.assertRaisesRegex(ValueError, "with 2 pinned checks, not one"):
            validate_report(report, allow_known_failures=True)

        with self.assertRaisesRegex(ValueError, "reported no error"):
            validate_report(known_failure_report(), allow_known_failures=True)

    def test_a_known_failure_that_passed_is_rejected(self):
        report = valid_report()
        report["suites"][0]["specs"][0]["tests"][0]["expectedStatus"] = "failed"

        with self.assertRaisesRegex(ValueError, "must be failed for a known failure"):
            validate_report(report, allow_known_failures=True)

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

    def test_rejects_invalid_tests_even_when_aggregate_stats_are_clean(self):
        for field, value, message in (
            ("status", "flaky", ".status must be expected"),
            ("expectedStatus", "failed", ".expectedStatus must be passed"),
        ):
            with self.subTest(field=field):
                report = valid_report()
                report["suites"][0]["specs"][0]["tests"][0][field] = value
                with self.assertRaisesRegex(ValueError, message):
                    validate_report(report)

    def test_rejects_invalid_results_even_when_aggregate_stats_are_clean(self):
        for field, value, message in (
            ("status", "failed", ".status must be passed"),
            ("errors", [{"message": "boom"}], "contains 1 errors"),
            ("retry", 1, ".retry must be 0"),
        ):
            with self.subTest(field=field):
                report = valid_report()
                report["suites"][0]["specs"][0]["tests"][0]["results"][0][field] = value
                with self.assertRaisesRegex(ValueError, message):
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

    def test_missing_tests_fails_cleanly_and_preserves_original_report(self):
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "results.json"
            report = valid_report()
            del report["suites"][0]["specs"][0]["tests"]
            original = json.dumps(report)
            report_path.write_text(original, encoding="utf-8")

            self.assertEqual(main(["--synthesize-invalid", str(report_path)]), 1)
            self.assertEqual(report_path.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
