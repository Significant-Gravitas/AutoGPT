import json
import unittest

from hosted_journeys_env import DEFAULT_FLAGS_FILE, hosted_env_lines


class HostedJourneysEnvTests(unittest.TestCase):
    def test_flags_become_backend_overrides(self):
        lines = hosted_env_lines(
            {"hire-experts": True, "enable-platform-payment": False}
        )
        self.assertIn("FORCE_FLAG_HIRE_EXPERTS=true", lines)
        self.assertIn("FORCE_FLAG_ENABLE_PLATFORM_PAYMENT=false", lines)
        self.assertIn("ENABLE_CREDIT=true", lines)

    def test_non_boolean_flags_are_rejected(self):
        with self.assertRaises(ValueError):
            hosted_env_lines({"copilot-model-routing": {"default": "x"}})

    def test_checked_in_flags_keep_fresh_users_off_the_paywall(self):
        # Journey users have no subscription; with the payment flag on every
        # run would be refused with a 402.
        flags = json.loads(DEFAULT_FLAGS_FILE.read_text(encoding="utf-8"))
        self.assertIs(flags["enable-platform-payment"], False)
        self.assertTrue(all(isinstance(v, bool) for v in flags.values()))


if __name__ == "__main__":
    unittest.main()
