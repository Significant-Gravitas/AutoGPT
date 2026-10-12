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

    def test_checked_in_flags_match_hosted_and_are_boolean(self):
        # Payments are on as in production; journey users get a paid tier
        # from the Playwright provisioning helper instead of Stripe.
        flags = json.loads(DEFAULT_FLAGS_FILE.read_text(encoding="utf-8"))
        self.assertIs(flags["enable-platform-payment"], True)
        self.assertTrue(all(isinstance(v, bool) for v in flags.values()))


if __name__ == "__main__":
    unittest.main()
