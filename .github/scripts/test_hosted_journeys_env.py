import json
import unittest

from hosted_journeys_env import DEFAULT_FLAGS_FILE, hosted_env_lines


class HostedJourneysEnvTests(unittest.TestCase):
    def test_flags_become_backend_overrides(self):
        lines = hosted_env_lines(
            {"generic-trigger-agents": True, "copilot-stream-runtime": False}
        )
        self.assertIn("FORCE_FLAG_GENERIC_TRIGGER_AGENTS=true", lines)
        self.assertIn("FORCE_FLAG_COPILOT_STREAM_RUNTIME=false", lines)
        self.assertIn("ENABLE_CREDIT=true", lines)

    def test_non_boolean_flags_are_rejected(self):
        with self.assertRaises(ValueError):
            hosted_env_lines({"copilot-model-routing": {"default": "x"}})

    def test_payments_are_on_as_in_production(self):
        # Journey users get a paid tier from the Playwright provisioning
        # helper instead of Stripe.
        self.assertIn("ENABLE_PLATFORM_PAYMENT=true", hosted_env_lines({}))

    def test_checked_in_flags_are_boolean(self):
        flags = json.loads(DEFAULT_FLAGS_FILE.read_text(encoding="utf-8"))
        self.assertTrue(all(isinstance(v, bool) for v in flags.values()))


if __name__ == "__main__":
    unittest.main()
