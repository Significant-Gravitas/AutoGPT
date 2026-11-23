import json
import re
import unittest
from pathlib import Path

from hosted_journeys_env import DEFAULT_FLAGS_FILE, hosted_env_lines

JOURNEYS_COMPOSE = (
    Path(__file__).resolve().parents[2] / "autogpt_platform/docker-compose.journeys.yml"
)


def compose_setting(name: str) -> str:
    """A setting docker-compose.journeys.yml gives a service."""
    text = JOURNEYS_COMPOSE.read_text(encoding="utf-8")
    match = re.search(rf'^\s+{name}: "?([^"\s]+)"?', text, re.MULTILINE)
    if match is None:
        raise AssertionError(f"{name} is not set in docker-compose.journeys.yml")
    return match.group(1)


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

    def test_stripe_is_the_fake_and_gets_only_a_test_key(self):
        settings = dict(line.split("=", 1) for line in hosted_env_lines({}))
        self.assertEqual(
            settings["STRIPE_API_BASE"], "http://scripted_model:4010/stripe"
        )
        # The fake refuses live keys; one must never be configured for it.
        self.assertTrue(settings["STRIPE_API_KEY"].startswith("sk_test_"))

    def test_the_fake_stripe_signs_webhooks_for_the_backend(self):
        settings = dict(line.split("=", 1) for line in hosted_env_lines({}))
        self.assertEqual(
            compose_setting("FAKE_STRIPE_WEBHOOK_SECRET"),
            settings["STRIPE_WEBHOOK_SECRET"],
        )
        self.assertEqual(
            compose_setting("FAKE_STRIPE_WEBHOOK_URL"),
            "http://rest_server:8006/api/credits/stripe_webhook",
        )


if __name__ == "__main__":
    unittest.main()
