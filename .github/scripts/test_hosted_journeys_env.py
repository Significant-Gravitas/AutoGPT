import json
import re
import sys
import unittest
from pathlib import Path

from hosted_journeys_env import (
    DEFAULT_FLAGS_FILE,
    FAKE_STRIPE_CATALOG,
    env_json,
    hosted_env_lines,
    load_catalog,
)

REPO = Path(__file__).resolve().parents[2]
JOURNEYS_COMPOSE = REPO / "autogpt_platform/docker-compose.journeys.yml"
PRICES_ENV = "FORCE_FLAG_COPILOT_TIER_STRIPE_PRICES"


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
        # The frontend reads these; the price map is the backend's alone.
        flags = json.loads(DEFAULT_FLAGS_FILE.read_text(encoding="utf-8"))
        self.assertTrue(all(isinstance(v, bool) for v in flags.values()))
        self.assertNotIn("copilot-tier-stripe-prices", flags)

    def test_the_plans_are_priced_with_the_fake_stripes_prices(self):
        settings = dict(line.split("=", 1) for line in hosted_env_lines({}))
        prices = json.loads(settings[PRICES_ENV])
        catalog = load_catalog()
        self.assertEqual(prices, {p["lookup_key"]: p["id"] for p in catalog["prices"]})
        self.assertLessEqual({"PRO", "PRO_YEARLY", "MAX", "MAX_YEARLY"}, set(prices))
        # Every id is a price the fake sells under that key, as it starts.
        sys.path.insert(0, str(REPO / "autogpt_platform/scripted_model"))
        try:
            from scripted_model.fake_stripe.fake import FakeStripe
        finally:
            sys.path.pop(0)
        store = FakeStripe("http://localhost:4010").store
        for key, price_id in prices.items():
            price = store.get(price_id, "price")
            self.assertIsNotNone(price, price_id)
            self.assertEqual(price["lookup_key"], key)
        self.assertEqual(
            FAKE_STRIPE_CATALOG,
            REPO
            / "autogpt_platform/scripted_model/scripted_model/fake_stripe"
            / "catalog.json",
        )

    def test_a_json_value_survives_compose_env_files(self):
        """An unquoted .env value is trimmed, cut at `` #`` and interpolated
        at ``$``; a quoted one is unescaped. The price map needs none of it."""
        line = next(
            line for line in hosted_env_lines({}) if line.startswith(PRICES_ENV)
        )
        value = line.split("=", 1)[1]
        self.assertEqual(value, value.strip())
        self.assertNotIn(" #", value)
        self.assertNotIn("$", value)
        self.assertNotIn(value[0], "\"'")
        self.assertIsInstance(json.loads(value), dict)
        for unsafe in ({"PRO": "price with space"}, {"PRO": "price_$HOME"}):
            with self.assertRaises(ValueError):
                env_json(unsafe)

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
