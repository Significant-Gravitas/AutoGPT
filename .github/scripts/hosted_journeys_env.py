"""Print the backend .env lines that put a compose stack in hosted configuration.

The journeys leg of platform-fullstack-ci appends these to backend/.env. Flag
values come from the same file the Playwright journeys hand the frontend, so
both sides always agree.

The plans' Stripe prices are a JSON flag only the backend reads
(``copilot-tier-stripe-prices``), so they stay out of that file: they are
built from the fake Stripe's own catalogue, the plans it sells under fixed
ids, so the platform asks the fake for prices it has.
"""

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_FLAGS_FILE = (
    REPO / "autogpt_platform/frontend/src/playwright/journeys/hosted-flags.json"
)
FAKE_STRIPE_CATALOG = (
    REPO / "autogpt_platform/scripted_model/scripted_model/fake_stripe/catalog.json"
)

# Settings the hosted product runs with that the compose defaults leave off.
# EXECUTION_COST_COUNT_THRESHOLD is NOT a hosted value (hosted bills one credit
# per 100 node executions): billing every node makes a run's cost visible, so
# journeys can assert it exactly.
HOSTED_SETTINGS = {
    "ENABLE_CREDIT": "true",
    "ENABLE_PLATFORM_PAYMENT": "true",
    "EXECUTION_COST_COUNT_THRESHOLD": "1",
    # Stripe is the fake in the scripted_model service
    # (docker-compose.journeys.yml), which takes any test key and signs its
    # webhooks with FAKE_STRIPE_WEBHOOK_SECRET, the same value as here.
    "STRIPE_API_BASE": "http://scripted_model:4010/stripe",
    "STRIPE_API_KEY": "sk_test_journeys_fake_stripe",  # pragma: allowlist secret
    "STRIPE_WEBHOOK_SECRET": "whsec_journeys_fake_stripe",  # pragma: allowlist secret
}


def flag_env_name(flag_key: str) -> str:
    return "FORCE_FLAG_" + flag_key.upper().replace("-", "_")


def hosted_env_lines(
    flags: dict[str, object], catalog: dict | None = None
) -> list[str]:
    lines = [f"{name}={value}" for name, value in HOSTED_SETTINGS.items()]
    for key, value in flags.items():
        if not isinstance(value, bool):
            raise ValueError(f"hosted flag {key!r} must be true or false")
        lines.append(f"{flag_env_name(key)}={'true' if value else 'false'}")
    for key, value in backend_json_flags(catalog or load_catalog()).items():
        lines.append(f"{flag_env_name(key)}={env_json(value)}")
    return lines


def backend_json_flags(catalog: dict) -> dict[str, object]:
    """The JSON flags only the backend reads, which ``FORCE_FLAG_<NAME>``
    sets for a flag listed in its ``_NON_BOOLEAN_FLAG_VALUES``.

    ``copilot-tier-stripe-prices`` maps a plan's key (``PRO``, ``PRO_YEARLY``)
    to its Stripe price id; the fake's prices carry that key as their
    ``lookup_key``.
    """
    prices = {price["lookup_key"]: price["id"] for price in catalog["prices"]}
    return {"copilot-tier-stripe-prices": prices}


def env_json(value: object) -> str:
    """A JSON value as one unquoted ``.env`` value that compose's env-file
    parser keeps as it is: no whitespace (a value is trimmed, and `` #``
    starts a comment), no ``$`` (interpolated), and no opening quote (a
    quoted value is unescaped)."""
    text = json.dumps(value, separators=(",", ":"), sort_keys=True)
    if any(c.isspace() or c == "$" for c in text) or text[0] in "\"'":
        raise ValueError(f"{text!r} would not survive compose's env-file parsing")
    return text


def load_catalog(path: Path = FAKE_STRIPE_CATALOG) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flags-file", type=Path, default=DEFAULT_FLAGS_FILE)
    args = parser.parse_args()
    flags = json.loads(args.flags_file.read_text(encoding="utf-8"))
    print("\n".join(hosted_env_lines(flags)))


if __name__ == "__main__":
    main()
