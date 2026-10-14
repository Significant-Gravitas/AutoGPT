"""Refresh ``backend/copilot/anthropic_rates.json`` from LiteLLM upstream.

Pulls the community-maintained model pricing JSON from BerriAI/litellm,
filters to entries for Claude / anthropic models, and writes the
result next to the rate-card module.

Run manually (``poetry run python scripts/refresh_anthropic_rates.py``)
or via the weekly GitHub Actions cron at
``.github/workflows/refresh-anthropic-rates.yml``.

Source URL is intentionally pinned to the ``main`` branch backup file,
which is the canonical pricing snapshot LiteLLM ships in their package.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

_SOURCE_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "litellm/model_prices_and_context_window_backup.json"
)
_OUT_PATH = (
    Path(__file__).resolve().parents[1] / "backend" / "copilot" / "anthropic_rates.json"
)

# Rates checked by hand against Anthropic's pricing page. A refresh that would
# change one stops without writing, so a person confirms the new price first.
# Sonnet 5.5 and Opus 5.5 cache reads cost 0.05x input, not the usual 0.1x.
CHECKED_RATES: dict[str, dict[str, float]] = {
    "claude-sonnet-5-5": {
        "cache_read_input_token_cost": 1e-07,
        "cache_read_input_token_cost_batches": 5e-08,
    },
    "claude-opus-5-5": {
        "cache_read_input_token_cost": 2e-07,
        "cache_read_input_token_cost_batches": 1e-07,
    },
}


def checked_rate_changes(rates: dict[str, Any]) -> list[str]:
    """The hand-checked rates *rates* would change, one line each."""
    changes = []
    for model, fields in CHECKED_RATES.items():
        entry = rates.get(model)
        for field, expected in fields.items():
            actual = entry.get(field) if isinstance(entry, dict) else None
            if actual != expected:
                changes.append(f"{model} {field}: {expected} -> {actual}")
    return changes


def main() -> int:
    print(f"Fetching {_SOURCE_URL} ...", file=sys.stderr)
    with urllib.request.urlopen(_SOURCE_URL, timeout=30) as resp:
        data = json.loads(resp.read())
    if not isinstance(data, dict):
        print("Unexpected JSON shape (root not an object)", file=sys.stderr)
        return 1
    rates = {
        k: v
        for k, v in data.items()
        if isinstance(k, str)
        and (k.startswith("claude") or k.startswith("anthropic/claude"))
        and isinstance(v, dict)
    }
    changes = checked_rate_changes(rates)
    if changes:
        print(
            "Upstream changes rates checked by hand. Confirm them against "
            "Anthropic's pricing page, then update CHECKED_RATES:",
            file=sys.stderr,
        )
        for line in changes:
            print(f"  {line}", file=sys.stderr)
        return 1
    out = {
        "_source": _SOURCE_URL,
        "_filter": "keys starting with claude or anthropic/claude",
        "rates": rates,
    }
    _OUT_PATH.write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote {len(rates)} entries to {_OUT_PATH.name} "
        f"({_OUT_PATH.stat().st_size} bytes)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
