"""Refresh the vendored LiteLLM Anthropic rate JSON.

Pulls the upstream LiteLLM model-pricing file and writes only the
Claude / anthropic-prefixed entries into
``backend/copilot/litellm_anthropic_rates.json``.

Run manually (``poetry run python scripts/refresh_litellm_anthropic_rates.py``)
or via the ``refresh-litellm-anthropic-rates`` GitHub Actions cron, which
opens a PR with the diff so price changes go through normal review.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

LITELLM_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/litellm/"
    "model_prices_and_context_window_backup.json"
)
TARGET = (
    Path(__file__).resolve().parent.parent
    / "backend"
    / "copilot"
    / "litellm_anthropic_rates.json"
)


def main() -> int:
    with urllib.request.urlopen(LITELLM_URL, timeout=30) as resp:
        raw = json.load(resp)
    if not isinstance(raw, dict):
        print(f"unexpected top-level shape: {type(raw).__name__}", file=sys.stderr)
        return 1
    filtered = {
        k: v
        for k, v in raw.items()
        if isinstance(v, dict)
        and (k.startswith("claude") or k.startswith("anthropic/claude"))
    }
    if not filtered:
        print(
            "no Claude entries in upstream JSON — refusing to overwrite",
            file=sys.stderr,
        )
        return 1
    serialised = json.dumps(filtered, indent=2, sort_keys=True) + "\n"
    TARGET.write_text(serialised, encoding="utf-8")
    print(f"wrote {len(filtered)} Claude entries to {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
