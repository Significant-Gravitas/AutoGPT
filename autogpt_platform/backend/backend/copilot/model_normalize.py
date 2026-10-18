"""Model names as the platform route sends them.

Shared by the SDK and the fast engine, and by the side calls that run on the
platform route; the rule itself is the route's row (``route_profile.py``).
"""

from __future__ import annotations

from backend.copilot.config import ChatConfig
from backend.copilot.route_profile import normalize_model


def normalize_model_for_transport(raw_model: str, cfg: ChatConfig | None = None) -> str:
    """*raw_model* as the platform route sends it.

    OpenRouter and a local server keep the name as configured; the Anthropic
    routes take native Claude ids and refuse other vendors (``ValueError``).
    With the platform route disabled the name passes through unchanged.

    *cfg* is optional so call sites that already hold a per-module
    ``ChatConfig`` reference can pass it through and keep monkeypatch-based
    test fixtures working against that exact reference.
    """
    config = cfg if cfg is not None else ChatConfig()
    row = config.route_row
    if row is None:
        return raw_model
    return normalize_model(row, raw_model)
