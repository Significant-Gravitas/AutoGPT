"""LaunchDarkly-aware model selection for the copilot.

Each cell of the ``(mode, tier)`` matrix has a static default baked into
``ChatConfig`` (see ``copilot/config.py``) and a matching LaunchDarkly
string-valued feature flag that can override it per-user.  This module
centralises the lookup so both the baseline and SDK paths agree on the
selection rule and so A/B experiments can target a single cell without
shipping a config change.

Matrix:

    +----------+-------------------------------------+-------------------------------------+
    |          | standard                            | advanced                            |
    +----------+-------------------------------------+-------------------------------------+
    | fast     | copilot-fast-standard-model         | copilot-fast-advanced-model         |
    | thinking | copilot-thinking-standard-model     | copilot-thinking-advanced-model     |
    +----------+-------------------------------------+-------------------------------------+

LD flag values are arbitrary strings (model identifiers, e.g.
``"anthropic/claude-sonnet-4-6"`` or ``"moonshotai/kimi-k2.6"``).  Empty
or non-string values fall back to the config default.
"""

from __future__ import annotations

import logging
from typing import Literal

from backend.copilot.config import ChatConfig
from backend.util.feature_flag import Flag, get_feature_flag_value

logger = logging.getLogger(__name__)

ModelMode = Literal["fast", "thinking"]
ModelTier = Literal["standard", "advanced"]


_FLAG_BY_CELL: dict[tuple[ModelMode, ModelTier], Flag] = {
    ("fast", "standard"): Flag.COPILOT_FAST_STANDARD_MODEL,
    ("fast", "advanced"): Flag.COPILOT_FAST_ADVANCED_MODEL,
    ("thinking", "standard"): Flag.COPILOT_THINKING_STANDARD_MODEL,
    ("thinking", "advanced"): Flag.COPILOT_THINKING_ADVANCED_MODEL,
}


def _config_default(config: ChatConfig, mode: ModelMode, tier: ModelTier) -> str:
    if mode == "fast":
        return (
            config.fast_advanced_model
            if tier == "advanced"
            else config.fast_standard_model
        )
    return (
        config.thinking_advanced_model
        if tier == "advanced"
        else config.thinking_standard_model
    )


async def resolve_model(
    mode: ModelMode,
    tier: ModelTier,
    user_id: str | None,
    *,
    config: ChatConfig,
) -> str:
    """Return the model identifier for a ``(mode, tier)`` cell.

    Consults the matching LaunchDarkly flag for *user_id* first and
    falls back to the ``ChatConfig`` default on missing user, missing
    flag, or non-string flag value.  Passing *config* explicitly keeps
    the resolver cheap to unit-test.
    """
    fallback = _config_default(config, mode, tier).strip()
    if not user_id:
        return fallback

    flag = _FLAG_BY_CELL[(mode, tier)]
    try:
        value = await get_feature_flag_value(flag.value, user_id, default=fallback)
    except Exception:
        logger.warning(
            "[model_router] LD lookup failed for %s — using config default %s",
            flag.value,
            fallback,
            exc_info=True,
        )
        return fallback

    if isinstance(value, str) and value.strip():
        return value.strip()
    if value != fallback:
        reason = (
            "empty string"
            if isinstance(value, str)
            else f"non-string ({type(value).__name__})"
        )
        logger.warning(
            "[model_router] LD flag %s returned %s — using config default %s",
            flag.value,
            reason,
            fallback,
        )
    return fallback
