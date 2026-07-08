"""LaunchDarkly-aware model selection for the copilot.

Each cell of the ``(mode, tier)`` matrix has a static default baked into
``ChatConfig`` (see ``copilot/config.py``) and a single JSON-valued
LaunchDarkly feature flag — ``copilot-model-routing`` — that can override
it per-user.  This module centralises the lookup so both the baseline
and SDK paths agree on the selection rule and so A/B experiments can
target a single cell without shipping a config change.

Matrix:

    +----------+----------+----------+
    |          | standard | advanced |
    +----------+----------+----------+
    | fast     |    .     |    .     |
    | thinking |    .     |    .     |
    +----------+----------+----------+

LD payload shape::

    {
      "fast":     {"standard": "anthropic/claude-sonnet-4-6", "advanced": "anthropic/claude-opus-4-6"},
      "thinking": {"standard": "moonshotai/kimi-k2.6",         "advanced": "anthropic/claude-opus-4-6"}
    }

Missing mode, missing tier-within-mode, non-string cell value, non-dict
payload, or LD failure all fall back to the corresponding ``ChatConfig``
default.
"""

from __future__ import annotations

import logging
from typing import Literal

from backend.copilot.config import ChatConfig
from backend.util.feature_flag import Flag, get_feature_flag_value, is_feature_enabled

logger = logging.getLogger(__name__)

ModelMode = Literal["fast", "thinking"]
ModelTier = Literal["standard", "advanced"]

# Single-role model cells for the two-phase planner/executor split. Unlike the
# ``(mode, tier)`` matrix these have no sub-tiers, so they live as top-level
# string values in the ``copilot-model-routing`` payload — mirroring how
# ``title_model`` / ``simulation_model`` are single-role in ``ChatConfig``::
#
#     {"fast": {...}, "thinking": {...},
#      "planner": "anthropic/claude-opus-4.7",
#      "executor": "anthropic/claude-sonnet-4-6"}
#
# Missing key, non-string value, non-dict payload, or LD failure all fall back
# to the corresponding ``ChatConfig`` default — backward compatible with the
# existing matrix-only shape (a payload without ``planner``/``executor`` keys
# simply serves the config defaults).
ModelRole = Literal["planner", "executor"]


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

    Consults ``copilot-model-routing`` (JSON) for *user_id* — LD targeting is
    still per-user so cohorts can receive different routing — and falls back to
    the ``ChatConfig`` default on missing user, missing / invalid flag payload,
    or non-string cell value.
    """
    fallback = _config_default(config, mode, tier).strip()
    if not user_id:
        return fallback

    try:
        payload: object = await get_feature_flag_value(
            Flag.COPILOT_MODEL_ROUTING.value, user_id, default=None
        )
    except Exception:
        logger.warning(
            "[model_router] LD lookup failed for copilot-model-routing — "
            "using config default %s for (%s, %s)",
            fallback,
            mode,
            tier,
            exc_info=True,
        )
        return fallback

    if payload is None:
        return fallback

    if not isinstance(payload, dict):
        logger.warning(
            "[model_router] copilot-model-routing expected a JSON object, got %r — "
            "using config default %s for (%s, %s)",
            payload,
            fallback,
            mode,
            tier,
        )
        return fallback

    mode_cell = payload.get(mode)
    if mode in payload and not isinstance(mode_cell, dict):
        # Operator typed something at the mode level (e.g. a string) instead of
        # a {tier: model} dict — surface the typo in logs.
        logger.warning(
            "[model_router] copilot-model-routing[%s] expected a JSON object, "
            "got %r — using config default %s for tier %s",
            mode,
            mode_cell,
            fallback,
            tier,
        )
    if not isinstance(mode_cell, dict):
        return fallback

    value = mode_cell.get(tier)
    if isinstance(value, str) and value.strip():
        return value.strip()
    if value is not None:
        reason = (
            "empty string"
            if isinstance(value, str)
            else f"non-string ({type(value).__name__})"
        )
        logger.warning(
            "[model_router] copilot-model-routing[%s][%s] returned %s — "
            "using config default %s",
            mode,
            tier,
            reason,
            fallback,
        )
    return fallback


def _role_config_default(config: ChatConfig, role: ModelRole) -> str:
    return config.planner_model if role == "planner" else config.executor_model


async def resolve_role_model(
    role: ModelRole,
    user_id: str | None,
    *,
    config: ChatConfig,
) -> str:
    """Return the model identifier for a single-role cell (planner/executor).

    Consults the same ``copilot-model-routing`` JSON flag as
    :func:`resolve_model`, reading a top-level ``role`` key whose value is a
    model-name string.  Falls back to the ``ChatConfig`` default on missing
    user, missing / invalid flag payload, or non-string cell value — so the
    existing matrix-only payload shape keeps working unchanged.
    """
    fallback = _role_config_default(config, role).strip()
    if not user_id:
        return fallback

    try:
        payload: object = await get_feature_flag_value(
            Flag.COPILOT_MODEL_ROUTING.value, user_id, default=None
        )
    except Exception:
        logger.warning(
            "[model_router] LD lookup failed for copilot-model-routing — "
            "using config default %s for role %s",
            fallback,
            role,
            exc_info=True,
        )
        return fallback

    if payload is None:
        return fallback

    if not isinstance(payload, dict):
        logger.warning(
            "[model_router] copilot-model-routing expected a JSON object, got %r — "
            "using config default %s for role %s",
            payload,
            fallback,
            role,
        )
        return fallback

    value = payload.get(role)
    if isinstance(value, str) and value.strip():
        return value.strip()
    if value is not None:
        reason = (
            "empty string"
            if isinstance(value, str)
            else f"non-string ({type(value).__name__})"
        )
        logger.warning(
            "[model_router] copilot-model-routing[%s] returned %s — "
            "using config default %s",
            role,
            reason,
            fallback,
        )
    return fallback


async def is_planner_executor_enabled(
    user_id: str | None,
    *,
    config: ChatConfig,
) -> bool:
    """Whether the two-phase planner/executor split is on for this user.

    Per-user ``copilot-planner-executor`` LaunchDarkly boolean, defaulting to
    ``config.planner_executor_enabled`` (the env/local fallback) when LD is
    unreachable.  Anonymous turns (no ``user_id``) can't be targeted, so they
    use the config default directly.  A ``FORCE_FLAG_COPILOT_PLANNER_EXECUTOR``
    env override still short-circuits inside ``is_feature_enabled`` for local
    dev/tests.
    """
    if not user_id:
        return config.planner_executor_enabled
    return await is_feature_enabled(
        Flag.COPILOT_PLANNER_EXECUTOR,
        user_id,
        default=config.planner_executor_enabled,
    )
