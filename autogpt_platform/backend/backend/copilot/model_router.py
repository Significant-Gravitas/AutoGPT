"""Model selection for the copilot: LaunchDarkly → registry cell → env.

Each cell of the ``(mode, tier)`` matrix resolves through three layers:

1. The JSON-valued LaunchDarkly flag ``copilot-model-routing`` (per-user —
   cohort experiments and rollouts live here, and the flag returns model
   slugs directly).
2. The LLM registry's admin-set routing cell (``LlmModelRoute``, surface
   ``"copilot"``) — the config layer the admin page edits.
3. The static ``ChatConfig`` default (env vars) — the bootstrap floor.

The registry is the serve-time gate for layers 1 and 2: a slug the registry
doesn't know, or one with ``isEnabled=false`` (the kill switch), is refused —
loudly (log + Sentry + route_warnings record) — and resolution falls through
to the next layer. ``visibility=HIDDEN`` models DO serve when routed: that's
the pre-launch testing state (registered + routable, not shown in pickers).
An EMPTY registry never gates anything, so installs where the registry is
dormant keep exact pre-registry behavior.

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
payload, or LD failure all fall through to the next layer.
"""

from __future__ import annotations

import logging
from typing import Literal, NamedTuple

import sentry_sdk

import backend.data.llm_registry as llm_registry
from backend.copilot.config import ChatConfig
from backend.copilot.route_warnings import record_route_warning
from backend.util.feature_flag import Flag, get_feature_flag_value

logger = logging.getLogger(__name__)

ModelMode = Literal["fast", "thinking"]
ModelTier = Literal["standard", "advanced"]
RoutingSource = Literal["ld", "db", "env"]

ROUTE_SURFACE_COPILOT = "copilot"


class ResolvedModel(NamedTuple):
    model: str
    source: RoutingSource


def _catalog_lookup(slug: str):
    """Look up *slug* in the catalog, tolerating transport spellings.

    The catalog registers Claude models under bare canonical enum slugs
    (``claude-opus-4-7``) while LaunchDarkly payloads and env defaults use
    OpenRouter forms (``anthropic/claude-opus-4.6``). Gate on the model,
    not the spelling: try the exact slug, then the ``anthropic/``-stripped
    tail, then its dots→dashes form.
    """
    candidates = [slug]
    if slug.startswith("anthropic/"):
        tail = slug.split("/", 1)[1]
        candidates += [tail, tail.replace(".", "-")]
    elif "/" not in slug and slug.startswith("claude-"):
        candidates.append(slug.replace(".", "-"))
    for candidate in candidates:
        model = llm_registry.get_model(candidate)
        if model is not None:
            return model
    return None


async def _registry_refuses(slug: str, layer: RoutingSource) -> str | None:
    """Return a refusal reason if the registry gates *slug*, else None.

    An empty registry gates nothing (dormant-registry installs must keep
    exact pre-registry behavior). Unknown slug or kill-switched model is
    refused; HIDDEN visibility serves fine when explicitly routed.
    """
    if not llm_registry.get_all_models():
        return None
    model = _catalog_lookup(slug)
    if model is None:
        reason = "unknown to the model registry"
    elif not model.is_enabled:
        reason = "disabled in the model registry (kill switch)"
    else:
        return None
    logger.warning(
        "[model_router] %s-layer slug %r refused: %s — falling through",
        layer,
        slug,
        reason,
    )
    sentry_sdk.capture_message(
        f"copilot routing refused {layer} slug {slug!r}: {reason}",
        level="warning",
    )
    await record_route_warning(slug, reason, layer)
    return reason


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


async def _ld_cell_value(mode: ModelMode, tier: ModelTier, user_id: str) -> str | None:
    """Extract the (mode, tier) slug from the LD JSON flag, or None."""
    try:
        payload: object = await get_feature_flag_value(
            Flag.COPILOT_MODEL_ROUTING.value, user_id, default=None
        )
    except Exception:
        logger.warning(
            "[model_router] LD lookup failed for copilot-model-routing — "
            "falling through for (%s, %s)",
            mode,
            tier,
            exc_info=True,
        )
        return None

    if payload is None:
        return None

    if not isinstance(payload, dict):
        logger.warning(
            "[model_router] copilot-model-routing expected a JSON object, got %r — "
            "falling through for (%s, %s)",
            payload,
            mode,
            tier,
        )
        return None

    mode_cell = payload.get(mode)
    if mode in payload and not isinstance(mode_cell, dict):
        # Operator typed something at the mode level (e.g. a string) instead of
        # a {tier: model} dict — surface the typo in logs.
        logger.warning(
            "[model_router] copilot-model-routing[%s] expected a JSON object, "
            "got %r — falling through for tier %s",
            mode,
            mode_cell,
            tier,
        )
    if not isinstance(mode_cell, dict):
        return None

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
            "falling through",
            mode,
            tier,
            reason,
        )
    return None


async def resolve_model_route(
    mode: ModelMode,
    tier: ModelTier,
    user_id: str | None,
    *,
    config: ChatConfig,
) -> ResolvedModel:
    """Resolve a ``(mode, tier)`` cell through LD → registry cell → env.

    Every layer's slug is validated against the registry (see module
    docstring); a refused slug falls through to the next layer. The returned
    ``source`` is stamped onto persisted chat messages so product
    intelligence can segment quality metrics by model and routing layer.
    """
    if user_id:
        ld_slug = await _ld_cell_value(mode, tier, user_id)
        if ld_slug and await _registry_refuses(ld_slug, "ld") is None:
            return ResolvedModel(ld_slug, "ld")

    # Catalog cells hold CLOUD slugs, and the local transport (Ollama/vLLM)
    # passes slugs through verbatim with no ValueError — a cell would
    # override the operator's CHAT_*_MODEL config with a model their backend
    # doesn't serve and 404 at request time. Local deployments resolve
    # LD → env only, exactly as before the catalog existed.
    if config.baseline_provider == "local":
        return ResolvedModel(_config_default(config, mode, tier).strip(), "env")

    cell_slug = llm_registry.get_route(ROUTE_SURFACE_COPILOT, mode, tier)
    if cell_slug and await _registry_refuses(cell_slug, "db") is None:
        # Catalog cells hold bare canonical Claude slugs; the OpenRouter
        # transport needs the vendor prefix (a bare ``claude-*`` would 404
        # there) and the direct-Anthropic transport strips it right back in
        # ``normalize_model_for_transport``. Prefix here so one cell value
        # works on every transport.
        if "/" not in cell_slug and cell_slug.startswith("claude-"):
            cell_slug = f"anthropic/{cell_slug}"
        return ResolvedModel(cell_slug, "db")

    return ResolvedModel(_config_default(config, mode, tier).strip(), "env")


async def resolve_model(
    mode: ModelMode,
    tier: ModelTier,
    user_id: str | None,
    *,
    config: ChatConfig,
) -> str:
    """Back-compat wrapper for callers that only need the model id."""
    resolved = await resolve_model_route(mode, tier, user_id, config=config)
    return resolved.model
