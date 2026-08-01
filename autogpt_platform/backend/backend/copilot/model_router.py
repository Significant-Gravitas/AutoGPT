"""Model selection for the copilot: LaunchDarkly → registry cell → env.

Each cell of the ``(mode, tier)`` matrix resolves through three layers:

1. The JSON-valued LaunchDarkly flag ``copilot-model-routing`` (per-user —
   cohort experiments and rollouts live here, and the flag returns model
   slugs directly).
2. The catalog's routing cell (``catalog.py`` ``routing`` section, surface
   ``"copilot"``) — our cloud's deployment config, shipped with the code.
3. The static ``ChatConfig`` default (env vars) — the bootstrap floor.

On our cloud, the registry is the serve-time gate for layers 1 and 2: a slug
the catalog doesn't know, or one with ``is_enabled=False`` (the kill switch),
is refused — loudly (log + Sentry) — and resolution falls through to the next
layer. ``visibility=HIDDEN`` models DO serve when routed: that's the
pre-launch testing state (registered + routable, not shown in pickers).
Self-hosted installs and local transports skip the gate entirely (their LD
and env slugs are their own business — the shipped catalog must not veto an
operator's custom model). An EMPTY registry never gates anything either;
production processes always load it (fail-hard boot), so that branch is
defense-in-depth for exotic embedders only.

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
from backend.copilot.model import RoutingSource
from backend.data.llm_registry.llm_models import LLMModel, transport_slug_candidates
from backend.util.feature_flag import Flag, get_feature_flag_value
from backend.util.settings import BehaveAs, Settings

logger = logging.getLogger(__name__)
settings = Settings()

ModelMode = Literal["fast", "thinking"]
ModelTier = Literal["standard", "advanced"]

ROUTE_SURFACE_COPILOT = "copilot"


class ResolvedModel(NamedTuple):
    model: str
    source: RoutingSource


def _catalog_lookup(slug: str) -> "llm_registry.RegistryModel | None":
    """Look up *slug* in the catalog, tolerating transport spellings.

    The catalog registers Claude models under bare canonical enum slugs
    (``claude-opus-4-7``) while LaunchDarkly payloads and env defaults use
    OpenRouter forms (``anthropic/claude-opus-4.6``). Gate on the model,
    not the spelling: try the exact slug, then the ``anthropic/``-stripped
    tail, then its dots→dashes form.
    """
    candidates = transport_slug_candidates(slug)
    for candidate in candidates:
        model = llm_registry.get_model(candidate)
        if model is not None:
            return model
    # Anthropic's API slugs carry a -YYYYMMDD snapshot suffix that the
    # OpenRouter canonical form drops (anthropic/claude-haiku-4-5 ↔ catalog
    # claude-haiku-4-5-20251001) — resolve via the date-stripped index
    # built at catalog load (O(1), no per-turn scan).
    for candidate in candidates:
        model = llm_registry.get_model_by_date_stripped_slug(candidate)
        if model is not None:
            return model
    # Transport spellings are fuzzy (prefix-strip, dot/dash, snapshot-date)
    # and deliberately don't know the enum's exact alias map, so an
    # OpenRouter OpenAI slug whose catalog entry carries a -YYYY-MM-DD
    # snapshot (openai/gpt-5.4 → gpt-5.4-2026-03-05) slips past them. Honor
    # the enum's alias resolution as a final gate so an LD/env cell set to
    # such a slug resolves instead of being refused as unknown.
    try:
        member = LLMModel(slug)
    except ValueError:
        return None
    return llm_registry.get_model(member.value)


_sentry_reported: set[tuple[str, str]] = set()
_unloaded_reported = False


async def _registry_refuses(slug: str, layer: RoutingSource) -> str | None:
    """Return a refusal reason if the registry gates *slug*, else None.

    An empty registry gates nothing (dormant-registry installs must keep
    exact pre-registry behavior). Unknown slug or kill-switched model is
    refused; HIDDEN visibility serves fine when explicitly routed.
    """
    if not llm_registry.has_models():
        global _unloaded_reported
        if not llm_registry.is_loaded() and not _unloaded_reported:
            # Empty-because-dormant is legitimate; empty-because-nobody-
            # called-load_catalog() in this process is a wiring bug that
            # would silently disable gating and cells — say so, once.
            _unloaded_reported = True
            logger.error(
                "[model_router] registry gating skipped: load_catalog() was "
                "never called in this process — routing cells and serve-time "
                "gating are inactive"
            )
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
    # Log every refusal (greppable), but Sentry only once per (layer, slug)
    # per process — a bad LD slug refuses on EVERY turn until fixed, and one
    # event per turn during an incident is noise, not signal.
    if (layer, slug) not in _sentry_reported:
        _sentry_reported.add((layer, slug))
        sentry_sdk.capture_message(
            f"copilot routing refused {layer} slug {slug!r}: {reason}",
            level="warning",
        )
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


async def _env_floor(
    config: ChatConfig, mode: ModelMode, tier: ModelTier
) -> ResolvedModel:
    """Serve the env default — the LAST layer, served even when the catalog
    refuses it (refusing would leave nothing). A kill switch pointing here
    is an incident the operator must hear about: log + Sentry, then serve.
    """
    env_slug = _config_default(config, mode, tier).strip()
    if await _registry_refuses(env_slug, "env") is not None:
        logger.error(
            "[model_router] env default %r is refused by the catalog "
            "but served anyway (last-resort floor) — change the "
            "CHAT_*_MODEL default or the routing cell",
            env_slug,
        )
    return ResolvedModel(env_slug, "env")


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
    # The catalog gates (and cells apply) on OUR CLOUD's hosted transports
    # only — they are not rules for everyone:
    # - self-hosted installs (behave_as != CLOUD) keep LD/env authority over
    #   their own slugs; the shipped catalog must not veto an operator's
    #   custom model (they can't edit our catalog to register it)
    # - local transports (Ollama/vLLM) pass slugs through verbatim; catalog
    #   gating would refuse every local model and a cloud-slug cell would
    #   404 at request time
    # Both resolve LD → env, exactly as before the catalog existed.
    gated = (
        settings.config.behave_as == BehaveAs.CLOUD
        and config.baseline_provider != "local"
    )

    if user_id:
        ld_slug = await _ld_cell_value(mode, tier, user_id)
        if ld_slug and (not gated or await _registry_refuses(ld_slug, "ld") is None):
            return ResolvedModel(ld_slug, "ld")

    if not gated:
        return ResolvedModel(_config_default(config, mode, tier).strip(), "env")

    cell_slug = llm_registry.get_route(ROUTE_SURFACE_COPILOT, mode, tier)
    if cell_slug and await _registry_refuses(cell_slug, "catalog") is None:
        # Cells carry TRANSPORT-READY spellings (e.g. the vendor-prefixed
        # dot form ``anthropic/claude-sonnet-4.6`` OpenRouter serves) and are
        # returned verbatim; the catalog guard tests enforce the convention,
        # and the slug-tolerant gate above maps them to catalog identity.
        return ResolvedModel(cell_slug, "catalog")

    return await _env_floor(config, mode, tier)
