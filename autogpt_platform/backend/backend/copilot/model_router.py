"""Model selection for the copilot: LaunchDarkly → registry cell → env.

Each tier resolves through three layers:

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

LD payload shape (flat)::

    {
      "standard": "anthropic/claude-sonnet-4-6",
      "advanced": "anthropic/claude-opus-4-6"
    }

The pre-collapse nested shape (``{"thinking": {"standard": ...}}``) is
still honored so in-flight flag edits keep working — re-author flat at
leisure. A missing tier, a non-string cell value, a non-dict payload, or
LD failure all fall through to the next layer.
"""

from __future__ import annotations

import logging
from typing import Literal, NamedTuple

import sentry_sdk

import backend.data.llm_registry as llm_registry
from backend.copilot.config import ChatConfig
from backend.copilot.model import RoutingSource
from backend.data.llm_registry.llm_models import LLMModel, transport_slug_candidates
from backend.integrations.codex.models import CodexModelInfo, CodexReasoningEffort
from backend.integrations.codex.transport import (
    CodexCredentialLease,
    get_codex_transport,
)
from backend.integrations.credential_lease import CredentialLease
from backend.util.feature_flag import Flag, get_feature_flag_value
from backend.util.settings import BehaveAs, Settings

logger = logging.getLogger(__name__)
settings = Settings()

ModelTier = Literal["standard", "advanced"]
CodexRoutingSource = Literal[
    "catalog",
    "preferred",
    "account_default",
    "account_available",
]

ROUTE_SURFACE_COPILOT = "copilot"
ROUTE_SURFACE_CODEX = "copilot_codex"


class ResolvedModel(NamedTuple):
    model: str
    source: RoutingSource


class ResolvedCodexModel(NamedTuple):
    model: str
    effort: CodexReasoningEffort | None
    source: CodexRoutingSource


_CODEX_PREFERRED_MODELS: dict[ModelTier, str] = {
    "standard": LLMModel.GPT5_6_TERRA.value,
    "advanced": LLMModel.GPT6_ASTRA.value,
}

_CODEX_PREFERRED_EFFORTS: dict[ModelTier, CodexReasoningEffort] = {
    "standard": "high",
    "advanced": "xhigh",
}


def catalog_lookup(slug: str) -> llm_registry.RegistryModel | None:
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
    model = catalog_lookup(slug)
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


def _config_default(config: ChatConfig, tier: ModelTier) -> str:
    return (
        config.thinking_advanced_model
        if tier == "advanced"
        else config.thinking_standard_model
    )


async def _ld_cell_value(tier: ModelTier, user_id: str) -> str | None:
    """Extract the tier slug from the LD JSON flag, or None.

    Reads the flat shape (``{"standard": ...}``); the pre-collapse
    nested shape (``{"thinking": {"standard": ...}}``) is honored while
    operators re-author. Flat wins when both are present.
    """
    try:
        payload: object = await get_feature_flag_value(
            Flag.COPILOT_MODEL_ROUTING.value, user_id, default=None
        )
    except Exception:
        logger.warning(
            "[model_router] LD lookup failed for copilot-model-routing — "
            "falling through for %s",
            tier,
            exc_info=True,
        )
        return None

    if payload is None:
        return None

    if not isinstance(payload, dict):
        logger.warning(
            "[model_router] copilot-model-routing expected a JSON object, got %r — "
            "falling through for %s",
            payload,
            tier,
        )
        return None

    value: object = payload.get(tier)
    if value is None:
        nested = payload.get("thinking")
        if isinstance(nested, dict):
            value = nested.get(tier)
    if isinstance(value, str) and value.strip():
        return value.strip()
    if value is not None:
        reason = (
            "empty string"
            if isinstance(value, str)
            else f"non-string ({type(value).__name__})"
        )
        logger.warning(
            "[model_router] copilot-model-routing[%s] returned %s — " "falling through",
            tier,
            reason,
        )
    return None


async def _env_floor(config: ChatConfig, tier: ModelTier) -> ResolvedModel:
    """Serve the env default — the LAST layer, served even when the catalog
    refuses it (refusing would leave nothing). A kill switch pointing here
    is an incident the operator must hear about: log + Sentry, then serve.
    """
    env_slug = _config_default(config, tier).strip()
    if await _registry_refuses(env_slug, "env") is not None:
        logger.error(
            "[model_router] env default %r is refused by the catalog "
            "but served anyway (last-resort floor) — change the "
            "CHAT_*_MODEL default or the routing cell",
            env_slug,
        )
    return ResolvedModel(env_slug, "env")


async def resolve_model_route(
    tier: ModelTier,
    user_id: str | None,
    *,
    config: ChatConfig,
) -> ResolvedModel:
    """Resolve a tier through LD → registry cell → env.

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
        settings.config.behave_as == BehaveAs.CLOUD and config.transport.name != "local"
    )

    if user_id:
        ld_slug = await _ld_cell_value(tier, user_id)
        if ld_slug and (not gated or await _registry_refuses(ld_slug, "ld") is None):
            return ResolvedModel(ld_slug, "ld")

    if not gated:
        return ResolvedModel(_config_default(config, tier).strip(), "env")

    cell_slug = llm_registry.get_route(ROUTE_SURFACE_COPILOT, tier)
    if cell_slug and await _registry_refuses(cell_slug, "catalog") is None:
        # Cells carry TRANSPORT-READY spellings (e.g. the vendor-prefixed
        # dot form ``anthropic/claude-sonnet-4.6`` OpenRouter serves) and are
        # returned verbatim; the catalog guard tests enforce the convention,
        # and the slug-tolerant gate above maps them to catalog identity.
        return ResolvedModel(cell_slug, "catalog")

    return await _env_floor(config, tier)


async def resolve_codex_model_route(
    tier: ModelTier,
    credential_lease: CredentialLease | CodexCredentialLease,
) -> ResolvedCodexModel:
    """Resolve a Codex model against both the catalog and the account."""
    advertised = await _advertised_codex_models(credential_lease)

    if catalog_route := _codex_catalog_route(advertised, tier):
        return catalog_route

    if preferred_route := _codex_preferred_route(advertised, tier):
        return preferred_route

    if account_route := _codex_account_route(advertised, tier):
        return account_route

    raise RuntimeError("codex_model_unavailable")


async def _advertised_codex_models(
    credential_lease: CredentialLease | CodexCredentialLease,
) -> list[CodexModelInfo]:
    if isinstance(credential_lease, CodexCredentialLease):
        return await credential_lease.models()
    return await get_codex_transport().models(credential_lease)


def _codex_catalog_route(
    advertised: list[CodexModelInfo],
    tier: ModelTier,
) -> ResolvedCodexModel | None:
    catalog_slug = llm_registry.get_route(ROUTE_SURFACE_CODEX, tier)
    if not catalog_slug:
        return None

    model = next((item for item in advertised if item.model == catalog_slug), None)
    if model is not None and _codex_catalog_allows(catalog_slug):
        return _resolved_codex_model(model, tier, "catalog")

    logger.warning(
        "[model_router] Codex catalog route %r is disabled or unavailable "
        "for this account; falling through for %s",
        catalog_slug,
        tier,
    )
    return None


def _codex_preferred_route(
    advertised: list[CodexModelInfo],
    tier: ModelTier,
) -> ResolvedCodexModel | None:
    preferred_slug = _CODEX_PREFERRED_MODELS[tier]
    preferred = next(
        (model for model in advertised if model.model == preferred_slug),
        None,
    )
    if preferred is None or not _codex_catalog_allows(preferred_slug):
        return None
    return _resolved_codex_model(preferred, tier, "preferred")


def _codex_account_route(
    advertised: list[CodexModelInfo],
    tier: ModelTier,
) -> ResolvedCodexModel | None:
    candidates: tuple[tuple[CodexRoutingSource, bool], ...] = (
        ("account_default", True),
        ("account_available", False),
    )
    for source, default_only in candidates:
        model = next(
            (
                item
                for item in advertised
                if (not default_only or item.is_default)
                and not item.hidden
                and _codex_account_fallback_allowed(item.model)
            ),
            None,
        )
        if model is not None:
            return _resolved_codex_model(model, tier, source)
    return None


def _resolved_codex_model(
    model: CodexModelInfo,
    tier: ModelTier,
    source: CodexRoutingSource,
) -> ResolvedCodexModel:
    return ResolvedCodexModel(
        model.model,
        _codex_effort(model, tier),
        source,
    )


def _codex_catalog_allows(slug: str) -> bool:
    if not llm_registry.has_models():
        return True
    model = catalog_lookup(slug)
    return bool(
        model is not None and model.is_enabled and model.metadata.provider == "openai"
    )


def _codex_account_fallback_allowed(slug: str) -> bool:
    if not llm_registry.has_models():
        return True
    model = catalog_lookup(slug)
    if model is None:
        return True
    return model.is_enabled and model.metadata.provider == "openai"


def _codex_effort(
    model: CodexModelInfo,
    tier: ModelTier,
) -> CodexReasoningEffort | None:
    preferred = _CODEX_PREFERRED_EFFORTS[tier]
    if preferred in model.supported_reasoning_efforts:
        return preferred
    if model.default_reasoning_effort in model.supported_reasoning_efforts:
        return model.default_reasoning_effort
    return None
