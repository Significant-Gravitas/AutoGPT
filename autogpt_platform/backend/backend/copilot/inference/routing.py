"""The route a background call takes, and the platform key it is made with.

Today every background call follows the deployment's chat transport
(``copilot/transport_routing.py``): its provider, the platform's own key for
it, and the fast model of the job's tier. Nothing is per user yet; the scope
is taken so a later resolver can be. The dream's Anthropic batch path is still
chosen by the orchestrator (``dream/routing.py``); ``anthropic_batch_route``
only describes it, so its cost rows are recorded like any other call's.
"""

from backend.copilot.config import ChatConfig
from backend.copilot.model_normalize import normalize_model_for_anthropic
from backend.copilot.transport_routing import (
    ProviderRoutingKwargs,
    routing_kwargs_for_chat_transport,
)
from backend.util.llm.providers import ProviderLiteral

from .context import (
    InferenceError,
    InferenceJob,
    InferenceScope,
    InferenceTier,
    RouteDecision,
)

# The ChatConfig field each tier reads, named in the route's ``reason``.
_TIER_FIELDS: dict[InferenceTier, str] = {
    "standard": "fast_standard_model",
    "advanced": "fast_advanced_model",
    "aux": "title_model",
}


def resolve_route(
    scope: InferenceScope, job: InferenceJob, *, config: ChatConfig | None = None
) -> RouteDecision:
    """The chat transport's provider and platform key, on the job's model.

    The model is the job's pinned one, else its tier's ``ChatConfig`` field,
    in the native spelling when the provider is the Anthropic API. A local
    backend bills nobody, so its calls are paid ``local``; everything else
    comes out of the platform allowance.

    Raises ``InferenceError`` when the model cannot go to the provider (a
    non-Anthropic model on the Anthropic API): the call fails exactly as it
    would have at dispatch.
    """
    transport = routing_kwargs_for_chat_transport()
    if job.pinned_model:
        model, source = job.pinned_model, "the job's pinned model"
    else:
        model = _tier_model(config or ChatConfig(), job.tier)
        source = f"{job.tier} tier ({_TIER_FIELDS[job.tier]})"
    return RouteDecision(
        engine="provider_sync",
        auth_provider="platform",
        provider=transport.provider,
        model=_dispatch_model(transport.provider, model),
        payer="local" if transport.provider == "ollama" else "platform_allowance",
        execution_path="sync_baseline",
        cost_log_provider=transport.cost_log_provider,
        reason=f"{transport.cost_log_provider} chat transport, {source}",
    )


def anthropic_batch_route(model: str) -> RouteDecision:
    """The route of a dream phase the orchestrator sent to Anthropic's
    Message Batches API: the platform's Anthropic key, at the batch discount
    (``dream/routing.batch_discount``), whatever the chat transport."""
    return RouteDecision(
        engine="provider_batch",
        auth_provider="platform",
        provider="anthropic",
        model=model,
        payer="platform_allowance",
        execution_path="anthropic_batch",
        cost_log_provider="anthropic",
        reason="dream batch path: Anthropic Message Batches API",
    )


def platform_credentials(route: RouteDecision) -> ProviderRoutingKwargs:
    """The platform's key (and base URL) for *route*'s provider.

    Raises ``InferenceError`` when the chat transport dispatches somewhere
    else, or when no key is configured for it (a local backend takes none).
    """
    transport = routing_kwargs_for_chat_transport()
    if transport.provider != route.provider:
        raise InferenceError(
            f"The platform's key is for {transport.provider!r}, "
            f"not the route's provider {route.provider!r}."
        )
    if not transport.api_key and route.provider != "ollama":
        raise InferenceError(_missing_api_key_message(route.provider))
    return transport


def _tier_model(config: ChatConfig, tier: InferenceTier) -> str:
    match tier:
        case "standard":
            return config.fast_standard_model
        case "advanced":
            return config.fast_advanced_model
        case "aux":
            return config.title_model


def _dispatch_model(provider: ProviderLiteral, model: str) -> str:
    """The native Anthropic API takes its own spelling; every other provider
    takes the configured one."""
    if provider != "anthropic":
        return model
    try:
        return normalize_model_for_anthropic(model)
    except ValueError as exc:
        raise InferenceError(str(exc)) from exc


def _missing_api_key_message(provider: ProviderLiteral) -> str:
    """Per-provider friendly error for the no-API-key case.

    Names the env var the operator needs to set and, for subscription mode,
    why the OAuth token is not enough. Surfaced through the dream's JobStatus
    ``error`` so an operator can fix it without a logs dive.
    """
    if provider == "anthropic":
        return (
            "Anthropic API key not configured — set ANTHROPIC_API_KEY to "
            "enable the dream pass under subscription / direct-Anthropic "
            "mode. The Claude Code OAuth token cannot be used for direct "
            "Messages API calls (see "
            "docs/platform/copilot-local-llm.md#subscription-mode-caveat)."
        )
    if provider == "open_router":
        return "OpenRouter API key not configured — set OPEN_ROUTER_API_KEY."
    return f"No API key configured for dream pass provider={provider!r}."
