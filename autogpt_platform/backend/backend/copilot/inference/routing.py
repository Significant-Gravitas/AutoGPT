"""The route a background call takes, and the platform key it is made with.

Today each call routed here follows the deployment's chat transport
(``copilot/transport_routing.py``): its provider, the platform's own key for
it, and the fast model of the job's tier. Nothing is per user yet; the scope
is taken so a later resolver can be. The dream's Anthropic batch path is still
chosen by the orchestrator (``dream/routing.py``); ``anthropic_batch_route``
only describes it, so its cost rows are recorded like the sync calls'.
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

NO_PLATFORM_API = (
    "The platform route has no API for background calls: a Claude "
    "subscription (CHAT_PLATFORM_ROUTE=subscription) or a disabled route. "
    "Choose an API route (openrouter, direct_anthropic or local) to run them."
)

# The ChatConfig field each tier reads, named in the route's ``reason``.
_TIER_FIELDS: dict[InferenceTier, str] = {
    "standard": "standard_model",
    "advanced": "advanced_model",
    "aux": "aux_model",
}


def resolve_route(
    scope: InferenceScope, job: InferenceJob, *, config: ChatConfig | None = None
) -> RouteDecision:
    """The chat transport's provider and platform key, on the job's model.

    The model is the job's pinned one, else its tier's ``ChatConfig`` field,
    in the native spelling when the provider is the Anthropic API. A local
    backend bills nobody, so its calls are paid ``local``; everything else
    comes out of the platform allowance.

    Raises ``InferenceError`` when the platform has no key for the
    transport's provider, or when the model cannot go to that provider (a
    non-Anthropic model on the Anthropic API). The key is checked first, as
    the callers' old ``structured_completion`` did, so a subscription install
    without ``ANTHROPIC_API_KEY`` is told about the key and the OAuth token
    before anything about its models.
    """
    transport = routing_kwargs_for_chat_transport()
    _require_platform_key(transport)
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
    (``inference/execution_path.batch_discount``), whatever the chat transport."""
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
    _require_platform_key(transport)
    return transport


def _require_platform_key(transport: ProviderRoutingKwargs) -> None:
    """Refuse a transport the platform has no key for; a local backend takes
    none."""
    if transport.provider is None:
        raise InferenceError(NO_PLATFORM_API)
    if not transport.api_key and transport.provider != "ollama":
        raise InferenceError(_missing_api_key_message(transport.provider))


def _tier_model(config: ChatConfig, tier: InferenceTier) -> str:
    match tier:
        case "standard":
            return config.job_tier_model("standard")
        case "advanced":
            return config.job_tier_model("advanced")
        case "aux":
            return config.aux_model or ""


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
    why the OAuth token is not enough. The calling job reports this error.
    """
    if provider == "anthropic":
        return "The direct_anthropic route's key is not set: CHAT_ANTHROPIC_API_KEY."
    if provider == "open_router":
        return "The openrouter route's key is not set: CHAT_OPENROUTER_API_KEY."
    return f"No API key configured for background inference provider={provider!r}."
