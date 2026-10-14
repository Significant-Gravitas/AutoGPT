"""Which route serves an SDK turn, and the gateway it launches behind.

:func:`resolve_turn_profile` builds the turn's route profile once: a
connected ChatGPT account's lease routes the turn to Codex, anything else
runs on the operator's platform route. :func:`start_route_gateway` is the one
place a route's loopback gateway is started; the shared model gateway plugs in
there.
"""

import logging

from backend.copilot.config import ChatConfig, CopilotLLMModel
from backend.copilot.model_normalize import normalize_model_for_transport
from backend.copilot.model_router import resolve_codex_model_route, resolve_model_route
from backend.copilot.route_profile import RouteProfile, codex_profile, platform_profile
from backend.copilot.route_table import RoutingSource
from backend.copilot.sdk.codex_compat_gateway import CodexAnthropicGateway
from backend.copilot.sdk.context_window import CodexEngineWindow
from backend.integrations.codex.transport import CodexCredentialLease
from backend.integrations.credential_lease import CredentialLease

logger = logging.getLogger(__name__)


async def resolve_turn_profile(
    config: ChatConfig,
    *,
    model: CopilotLLMModel | None,
    session_id: str,
    user_id: str | None,
    credential_lease: CredentialLease | CodexCredentialLease | None,
) -> RouteProfile:
    """The profile this SDK turn runs on, for the tier the request asked for."""
    tier: CopilotLLMModel = "advanced" if model == "advanced" else "standard"
    if credential_lease is None:
        sdk_model, source = await resolve_platform_model(
            config, tier, session_id, user_id
        )
        return platform_profile(config, sdk_model, source)
    # The executor hands this engine a Codex lease only.
    if not isinstance(credential_lease, CodexCredentialLease):
        raise TypeError("The SDK engine takes only a Codex credential lease")
    resolved = await resolve_codex_model_route("thinking", tier, credential_lease)
    # The account's advertised window for this model, when it sent one: the
    # pin and every budget derived from it follow the account.
    engine = (
        CodexEngineWindow(
            context_window=resolved.context_window,
            auto_compact_token_limit=resolved.auto_compact_token_limit,
        )
        if resolved.context_window is not None
        else None
    )
    return codex_profile(
        resolved.model,
        effort=resolved.effort,
        routing_source=resolved.source,
        codex_engine=engine,
    )


async def start_route_gateway(
    profile: RouteProfile,
    credential_lease: CredentialLease | CodexCredentialLease | None,
) -> CodexAnthropicGateway | None:
    """Start the loopback gateway *profile* launches behind, or return None.

    Codex runs behind today's Codex compatibility gateway. The shared model
    gateway takes over here: the local route's adapters and Codex's
    Responses adapter start from this function.
    """
    if profile.gateway_adapter is None:
        return None
    if profile.gateway_adapter != "codex_responses" or not isinstance(
        credential_lease, CodexCredentialLease
    ):
        raise RuntimeError(
            f"No gateway serves the {profile.name} route's "
            f"{profile.gateway_adapter} adapter yet"
        )
    gateway = CodexAnthropicGateway(
        agent_session=credential_lease,
        model=profile.model,
        effort=profile.codex_effort,
        credential_id=credential_lease.credentials.id,
    )
    await gateway.start()
    return gateway


def turn_route_name(
    profile: RouteProfile | None,
    credential_lease: CredentialLease | CodexCredentialLease | None,
    config: ChatConfig,
) -> str:
    """The route a turn ran on, also for one that ended before its profile."""
    if profile is not None:
        return profile.name
    return "codex" if credential_lease is not None else config.platform_route


async def resolve_platform_model(
    config: ChatConfig,
    tier: CopilotLLMModel,
    session_id: str,
    user_id: str | None,
) -> tuple[str, RoutingSource]:
    """The platform route's model for *tier*, plus which layer picked it.

    Per-user flag targeting, then the catalog's cell, then the route's tier
    (``CHAT_STANDARD_MODEL`` / ``CHAT_ADVANCED_MODEL`` or the route's
    default). A flag-served name the route cannot serve falls back to the
    tier's own model, never to the other tier's.
    """
    resolved = await resolve_model_route("thinking", tier, user_id, config=config)
    try:
        return normalize_model_for_transport(resolved.model, config), resolved.source
    except ValueError as exc:
        tier_default = config.tier_model(tier)
        if tier_default is None:
            raise
        try:
            sdk_model = normalize_model_for_transport(tier_default, config)
        except ValueError:
            # The tier's own model is invalid for the route too: a deployment
            # misconfiguration the startup check reports; surface it loudly.
            raise exc
        logger.warning(
            "[SDK] [%s] Routed model %r rejected for tier=%s (%s); falling back "
            "to tier default %s",
            session_id[:12] if session_id else "?",
            resolved.model,
            tier,
            exc,
            sdk_model,
        )
        return sdk_model, "env"
