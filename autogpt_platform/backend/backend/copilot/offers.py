"""Server-owned description of the AI connections a user can chat over.

``transports`` answers "which routes exist and which is default". That is
enough to route a turn, but not enough to render one: the client has been
deciding what a connection is called, what backs it, and which provider it
belongs to. Those are product and billing statements, and a client that
computes them drifts from the server that enforces them.

This module states them once, on the side that knows. It is additive —
``GET /transports`` keeps its shape, so nothing has to migrate at once.
"""

import logging

from pydantic import BaseModel

from backend.copilot.config import ChatConfig, CopilotLlmAuthProvider, CopilotLLMModel
from backend.copilot.engine import resolve_use_sdk
from backend.copilot.model_router import (
    ROUTE_SURFACE_CODEX,
    ModelMode,
    resolve_model_route,
)
from backend.copilot.transports import (
    ChatTransportResponse,
    get_chat_transports,
    is_deployment_chat_available,
    settings,
)
from backend.data import llm_registry
from backend.integrations.codex.access import CODEX_MINIMUM_PLAN_ERROR, has_codex_access
from backend.util.entitlements import Entitlement, has_entitlement
from backend.util.feature_flag import Flag, is_feature_enabled
from backend.util.settings import BehaveAs

logger = logging.getLogger(__name__)

# Product vocabulary. "Fast" and "Thinking" are execution paths, not tiers a
# user picks between; the two labels below are what the product exposes.
TIER_LABELS: dict[CopilotLLMModel, str] = {
    "standard": "Balanced",
    "advanced": "Advanced",
}


class ConnectionTier(BaseModel):
    """One quality level on a connection.

    ``display_model`` is the model this tier resolves to, run through the
    same router the turn will use — LaunchDarkly cell, then registry, then
    config — so it cannot drift from what actually answers.

    On a ChatGPT connection it is the model the catalog pins for that cell,
    read straight from the registry -- no lease, no call to the account.
    What a lease would add is confirmation that this particular account still
    advertises it; the router checks that when the turn runs and falls back
    if not, so the name here is the routed model rather than a guarantee.
    ``None`` only when the catalog pins nothing for the cell.
    """

    tier: CopilotLLMModel
    label: str
    selectable: bool
    display_model: str | None = None
    # Why this tier cannot be picked, when it cannot. Advanced is a paid
    # tier on hosted, so the row stays visible and says what unlocks it --
    # hiding it would remove the upgrade reason it exists to create.
    lock_reason: str | None = None


class AIConnectionOffer(BaseModel):
    offer_id: str
    provider_family: str
    display_name: str
    auth_method: str
    credential_id: str | None
    backed_by_label: str
    description: str
    state: str
    selectable: bool
    is_default: bool
    tiers: list[ConnectionTier]
    limitations: list[str]
    lock_reason: str | None = None
    unlock_href: str | None = None


class AIConnectionOffersResponse(BaseModel):
    offers: list[AIConnectionOffer]


async def get_connection_offers(user_id: str) -> list[AIConnectionOffer]:
    """Describe every connection, naming the models where they are knowable.

    The engine is resolved once for the user rather than per offer: it is a
    property of the deployment and the user, not of which connection they
    pick.
    """
    config = ChatConfig()
    # One engine decision for the whole response: it is a property of the
    # deployment and the user, not of which connection they pick.
    use_sdk = await resolve_use_sdk(
        user_id,
        use_claude_code_subscription=config.use_claude_code_subscription,
        config_default=config.use_claude_agent_sdk,
        thinking_available=config.thinking_available,
    )
    mode = "thinking" if use_sdk else "fast"
    models = await _platform_tier_models(mode, user_id, config)
    codex_models = _codex_tier_models(mode)
    advanced_allowed = await _advanced_tier_allowed(user_id)
    offers = [
        _offer(transport, models, codex_models, advanced_allowed)
        for transport in await get_chat_transports(user_id)
    ]
    locked = await _locked_codex_offer(user_id, offers, codex_models)
    return offers + ([locked] if locked else [])


async def _locked_codex_offer(
    user_id: str,
    offers: list[AIConnectionOffer],
    codex_models: dict[CopilotLLMModel, str | None],
) -> AIConnectionOffer | None:
    """The ChatGPT connection a plan does not include, shown rather than hidden.

    ``get_chat_transports`` answers what may run, and it is right to omit a
    transport the entitlement forbids -- anything that routes a turn reads
    that list, and an unroutable entry there would be a bug waiting to
    happen. What it cannot say is *why* the connection is missing, so a user
    below the plan sees no ChatGPT at all and no way to learn one exists.

    This offer says so, in the one place that exists to describe rather than
    route. It is never selectable and carries no credential, so nothing can
    accidentally send a turn down it.
    """
    if any(offer.provider_family == "openai" for offer in offers):
        return None
    if not _is_hosted():
        # Self-host grants the entitlement outright; there is nothing to sell.
        return None
    try:
        entitled = await has_codex_access(user_id)
    except Exception:
        # A missing upsell is harmless; falsely telling an already-entitled
        # user to buy a plan during an entitlement outage is not.
        logger.warning(
            "Unable to resolve Codex entitlement for user %s; hiding upsell",
            user_id,
            exc_info=True,
        )
        return None
    if entitled:
        # Entitled but unconnected: the settings page owns that invitation.
        return None
    if not await is_feature_enabled(Flag.CHAT_CONNECTION_UPSELL, user_id):
        return None

    return AIConnectionOffer(
        offer_id="codex:locked",
        provider_family="openai",
        display_name="ChatGPT",
        auth_method="chatgpt_oauth",
        credential_id=None,
        backed_by_label="Your ChatGPT plan",
        description=(
            "Run chats on a ChatGPT plan you already pay for, spending no "
            "AutoGPT credits."
        ),
        state="locked",
        selectable=False,
        is_default=False,
        # Named even though nothing here can be picked: "what you get" is the
        # whole argument for connecting, and a surface that cannot say which
        # models is asking the user to take it on faith. Not selectable, so
        # naming them cannot be mistaken for offering them.
        tiers=[
            ConnectionTier(
                tier=tier,
                label=label,
                selectable=False,
                display_model=codex_models.get(tier),
            )
            for tier, label in TIER_LABELS.items()
        ],
        limitations=[],
        lock_reason=CODEX_MINIMUM_PLAN_ERROR,
        unlock_href="/settings/billing",
    )


async def _platform_tier_models(
    mode: ModelMode, user_id: str, config: ChatConfig
) -> dict[CopilotLLMModel, str | None]:
    """Resolve each tier against the engine this user's turns will run on.

    Answerable at all only because nothing can name an engine per request
    any more — the decision is the server's, so it can be made before a turn
    exists rather than during one.
    """
    resolved: dict[CopilotLLMModel, str | None] = {}
    for tier in TIER_LABELS:
        try:
            route = await resolve_model_route(mode, tier, user_id, config=config)
            resolved[tier] = _display_name(route.model)
        except Exception:
            # A tier that cannot be resolved is described without a name
            # rather than failing the whole list.
            logger.warning(
                "Could not resolve the %s model for the platform connection",
                tier,
                exc_info=True,
            )
            resolved[tier] = None
    return resolved


def offer_id_for(transport: ChatTransportResponse) -> str:
    """Stable across requests, and unique per account.

    The platform route carries no credential, and a user may hold several
    ChatGPT accounts, so neither half identifies an offer on its own.
    """
    return f"{transport.auth_provider}:{transport.credential_id or 'deployment'}"


ADVANCED_TIER_LOCK_REASON = "A Max plan or higher is required for Advanced."


async def _advanced_tier_allowed(user_id: str) -> bool:
    """Whether this user may pick the Advanced tier.

    A failure to resolve the entitlement leaves the tier open rather than
    locked: a billing hiccup should not silently downgrade someone's model.
    """
    try:
        return await has_entitlement(user_id, Entitlement.ADVANCED_MODEL_TIER)
    except Exception:
        logger.warning(
            "Could not resolve the Advanced tier entitlement; leaving it open",
            exc_info=True,
        )
        return True


def _display_name(slug: str | None) -> str | None:
    """What the catalog calls this model, rather than its slug.

    The PRD labels tiers "Balanced · 5.6 Terra", not
    "Balanced · gpt-5.6-terra": the slug is a routing key and reads as one.
    Falls back to the slug when the registry has no entry, which is better
    than showing nothing.
    """
    if not slug:
        return None
    model = llm_registry.get_model(slug)
    return model.display_name if model else slug


def _codex_tier_models(mode: str) -> dict[CopilotLLMModel, str | None]:
    """The models the catalog pins for the Codex cells of this engine.

    A registry read, so it costs nothing and needs no credential. The router
    validates the pinned slug against what the account actually advertises
    when the turn runs, and falls back if it is gone -- so this names the
    routed model, not a promise about the account.
    """
    return {
        tier: _display_name(llm_registry.get_route(ROUTE_SURFACE_CODEX, mode, tier))
        for tier in TIER_LABELS
    }


def _offer(
    transport: ChatTransportResponse,
    platform_models: dict[CopilotLLMModel, str | None],
    codex_models: dict[CopilotLLMModel, str | None],
    advanced_allowed: bool,
) -> AIConnectionOffer:
    return AIConnectionOffer(
        offer_id=offer_id_for(transport),
        provider_family=_provider_family(transport.auth_provider),
        display_name=transport.label,
        auth_method=_auth_method(transport.auth_provider),
        credential_id=transport.credential_id,
        backed_by_label=_backed_by_label(transport),
        description=_description(transport),
        state="ready" if transport.available else "unavailable",
        selectable=transport.available,
        is_default=transport.default,
        tiers=[
            ConnectionTier(
                tier=tier,
                label=label,
                selectable=(
                    transport.available and (advanced_allowed or tier != "advanced")
                ),
                display_model=(
                    platform_models.get(tier)
                    if transport.auth_provider == "platform"
                    else codex_models.get(tier)
                ),
                lock_reason=(
                    None
                    if advanced_allowed or tier != "advanced"
                    else ADVANCED_TIER_LOCK_REASON
                ),
            )
            for tier, label in TIER_LABELS.items()
        ],
        limitations=_limitations(transport),
    )


def _provider_family(auth_provider: CopilotLlmAuthProvider) -> str:
    # Presentation grouping, not the credential discriminator: ChatGPT OAuth
    # and an OpenAI API key are one family and two very different credentials.
    return "openai" if auth_provider == "codex" else "autogpt"


def _auth_method(auth_provider: CopilotLlmAuthProvider) -> str:
    return "chatgpt_oauth" if auth_provider == "codex" else "deployment"


def _is_hosted() -> bool:
    return settings.config.behave_as == BehaveAs.CLOUD


def _backed_by_label(transport: ChatTransportResponse) -> str:
    if transport.auth_provider == "codex":
        return "Your ChatGPT plan"
    return "Your AutoGPT plan" if _is_hosted() else "This server's chat provider"


def _description(transport: ChatTransportResponse) -> str:
    """What backs a run, in the user's terms.

    The credits contrast only means something where credits exist, so the
    self-hosted line says what it actually has instead of denying something
    that was never there.
    """
    if transport.auth_provider == "codex":
        return (
            "New chats are backed by your ChatGPT plan, and spend no "
            "AutoGPT credits."
        )
    if _is_hosted():
        return "New chats are backed by your AutoGPT plan, and spend AutoGPT credits."
    return "New chats are backed by the chat provider configured on this server."


def _limitations(transport: ChatTransportResponse) -> list[str]:
    limitations: list[str] = []
    if transport.auth_provider == "codex":
        # Stated because it is a real edge a user can hit, not a policy note:
        # the builder panel rejects a codex route outright.
        limitations.append("The agent builder's chat panel always runs on AutoGPT.")
    if transport.auth_provider == "platform" and not is_deployment_chat_available():
        limitations.append("No chat provider is configured on this server yet.")
    return limitations
