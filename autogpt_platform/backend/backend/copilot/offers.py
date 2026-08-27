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
from backend.copilot.provider_tiers import (
    TIER_LABELS,
    codex_tier_models,
    platform_tier_models,
    resolve_engine_mode,
)
from backend.copilot.transports import (
    ChatTransportResponse,
    get_chat_transports,
    is_deployment_chat_available,
    settings,
)
from backend.copilot.subscription_providers import (
    linked_profiles,
    profile_for,
)
from backend.util.entitlements import (
    Entitlement,
    has_entitlement,
    has_entitlement_for_discovery,
)
from backend.util.feature_flag import Flag, is_feature_enabled
from backend.util.settings import BehaveAs

logger = logging.getLogger(__name__)


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
    # What the sign-in button says, and whose terms it sends the user to.
    # Server-owned so a new provider does not need a matching branch in the
    # dialog that renders it.
    connect_button_label: str | None = None
    terms_company: str | None = None


class AIConnectionOffersResponse(BaseModel):
    offers: list[AIConnectionOffer]


async def get_connection_offers(user_id: str) -> list[AIConnectionOffer]:
    """Describe every connection, naming the models where they are knowable.

    The engine is resolved once for the user rather than per offer: it is a
    property of the deployment and the user, not of which connection they
    pick.
    """
    config = ChatConfig()
    mode = await resolve_engine_mode(user_id, config)
    models = await platform_tier_models(mode, user_id, config)
    codex_models = codex_tier_models(mode)
    # What each linked provider's tiers resolve to, for the rows that name
    # models without offering them.
    tier_models = {"codex": codex_models}
    advanced_allowed = await advanced_tier_allowed(user_id)
    offers = [
        _offer(transport, models, codex_models, advanced_allowed)
        for transport in await get_chat_transports(user_id)
    ]
    return offers + await _locked_offers(user_id, offers, tier_models)


async def _locked_offers(
    user_id: str,
    offers: list[AIConnectionOffer],
    tier_models: dict[str, dict[CopilotLLMModel, str | None]],
) -> list[AIConnectionOffer]:
    """Connections a plan does not include, shown rather than hidden.

    ``get_chat_transports`` answers what may *run*, and it is right to omit a
    transport the entitlement forbids -- anything that routes a turn reads
    that list, and an unroutable entry there would be a bug waiting to
    happen. What it cannot say is *why* a connection is missing, so a user
    below the plan sees no ChatGPT at all and no way to learn one exists.

    These offers say so, in the one place that exists to describe rather than
    route. They are never selectable and carry no credential, so nothing can
    accidentally send a turn down one.
    """
    if not _is_hosted():
        # Self-host grants these entitlements outright; there is nothing
        # to sell.
        return []
    if not await is_feature_enabled(Flag.CHAT_CONNECTION_UPSELL, user_id):
        return []

    connected = {offer.provider_family for offer in offers}
    locked: list[AIConnectionOffer] = []
    for profile in linked_profiles():
        if profile.provider_family in connected:
            continue
        if profile.entitlement is None:
            # Nothing gates it, so its absence means "not connected yet",
            # which the settings page owns rather than an upsell.
            continue
        if await has_entitlement_for_discovery(user_id, profile.entitlement):
            # Entitled but unconnected: the settings page owns that
            # invitation too.
            continue
        locked.append(
            AIConnectionOffer(
                offer_id=f"{profile.key}:locked",
                provider_family=profile.provider_family,
                display_name=profile.display_name,
                auth_method=profile.auth_method,
                credential_id=None,
                backed_by_label=profile.backed_by_label,
                description=profile.description,
                state="locked",
                selectable=False,
                is_default=False,
                # Named even though nothing here can be picked: "what you
                # get" is the whole argument for connecting, and a surface
                # that cannot say which models is asking the user to take it
                # on faith. Not selectable, so naming them cannot be mistaken
                # for offering them.
                tiers=[
                    ConnectionTier(
                        tier=tier,
                        label=label,
                        selectable=False,
                        display_model=tier_models.get(profile.key, {}).get(tier),
                    )
                    for tier, label in TIER_LABELS.items()
                ],
                limitations=[],
                lock_reason=profile.lock_reason,
                unlock_href=profile.unlock_href,
                connect_button_label=profile.connect_button_label,
                terms_company=profile.terms_company,
            )
        )
    return locked


def offer_id_for(transport: ChatTransportResponse) -> str:
    """Stable across requests, and unique per account.

    The platform route carries no credential, and a user may hold several
    ChatGPT accounts, so neither half identifies an offer on its own.
    """
    return f"{transport.auth_provider}:{transport.credential_id or 'deployment'}"


ADVANCED_TIER_LOCK_REASON = "A Max plan or higher is required for Advanced."


class EntitlementUnavailable(Exception):
    """The entitlement service could not answer.

    Raised only on the enforcement path, where "we do not know" has to be
    refused rather than guessed at.
    """


async def advanced_tier_entitled(user_id: str) -> bool:
    """Whether this user actually holds the Advanced entitlement.

    This is the enforcement answer, so it never invents one: a lookup that
    fails raises rather than returning a verdict. The presentation answer
    lives in ``advanced_tier_allowed`` and is deliberately more forgiving --
    the two must not be the same function, because a billing hiccup that
    should merely leave a control enabled must not also authorize spend.
    """
    try:
        return await has_entitlement(user_id, Entitlement.ADVANCED_MODEL_TIER)
    except Exception as exc:
        raise EntitlementUnavailable(str(exc)) from exc


async def advanced_tier_allowed(user_id: str) -> bool:
    """Whether to offer this user the Advanced tier in the picker.

    Presentation only. A failure to resolve the entitlement leaves the tier
    on offer rather than locked: a billing hiccup should not make someone's
    model quietly disappear from the menu. The turn itself is still checked
    by ``advanced_tier_entitled``, which refuses when it cannot tell -- so
    being generous here costs nothing.
    """
    try:
        return await advanced_tier_entitled(user_id)
    except EntitlementUnavailable:
        logger.warning(
            "Could not resolve the Advanced tier entitlement; leaving it on "
            "offer. The turn is still enforced separately.",
            exc_info=True,
        )
        return True


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
        connect_button_label=profile_for(transport.auth_provider).connect_button_label,
        terms_company=profile_for(transport.auth_provider).terms_company,
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
    return profile_for(auth_provider).provider_family


def _auth_method(auth_provider: CopilotLlmAuthProvider) -> str:
    return profile_for(auth_provider).auth_method


def _is_hosted() -> bool:
    return settings.config.behave_as == BehaveAs.CLOUD


def _backed_by_label(transport: ChatTransportResponse) -> str:
    # The platform row is the one thing the table cannot state flatly: what
    # backs it depends on whether this deployment bills anyone.
    if transport.auth_provider == "platform" and not _is_hosted():
        return "This server's chat provider"
    return profile_for(transport.auth_provider).backed_by_label


def _description(transport: ChatTransportResponse) -> str:
    """What backs a run, in the user's terms.

    The credits contrast only means something where credits exist, so the
    self-hosted line says what it actually has instead of denying something
    that was never there.
    """
    if transport.auth_provider == "platform" and not _is_hosted():
        return "New chats are backed by the chat provider configured on this server."
    return profile_for(transport.auth_provider).description


def _limitations(transport: ChatTransportResponse) -> list[str]:
    limitations = list(profile_for(transport.auth_provider).limitations)
    if transport.auth_provider == "platform" and not is_deployment_chat_available():
        limitations.append("No chat provider is configured on this server yet.")
    return limitations
