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

from backend.copilot.config import CopilotLlmAuthProvider, CopilotLLMModel
from backend.copilot.transports import (
    ChatTransportResponse,
    get_chat_transports,
    is_deployment_chat_available,
    settings,
)
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

    ``display_model`` is deliberately absent. Naming the model a tier will
    resolve to needs the execution path, which is decided per turn by which
    service handles it — and on ChatGPT it needs a live call against the
    account. A name that is right half the time is worse here than no name.
    """

    tier: CopilotLLMModel
    label: str
    selectable: bool


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


class AIConnectionOffersResponse(BaseModel):
    offers: list[AIConnectionOffer]


async def get_connection_offers(user_id: str) -> list[AIConnectionOffer]:
    return [_offer(transport) for transport in await get_chat_transports(user_id)]


def offer_id_for(transport: ChatTransportResponse) -> str:
    """Stable across requests, and unique per account.

    The platform route carries no credential, and a user may hold several
    ChatGPT accounts, so neither half identifies an offer on its own.
    """
    return f"{transport.auth_provider}:{transport.credential_id or 'deployment'}"


def _offer(transport: ChatTransportResponse) -> AIConnectionOffer:
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
            ConnectionTier(tier=tier, label=label, selectable=transport.available)
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
