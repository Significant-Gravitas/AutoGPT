"""Expert and model-tier choices for bot chat turns.

Expert sessions stay DM-only: a server turn runs as the server-link owner but
is typed by anyone in the server, and expert sessions carry the owner's
private expert grants and memory, so a shared server must never reach one
(see ``fail closed on non-private experts``).
"""

import logging

from backend.copilot.config import CopilotLLMModel
from backend.copilot.model import ChatSession
from backend.copilot.offers import EntitlementUnavailable, advanced_tier_entitled
from backend.data.db_accessors import experts_db
from backend.util.exceptions import NotFoundError

from .models import BotExpert

logger = logging.getLogger(__name__)

SERVER_EXPERT_REFUSAL = "Experts are only available in DMs for now."
ADVANCED_FALLBACK_NOTICE = (
    "_Smart mode isn't on your plan, so this reply uses fast mode._"
)


async def list_owner_experts(owner_user_id: str) -> list[BotExpert]:
    """The owner's live hires, as the bot shows and matches them."""
    identities = await experts_db().list_expert_identities(owner_user_id)
    return [
        BotExpert(id=e.id, name=e.name, role=e.role)
        for e in identities
        if not e.is_archived
    ]


async def require_turn_expert(
    owner_user_id: str, expert_id: str | None, platform_server_id: str | None
) -> None:
    """Refuse an expert the turn may not use, before any session is touched."""
    if expert_id is None:
        return
    if platform_server_id is not None:
        raise NotFoundError(SERVER_EXPERT_REFUSAL)
    if not await experts_db().owns_active_expert(owner_user_id, expert_id):
        raise NotFoundError("Expert not found.")


async def session_reusable(
    session: ChatSession,
    owner_user_id: str,
    expert_id: str | None,
    allow_expert_session: bool,
) -> bool:
    """Whether a cached session can carry this turn.

    No expert asked for keeps whatever the conversation already is, as long
    as that expert is still a live hire. Asking for an expert the session
    isn't bound to starts a new session for it.
    """
    if session.expert_id is None:
        return expert_id is None
    if not allow_expert_session:
        return False
    if expert_id is not None and expert_id != session.expert_id:
        return False
    return await experts_db().owns_active_expert(owner_user_id, session.expert_id)


async def resolve_turn_model(
    owner_user_id: str, model: CopilotLLMModel | None, session: ChatSession
) -> tuple[CopilotLLMModel | None, str | None]:
    """Apply the web route's Advanced entitlement check.

    Returns the tier to run and a one-line notice when Advanced was asked for
    but can't be used (not entitled, or the entitlement lookup failed).
    """
    if model != "advanced":
        return model, None
    if session.metadata.llm_auth_provider == "microsoft_365_copilot":
        return model, None
    try:
        entitled = await advanced_tier_entitled(owner_user_id)
    except EntitlementUnavailable:
        logger.warning(
            "Advanced entitlement lookup failed for ...%s", owner_user_id[-8:]
        )
        entitled = False
    if entitled:
        return model, None
    return "standard", ADVANCED_FALLBACK_NOTICE
