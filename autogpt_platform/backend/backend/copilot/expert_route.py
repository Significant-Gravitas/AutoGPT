"""The AI connection an expert's new threads run on.

An expert can be pinned to one chat LLM route — the platform, a linked ChatGPT
account, or Microsoft 365 Copilot — and every thread, kickoff, routine,
follow-up and delegation addressed to that expert then starts there, unless
the user explicitly picks a route for one chat. An unpinned expert follows the
owner's account default exactly as a plain chat does.

The pin is resolved, never trusted: a credential that has been unlinked, a plan
that no longer includes ChatGPT, or a Microsoft pin on a turn that needs tools
all fall through to the account default, with the pin left in place so
reconnecting restores it. Nothing here raises for an unattended caller.
"""

import logging

from backend.copilot.config import CopilotLlmAuthProvider
from backend.copilot.transports import (
    resolve_default_chat_route,
    resolve_pinned_chat_route,
)
from backend.data.db_accessors import experts_db

logger = logging.getLogger(__name__)


async def resolve_expert_chat_route(
    user_id: str, expert_id: str
) -> tuple[CopilotLlmAuthProvider, str | None]:
    """The route for an unattended turn addressed to an expert.

    The expert's own pin when it still resolves and can run tools, else
    whatever :func:`resolve_default_chat_route` would give a plain chat.
    Never raises and never asks, for the same callers and the same reasons.
    """
    auth_provider, credential_id = await expert_pinned_chat_route(user_id, expert_id)
    pinned = await resolve_pinned_chat_route(
        user_id, auth_provider, credential_id, unattended=True
    )
    if pinned is not None:
        return pinned.auth_provider, pinned.credential_id
    return await resolve_default_chat_route(user_id)


async def expert_pinned_chat_route(
    user_id: str, expert_id: str
) -> tuple[str | None, str | None]:
    """The (provider, credential) an expert is pinned to, raw and unvalidated.

    ``(None, None)`` for an expert on the account default — and for one that
    cannot be read: an archived or missing expert has no pin worth honouring,
    and a lookup failure must degrade to the default rather than fail a turn.
    """
    try:
        expert = await experts_db().get_expert(
            user_id, expert_id, include_workflows=False
        )
    except Exception:
        logger.warning(
            "Could not read the chat route pinned on expert %s; using the "
            "account default",
            expert_id[:12],
            exc_info=True,
        )
        return None, None
    if expert is None:
        return None, None
    return expert.llm_auth_provider, expert.llm_credential_id
