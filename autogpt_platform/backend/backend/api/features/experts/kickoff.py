"""Start a hire's day-one kickoff on the server.

The page a browser hire lands on sends the kickoff itself
(``/copilot?kickoff=1``, or the team chat drawer's first open). A hire made
from Otto's chat or by an API caller is followed by no such page, so its
expert never introduced itself until someone opened it. This sends the same
kickoff turn for those hires: same session, same server-derived message id,
so a browser that opens the expert afterwards adopts it instead of sending a
second one.
"""

import logging

from backend.api.features.experts.models import HireSurface

from backend.copilot import turn_queue
from backend.copilot.active_turns import (
    ConcurrentTurnLimitError,
    SessionNotIdle,
    get_inflight_turn_limit,
)
from backend.copilot.db import chat_message_has_assistant_reply
from backend.copilot.executor.utils import schedule_chat_turn
from backend.copilot.expert_kickoff import (
    EXPERT_KICKOFF_PROMPT,
    expert_kickoff_message_id,
    expert_kickoff_metadata,
)
from backend.copilot.model import (
    get_or_create_expert_kickoff_session,
    get_user_sessions,
)
from backend.copilot.session_permissions import resolve_session_permissions

logger = logging.getLogger(__name__)

# Hire surfaces no browser follows into the new expert's first thread: Otto's
# chat ("copilot") and API callers that name no surface. The expert page and
# onboarding land on a page that sends the kickoff itself, and the trial's
# team hire ("onboarding") kicks each expert off when it is first opened.
SERVER_KICKOFF_SURFACES: tuple[HireSurface | None, ...] = (None, "copilot")


async def start_expert_kickoff(
    user_id: str,
    expert_id: str,
    *,
    organization_id: str | None = None,
    team_id: str | None = None,
) -> bool:
    """Send *expert_id*'s kickoff turn unless it already has a thread.

    *organization_id* must be the org the owner's browser scopes its chats
    to, or a later browser open would not find this thread and kick off
    again. Best-effort: returns whether a turn was started or queued, and
    never raises, because a hire must not fail on its greeting.
    """
    try:
        return await _start(user_id, expert_id, organization_id, team_id)
    except Exception:
        logger.exception(f"Kickoff for expert #{expert_id} could not be sent")
        return False


async def _start(
    user_id: str,
    expert_id: str,
    organization_id: str | None,
    team_id: str | None,
) -> bool:
    # A rehire, or a hire someone already talked to, has its thread: the
    # kickoff is for an expert nobody has heard from yet.
    existing, _ = await get_user_sessions(
        user_id,
        limit=1,
        organization_id=organization_id,
        expert_id=expert_id,
        pinned_first=False,
    )
    if existing:
        return False
    session = await get_or_create_expert_kickoff_session(
        user_id,
        expert_id,
        dry_run=False,
        organization_id=organization_id,
        team_id=team_id,
    )
    session_id = session.session_id
    message_id = expert_kickoff_message_id(user_id, session_id, expert_id)
    if await chat_message_has_assistant_reply(message_id, session_id) is not None:
        # A browser sent it between the check above and here.
        return False
    message_metadata = expert_kickoff_metadata(expert_id)
    permissions = resolve_session_permissions(session)
    try:
        turn_id = await schedule_chat_turn(
            session_id=session_id,
            user_id=user_id,
            message=EXPERT_KICKOFF_PROMPT,
            message_id=message_id,
            dedupe_content=False,
            message_metadata=message_metadata,
            expert_id=expert_id,
            session_origin=session.metadata.origin,
            session_source_platform=session.metadata.source_platform,
            organization_id=session.organization_id,
            team_id=session.team_id,
            llm_auth_provider=session.metadata.llm_auth_provider,
            llm_credential_id=session.metadata.llm_credential_id,
            permissions=permissions,
            require_fresh_slot=True,
        )
    except SessionNotIdle:
        # Another kickoff (a browser's) holds the session.
        return False
    except ConcurrentTurnLimitError:
        # The owner is at the running cap: queue it the way the chat route
        # queues a message, so the expert still introduces itself once a
        # slot frees instead of leaving an empty thread behind.
        await turn_queue.try_enqueue_turn(
            user_id=user_id,
            inflight_cap=get_inflight_turn_limit(),
            session_id=session_id,
            message=EXPERT_KICKOFF_PROMPT,
            message_id=message_id,
            message_metadata=message_metadata,
            llm_auth_provider=session.metadata.llm_auth_provider,
            llm_credential_id=session.metadata.llm_credential_id,
            permissions=(
                permissions.model_dump(exclude_none=True) if permissions else None
            ),
        )
        return True
    return turn_id is not None
