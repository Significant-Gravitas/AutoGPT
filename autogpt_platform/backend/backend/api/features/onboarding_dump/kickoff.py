"""Otto starts the user's first suggested tasks during onboarding.

Right after the brain dump, the first one or two open rows of the Home
checklist are handed to Otto with nobody in the browser, the way the
scheduler fires a copilot turn. Each gets its own chat, tagged
``ONBOARDING_DRAFT_KIND`` so the greeting still meets the user, and the row
is marked ``kickoff`` so the copilot home can show it as Otto's work.

Safe to call any number of times: a take kicks off once, and each row's
chat id is derived from the take and the row, so two racing calls cannot
both create it.
"""

import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Literal

from prisma.enums import BrainDumpInputMode, BrainDumpStatus
from prisma.models import OnboardingBrainDump
from pydantic import BaseModel

from backend.api.features.onboarding_dump import db, suggested_tasks
from backend.api.features.onboarding_dump.models import KickoffResponse, SuggestedTask
from backend.api.features.orgs.db import resolve_default_tenancy
from backend.copilot.active_turns import ConcurrentTurnLimitError
from backend.copilot.config import CopilotLlmAuthProvider
from backend.copilot.executor.utils import schedule_turn
from backend.copilot.model import (
    ONBOARDING_DRAFT_KIND,
    create_chat_session,
    delete_chat_session,
    get_chat_session_metadata,
)
from backend.copilot.tracking import ONBOARDING_DRAFT_KICKOFF_KIND
from backend.copilot.transports import resolve_default_chat_route
from backend.copilot.turn_queue import turn_refusal
from backend.util import product_analytics
from backend.util.exceptions import DatabaseError

logger = logging.getLogger(__name__)

MAX_KICKOFF_TASKS = 2

# Drafts are for someone who just finished onboarding. A take older than
# this (someone who onboarded before drafts shipped, say) gets none.
KICKOFF_MAX_DUMP_AGE = timedelta(hours=24)

# What came of starting one row: its chat runs, the user moved the row on
# first, or the user is at the turn cap and nothing more can start.
_DraftOutcome = Literal["started", "skipped", "capped"]


class _DraftTarget(BaseModel):
    """Who a draft runs for, and on which connection and tenant."""

    user_id: str
    recording_id: str
    llm_auth_provider: CopilotLlmAuthProvider
    llm_credential_id: str | None
    organization_id: str | None
    team_id: str | None


async def kickoff_suggested_tasks(user_id: str) -> KickoffResponse:
    """Start up to ``MAX_KICKOFF_TASKS`` open suggested tasks for the user.

    ``pending`` means the greeting has not stored its prompts yet, so the
    caller should ask again; ``nothing`` covers a user with no take, a
    skipped or failed one, one finished more than ``KICKOFF_MAX_DUMP_AGE``
    ago, and any failure here, which is logged rather than raised because
    nobody is waiting on this.
    """
    try:
        return await _kick_off(user_id)
    except Exception:
        logger.exception(f"Onboarding kickoff failed for user {user_id}")
        return KickoffResponse(status="nothing")


async def _kick_off(user_id: str) -> KickoffResponse:
    dump = await db.get_dump(user_id)
    if dump is None or _no_tasks_coming(dump):
        return KickoffResponse(status="nothing")
    tasks = suggested_tasks.stored_tasks(dump.suggestedPrompts)
    if not tasks:
        status = "nothing" if dump.status == BrainDumpStatus.completed else "pending"
        return KickoffResponse(status=status)
    kicked_off = [task.id for task in tasks if task.kickoff]
    if kicked_off:
        return KickoffResponse(status="started", started_task_ids=kicked_off)
    open_tasks = [task for task in tasks if task.state == "open"][:MAX_KICKOFF_TASKS]
    if not open_tasks or _too_old(dump):
        return KickoffResponse(status="nothing")
    target = await _draft_target(user_id, dump.recordingId)
    if target is None:
        return KickoffResponse(status="refused")
    started: list[str] = []
    for task in open_tasks:
        outcome = await _start_draft(target, task)
        if outcome == "capped":
            break
        if outcome == "started":
            started.append(task.id)
    return KickoffResponse(
        status="started" if started else "nothing", started_task_ids=started
    )


def _no_tasks_coming(dump: OnboardingBrainDump) -> bool:
    return (
        dump.inputMode == BrainDumpInputMode.skipped
        or dump.status == BrainDumpStatus.failed
    )


def _too_old(dump: OnboardingBrainDump) -> bool:
    # The row has no completion time of its own; its last write is the
    # nearest. Later writes (the team, a row sent or hidden) only move it
    # on, so the cutoff errs towards starting.
    return datetime.now(UTC) - dump.updatedAt > KICKOFF_MAX_DUMP_AGE


async def _draft_target(user_id: str, recording_id: str) -> _DraftTarget | None:
    """Where the user's drafts run; None when they may not start a turn."""
    llm_auth_provider, llm_credential_id = await resolve_default_chat_route(user_id)
    if await turn_refusal(user_id, llm_auth_provider) is not None:
        return None
    organization_id, team_id = await resolve_default_tenancy(user_id)
    return _DraftTarget(
        user_id=user_id,
        recording_id=recording_id,
        llm_auth_provider=llm_auth_provider,
        llm_credential_id=llm_credential_id,
        organization_id=organization_id,
        team_id=team_id,
    )


async def _start_draft(target: _DraftTarget, task: SuggestedTask) -> _DraftOutcome:
    """Open the task's chat, start Otto's turn in it and mark the row."""
    session_id = draft_session_id(target.user_id, target.recording_id, task.id)
    if not await _open_draft_chat(target, session_id):
        # Its chat exists: another call started it, or an earlier one did
        # and never got to mark the row. Its turn is not started twice.
        marked = await suggested_tasks.mark_kicked_off(
            target.user_id, task.id, session_id
        )
        return "started" if marked else "skipped"
    try:
        await _schedule_draft_turn(target, session_id, task.prompt)
    except ConcurrentTurnLimitError:
        logger.info(f"Onboarding kickoff hit the turn cap for user {target.user_id}")
        # An empty chat would sit in the sidebar, and its id must be free
        # for the next try.
        await delete_chat_session(session_id, target.user_id)
        return "capped"
    except Exception:
        # Left behind, a later call would find the chat and mark a draft no
        # turn ever ran in.
        await delete_chat_session(session_id, target.user_id)
        raise
    if not await suggested_tasks.mark_kicked_off(target.user_id, task.id, session_id):
        # The user sent or hid the row while its chat was being opened.
        await delete_chat_session(session_id, target.user_id)
        return "skipped"
    product_analytics.track_onboarding_draft_started(
        user_id=target.user_id, task_id=task.id, session_id=session_id
    )
    return "started"


async def _schedule_draft_turn(
    target: _DraftTarget, session_id: str, prompt: str
) -> None:
    await schedule_turn(
        session_id=session_id,
        user_id=target.user_id,
        turn_id=str(uuid.uuid4()),
        message=prompt,
        # The user picked the task but did not send it: not their turn.
        message_metadata={"kind": ONBOARDING_DRAFT_KICKOFF_KIND},
        # Nobody is in the browser to answer this turn's questions.
        unattended=True,
        organization_id=target.organization_id,
        team_id=target.team_id,
        llm_auth_provider=target.llm_auth_provider,
        llm_credential_id=target.llm_credential_id,
    )


async def _open_draft_chat(target: _DraftTarget, session_id: str) -> bool:
    """Create the draft's chat; False when this user already has it."""
    if await get_chat_session_metadata(session_id, target.user_id) is not None:
        return False
    try:
        await create_chat_session(
            target.user_id,
            dry_run=False,
            session_id=session_id,
            organization_id=target.organization_id,
            team_id=target.team_id,
            # The prompt is the user's own pick from their checklist, so the
            # chat is theirs: workflows it starts still pause before anything
            # that cannot be undone.
            origin="interactive",
            llm_auth_provider=target.llm_auth_provider,
            llm_credential_id=target.llm_credential_id,
            kind=ONBOARDING_DRAFT_KIND,
        )
    except DatabaseError:
        # The id is unique, so a racing call may have created it in between.
        if await get_chat_session_metadata(session_id, target.user_id) is None:
            raise
        return False
    return True


def draft_session_id(user_id: str, recording_id: str, task_id: str) -> str:
    """The one chat a take's suggested task can be kicked off into."""
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL, f"onboarding-draft:{user_id}:{recording_id}:{task_id}"
        )
    )
