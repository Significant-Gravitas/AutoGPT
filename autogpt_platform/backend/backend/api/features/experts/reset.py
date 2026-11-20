"""Reset a hired expert to how it arrived (SECRT-2692).

Firing stays undoable: a rehire brings the same expert back with everything
it learned. A reset is the other thing people asked for, a factory reset
that cannot be undone. It erases the expert's memory graph and its
computer's volume, puts its soul and routines back to the template, and
archives its chats (the product calls them removed: they never open again).
Approvals still waiting in those chats are declined, so Needs you does not
offer to run something in a chat that is gone.

It leaves alone what is not the expert's own: the account's memory and
business understanding (every expert reads those), files in the user's
workspace, installed workflows, connections, budget, run history and the
name the user gave it. A raised expert has no template, so its soul and
routines stay as they are.
"""

import logging
from datetime import UTC, datetime

import prisma.enums
import prisma.models
from e2b import AsyncVolume
from pydantic import BaseModel

from backend.api.features.experts.models import decode_voice_preferences
from backend.api.features.experts.presentation import template_presentation
from backend.api.features.experts.routines import install_routines
from backend.blocks.desktop._common import expert_volume_name
from backend.copilot.config import ChatConfig
from backend.copilot.constants import COPILOT_SESSION_PREFIX
from backend.copilot.graphiti.erase import erase_scope
from backend.copilot.graphiti.scope import MemoryScope
from backend.copilot.model import invalidate_session_cache
from backend.copilot.scheduled_chats.forget import forget_routine
from backend.copilot.sharing.db import disable_chat_session_share
from backend.copilot.tools.e2b_sandbox import (
    SandboxOwner,
    find_owned_sandbox_id,
    kill_expert_sandbox,
)
from backend.util.clients import get_scheduler_client
from backend.util.exceptions import ExpertNotFoundError, NotFoundError
from backend.util.funnel_analytics import emit_funnel_event
from backend.util.posthog_events import PostHogEvent

logger = logging.getLogger(__name__)


class ExpertResetError(Exception):
    """A step of the reset could not be completed; trying again is safe."""


class ExpertBusyError(Exception):
    """The expert has a turn running or queued; reset once it finishes."""

    def __init__(self, expert_name: str):
        super().__init__(f"{expert_name} is still working")
        self.expert_name = expert_name


class ExpertResetResult(BaseModel):
    """What one reset did."""

    memory_nodes_erased: int
    computer_erased: bool
    soul_reset: bool
    routines_reset: int
    chats_archived: int
    approvals_cleared: int


async def reset_expert(user_id: str, expert_id: str) -> ExpertResetResult:
    """Reset *expert_id*, a live hire of *user_id*.

    The external erasures run first: if FalkorDB or E2B fails, nothing in the
    database has changed and the reset can simply be tried again. The chats
    are archived last for the same reason.
    """
    expert = await prisma.models.Expert.prisma().find_first(
        where={
            "id": expert_id,
            "ownerUserId": user_id,
            "isTemplate": False,
            "isArchived": False,
            "visibility": prisma.enums.ResourceVisibility.PRIVATE,
        }
    )
    if expert is None:
        raise ExpertNotFoundError(expert_id)
    # A turn that ends during the reset hands its conversation to memory
    # ingestion, which would write it back after the erase.
    if await _has_turn_in_flight(user_id, expert_id):
        raise ExpertBusyError(expert.name)

    scope = MemoryScope.for_expert(user_id, expert_id)
    memory_nodes = await erase_scope(scope)
    computer_erased = await _erase_computer(expert_id)
    template = (
        await prisma.models.Expert.prisma().find_unique(
            where={"id": expert.sourceTemplateId}
        )
        if expert.sourceTemplateId
        else None
    )
    routines = 0
    if template is not None:
        await _reset_soul(expert_id, template)
        routines = await _reset_routines(user_id, expert_id, template.id)
    chats = await _archive_chats(user_id, expert_id)
    approvals = await _decline_waiting_approvals(user_id, expert_id)
    # Again, now nothing can reach the archived chats: catches an episode
    # whose ingestion landed while the steps above ran. One queued before the
    # reset can still land later; closing that needs a reset generation the
    # ingestion worker checks (follow-up on SECRT-2692).
    memory_nodes += await erase_scope(scope)
    logger.info(
        f"Reset expert #{expert_id}: {memory_nodes} memory nodes, computer "
        f"{'erased' if computer_erased else 'untouched'}, {routines} routines, "
        f"{chats} chats archived, {approvals} approvals declined"
    )
    result = ExpertResetResult(
        memory_nodes_erased=memory_nodes,
        computer_erased=computer_erased,
        soul_reset=template is not None,
        routines_reset=routines,
        chats_archived=chats,
        approvals_cleared=approvals,
    )
    # A reset is the clearest sign an expert got something wrong. Ids and
    # counts only: never its name or anything from the chats.
    emit_funnel_event(
        user_id,
        PostHogEvent.EXPERT_RESET,
        {
            "expert_id": expert_id,
            "template_id": expert.sourceTemplateId or "created",
            **result.model_dump(),
        },
    )
    return result


async def _has_turn_in_flight(user_id: str, expert_id: str) -> bool:
    busy = await prisma.models.ChatSession.prisma().count(
        where={
            "userId": user_id,
            "expertId": expert_id,
            "archivedAt": None,
            "chatStatus": {
                "in": [
                    prisma.enums.ChatSessionStatus.running,
                    prisma.enums.ChatSessionStatus.queued,
                ]
            },
        }
    )
    return busy > 0


async def _erase_computer(expert_id: str) -> bool:
    """Kill the expert's box and delete its volume, its notes with it.

    Returns False when no computer is configured (no E2B key) or it never
    had a volume. E2B has no delete-by-name, so the volume is found by its
    name in the account's list. That list is whole: ``GET /volumes`` takes no
    limit or page token and answers one array (e2b 2.49.1), unlike the
    sandbox list, which pages.
    """
    api_key = ChatConfig().active_e2b_api_key
    if not api_key:
        return False
    if not await kill_expert_sandbox(expert_id, api_key):
        # False means "no box" or "the kill failed". A box that survives would
        # come back with its old screen and processes, so find out which; the
        # lookup itself raises when E2B cannot say.
        owner = SandboxOwner(kind="expert", id=expert_id)
        if await find_owned_sandbox_id(owner, api_key) is not None:
            raise ExpertResetError(f"Could not stop expert #{expert_id}'s computer")
    name = expert_volume_name(expert_id)
    volumes = await AsyncVolume.list(api_key=api_key)
    erased = False
    for volume in volumes:
        if volume.name == name:
            erased = await AsyncVolume.destroy(volume.volume_id, api_key=api_key)
    return erased


async def _reset_soul(expert_id: str, template: prisma.models.Expert) -> None:
    """Put identity, voice and boundaries back to the template's, as a hire
    copies them. The name stays: the user gave it."""
    voice, _ = decode_voice_preferences(template.voicePreferences)
    await prisma.models.Expert.prisma().update(
        where={"id": expert_id},
        data={
            "identity": template_presentation(template)["identity"],
            "voicePreferences": voice,
            "boundaries": template.boundaries,
        },
    )


async def _reset_routines(user_id: str, expert_id: str, template_id: str) -> int:
    """Switch every routine off and replace them with the template's offers.

    A routine the user switched on has scheduler jobs; those go first so
    nothing keeps firing into a row that no longer exists.
    """
    current = await prisma.models.ExpertRoutine.prisma().find_many(
        where={"expertId": expert_id}
    )
    scheduler = get_scheduler_client()
    for row in current:
        # Unlike switching a routine off, the row goes away, so a job left
        # behind would fire with nothing to stop it: a failed delete stops
        # the reset instead (it is safe to try again).
        for schedule_id in row.scheduleIds:
            try:
                await scheduler.delete_schedule(schedule_id, user_id=user_id)
            except NotFoundError:
                pass  # already gone, which is what the reset wants
        try:
            await forget_routine(user_id, row.id)
        except Exception:
            logger.warning(
                f"Could not clear the failure streak of routine #{row.id}",
                exc_info=True,
            )
    await prisma.models.ExpertRoutine.prisma().delete_many(
        where={"expertId": expert_id}
    )
    template_routines = await prisma.models.ExpertRoutine.prisma().find_many(
        where={"expertId": template_id},
        order=[{"createdAt": "asc"}, {"id": "asc"}],
    )
    failed = await install_routines(expert_id, template_routines)
    return len(template_routines) - len(failed)


async def _archive_chats(user_id: str, expert_id: str) -> int:
    """Archive every chat with the expert, then stop sharing every archived
    one that is still shared, and drop each from the session cache so no
    cached copy still reads as open.

    The unsharing reads after the archive stamp, so a chat shared while this
    ran, or left shared by an earlier reset that was interrupted, is caught
    in the same call: a chat is never left archived but still shared.
    """
    sessions = prisma.models.ChatSession.prisma()
    archived = await sessions.update_many(
        where={"userId": user_id, "expertId": expert_id, "archivedAt": None},
        data={"archivedAt": datetime.now(UTC)},
    )
    still_shared = await sessions.find_many(
        where={
            "userId": user_id,
            "expertId": expert_id,
            "archivedAt": {"not": None},
            "isShared": True,
        }
    )
    for chat in still_shared:
        # The cascade the owner's own "stop sharing" runs: it also revokes
        # runs shared through this chat, and keeps runs shared on their own.
        await disable_chat_session_share(chat.id, user_id)
    for chat in await sessions.find_many(
        where={"userId": user_id, "expertId": expert_id}
    ):
        await invalidate_session_cache(chat.id)
    return archived


async def _decline_waiting_approvals(user_id: str, expert_id: str) -> int:
    """Decline every approval still waiting in the expert's archived chats.

    Approving one would wake a turn in a chat that no longer opens, running
    the held call against the expert the reset just cleared. Declined, the
    card leaves Needs you. A workflow step's own review is kept: the run
    belongs to run history, which the reset leaves alone.
    """
    archived = await prisma.models.ChatSession.prisma().find_many(
        where={"userId": user_id, "expertId": expert_id, "archivedAt": {"not": None}}
    )
    if not archived:
        return 0
    ids = [chat.id for chat in archived]
    return await prisma.models.PendingHumanReview.prisma().update_many(
        where={
            "userId": user_id,
            "status": prisma.enums.ReviewStatus.WAITING,
            "OR": [
                {"chatSessionId": {"in": ids}},
                # Rows an older deploy wrote in the synthetic-graph shape.
                {"graphExecId": {"in": [f"{COPILOT_SESSION_PREFIX}{i}" for i in ids]}},
            ],
        },
        data={
            "status": prisma.enums.ReviewStatus.REJECTED,
            "reviewMessage": "The expert was reset",
            "processed": True,
            "reviewedAt": datetime.now(UTC),
        },
    )
