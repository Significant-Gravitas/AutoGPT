"""Start the task a new hire was approved for, in the same confirm.

A hire Otto proposes for a task the user already described carries that task
(``hire_expert``'s ``handoff_title`` / ``handoff_task``), and its card offers
"hire and hand off" beside "hire only". Without this the approval only hired:
the model then asked in prose whether to hand the task over, and the user had
to type a second answer to something they had just approved (SECRT-3138).

Confirming the hand-off id hires and then runs the task as a normal
``delegate_to_expert`` from this chat (``open_delegation_thread`` and
``run_delegated_turn``), so the thread, its permissions, mode and result
contract are exactly a delegation's, and a paused teammate is refused the
same way. The user's click on the card is the approval for both; the hand-off
is not put to them again.

The hire stands whatever happens to the task, and what is reported is only
what is known: started, queued for a task slot, not started (with whether a
retry can help), or sent but unconfirmed. When the send raises, the thread
says which: a dispatch that failed released its turn, leaving the thread
idle, so it is removed and a retry is safe; anything else may be running and
is never sent a second time.
"""

import logging
from typing import Literal

from backend.api.features.experts.models import Expert
from backend.copilot.model import CHAT_STATUS_IDLE, ChatSession, delete_chat_session
from backend.data.db_accessors import chat_db

from .delegate_to_expert import (
    open_delegation_thread,
    paused_refusal,
    run_delegated_turn,
)
from .expert_delegation import chain_refusal, resolve_target_expert
from .expert_proposal import hire_message
from .models import (
    ExpertChangeAppliedResponse,
    ExpertHandoffOutcome,
    ExpertHandoffTask,
    SubSessionStatusResponse,
)
from .run_sub_session import _sub_session_link

logger = logging.getLogger(__name__)

_STARTED = frozenset({"running", "completed"})


class _NotStarted(Exception):
    """The task did not reach the new hire; carries the model-facing reason
    and whether trying the delegation again can help."""

    def __init__(self, reason: str, *, retry: bool) -> None:
        super().__init__(reason)
        self.reason = reason
        self.retry = retry


async def hand_off_to_new_hire(
    user_id: str,
    session: ChatSession,
    applied: ExpertChangeAppliedResponse,
    task: ExpertHandoffTask,
) -> ExpertChangeAppliedResponse:
    """Start *task* with the expert *applied* just hired."""
    expert = applied.expert
    try:
        target = await _target(user_id, session, expert.id)
        thread = await open_delegation_thread(user_id, session, target)
    except _NotStarted as e:
        return _report(applied, task, "failed", None, e.reason, e.retry)
    except Exception:
        logger.warning("Hand-off to a new hire could not open", exc_info=True)
        return _report(applied, task, "failed", None, "It could not be sent.", True)
    try:
        sent = await run_delegated_turn(
            user_id=user_id,
            session=session,
            target=target,
            inner_session_id=thread,
            prompt=task.brief,
            opened_here=True,
            wait_for_result=0,
        )
    except Exception:
        logger.warning("Hand-off to a new hire raised", exc_info=True)
        if await _was_sent(thread):
            return _report(applied, task, "unconfirmed", thread)
        await _discard(thread, user_id)
        return _report(applied, task, "failed", None, "It could not be sent.", True)
    if not isinstance(sent, SubSessionStatusResponse) or sent.status == "error":
        return _report(applied, task, "failed", None, sent.message, True)
    status = "started" if sent.status in _STARTED else "queued"
    return _report(applied, task, status, sent.sub_session_id)


def with_hire_only(
    applied: ExpertChangeAppliedResponse, task: ExpertHandoffTask
) -> ExpertChangeAppliedResponse:
    """The user picked "hire only" on a card that offered the hand-off too."""
    return applied.model_copy(
        update={
            "message": (
                f"{hire_message(applied.expert.name)} The user chose hire only, "
                f"without handing over {task.title}: don't hand it over, and "
                "don't ask about it unless they bring it up."
            )
        }
    )


async def _target(user_id: str, session: ChatSession, expert_id: str) -> Expert:
    """The hire, if it can take delegated work; ``delegate_to_expert``'s rules."""
    target = await resolve_target_expert(user_id, expert_id)
    if target is None or target.is_archived:
        raise _NotStarted("They are not on the team right now.", retry=False)
    if (paused := paused_refusal(target)) is not None:
        raise _NotStarted(paused, retry=False)
    if (refusal := await chain_refusal(user_id, session, target)) is not None:
        raise _NotStarted(refusal, retry=False)
    return target


async def _was_sent(thread: str) -> bool:
    """Whether a send that raised may have reached *thread*.

    Only an idle thread proves it did not: a failed dispatch releases its
    turn slot, which sets the thread back to idle. An unreadable status
    counts as sent, so the task is never handed over twice.
    """
    try:
        return await chat_db().get_chat_session_status(thread) != CHAT_STATUS_IDLE
    except Exception:
        logger.warning("Could not read a hand-off thread's status", exc_info=True)
        return True


async def _discard(thread: str, user_id: str) -> None:
    """Remove a thread nothing was sent to. Best-effort, like
    ``discard_unused_sub_session``: an empty thread is the lesser harm."""
    try:
        await delete_chat_session(thread, user_id)
    except Exception:
        logger.warning("Failed to remove an unused hand-off thread", exc_info=True)


def _report(
    applied: ExpertChangeAppliedResponse,
    task: ExpertHandoffTask,
    status: Literal["started", "queued", "unconfirmed", "failed"],
    thread: str | None,
    reason: str = "",
    retry: bool = False,
) -> ExpertChangeAppliedResponse:
    expert = applied.expert
    link = _sub_session_link(thread)
    if status == "failed":
        message = _failed_message(expert.name, expert.id, task, reason, retry)
    else:
        message = _sent_message(status, expert.name, task, thread or "", link)
    return applied.model_copy(
        update={
            "message": message,
            "handoff": ExpertHandoffOutcome(
                title=task.title, status=status, sub_session_id=thread, link=link
            ),
        }
    )


def _sent_message(
    status: Literal["started", "queued", "unconfirmed"],
    name: str,
    task: ExpertHandoffTask,
    thread: str,
    link: str | None,
) -> str:
    where = f"their own chat at {link}"
    if status == "unconfirmed":
        return (
            f"{name} is hired. {task.title} was sent to {where}, but whether it "
            "started could not be confirmed. Check it with "
            f"tool:get_sub_session_result (sub_session_id: {thread}) before "
            "anything else, and don't hand it over again: it may already be "
            "running."
        )
    state = (
        f"has started on {task.title} in {where}"
        if status == "started"
        else f"will start on {task.title} in {where} when one of the user's "
        "task slots frees"
    )
    return (
        f"{name} is hired and {state}. Tell the user in one short line that "
        f"{name} is on it and where to follow along. Don't ask whether to hand "
        "it over: they chose that on the card. Their result comes back here; "
        f"check it with tool:get_sub_session_result (sub_session_id: {thread}) "
        "when you need it. Their workflows and skills finish installing in the "
        "background."
    )


def _failed_message(
    name: str, expert_id: str, task: ExpertHandoffTask, reason: str, retry: bool
) -> str:
    lead = f"{name} is hired and on the team, but {task.title} did not reach them: "
    if not retry:
        return (
            f"{lead}{reason} Tell the user why in one short line, and don't ask "
            "them again whether to hand it over."
        )
    return (
        f"{lead}{reason} The user already approved handing it over on the card, "
        f"so start it with delegate_to_expert (expert_id: {expert_id}) and the "
        "same task instead of asking them again."
    )
