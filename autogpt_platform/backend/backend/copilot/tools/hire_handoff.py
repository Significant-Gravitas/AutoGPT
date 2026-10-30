"""Start the task a new hire was approved for, in the same confirm.

A hire Otto proposes for a task the user already described carries that task
(``hire_expert``'s ``handoff_title`` / ``handoff_task``), and its card asks
Hire and start or Not now. Without this the approval only hired:
the model then asked in prose whether to hand the task over, and the user had
to type a second answer to something they had just approved (SECRT-3138).

Confirming the hand-off id hires and then runs the task as a normal
``delegate_to_expert`` from this chat (``open_delegation_thread`` and
``run_delegated_turn``), so the thread, its permissions, mode and result
contract are exactly a delegation's, and a paused teammate is refused the
same way. The user's click on the card is the approval for both; the hand-off
is not put to them again.

The hire stands whatever happens to the task, and what is reported is only
what is known: started, queued for a task slot, refused (no retry: the
refusal says why), or, when the send raised, whether it provably never left.
``TurnNotDispatched`` means nothing reached the queue: ``run_delegated_turn``
has removed the empty thread and a retry is safe, through one path only: the
card's Try again (``handoff.retry``), never Otto re-sending it as well. Any other error, a publish
that raised included, may have delivered the task, so it is reported
unconfirmed with its thread and never sent a second time.
"""

import logging
from typing import Literal

from backend.api.features.experts.models import Expert
from backend.copilot.model import ChatSession
from backend.copilot.sdk.session_waiter import TurnNotDispatched

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

# With nothing waited for, "running" only means the turn is on the queue: a
# worker can still refuse it at pickup, so the hand-off claims "sent", never
# "started".
_SENT = frozenset({"running", "completed"})


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
    except TurnNotDispatched:
        logger.warning("Hand-off to a new hire was not sent", exc_info=True)
        return _report(applied, task, "failed", None, "It could not be sent.", True)
    except Exception:
        logger.warning("Hand-off to a new hire is unconfirmed", exc_info=True)
        return _report(applied, task, "unconfirmed", thread)
    if not isinstance(sent, SubSessionStatusResponse) or sent.status == "error":
        # A refusal (tree budget or depth, a full set of task slots) says
        # itself what has to change first.
        return _report(applied, task, "failed", None, sent.message, False)
    status = "sent" if sent.status in _SENT else "queued"
    return _report(applied, task, status, sent.sub_session_id)


def with_hire_only(
    applied: ExpertChangeAppliedResponse, task: ExpertHandoffTask
) -> ExpertChangeAppliedResponse:
    """The plain hire id was confirmed for a hire proposed with a task: the
    user asked in chat to hire without starting it (the card itself offers
    only Hire and start or Not now)."""
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


def _report(
    applied: ExpertChangeAppliedResponse,
    task: ExpertHandoffTask,
    status: Literal["sent", "queued", "unconfirmed", "failed"],
    thread: str | None,
    reason: str = "",
    retry: bool = False,
) -> ExpertChangeAppliedResponse:
    expert = applied.expert
    link = _sub_session_link(thread)
    if status == "failed":
        message = _failed_message(expert.name, task, reason, retry)
    else:
        message = _sent_message(status, expert.name, task, thread or "", link)
    return applied.model_copy(
        update={
            "message": message,
            "handoff": ExpertHandoffOutcome(
                title=task.title,
                status=status,
                sub_session_id=thread,
                link=link,
                retry=status == "failed" and retry,
            ),
        }
    )


def _sent_message(
    status: Literal["sent", "queued", "unconfirmed"],
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
        f"{task.title} was sent to {where}"
        if status == "sent"
        else f"{task.title} is waiting in {where} for one of the user's task "
        "slots to free"
    )
    return (
        f"{name} is hired and {state}. Tell the user in one short line that "
        f"it is with {name} and where to follow along. Don't ask whether to hand "
        "it over: they chose that on the card. Nothing sends their result back "
        "to this chat: read it with tool:get_sub_session_result "
        f"(sub_session_id: {thread}) when you need it. Their workflows and "
        "skills finish installing in the background."
    )


def _failed_message(
    name: str, task: ExpertHandoffTask, reason: str, retry: bool
) -> str:
    lead = f"{name} is hired and on the team, but {task.title} did not reach them: "
    if not retry:
        return (
            f"{lead}{reason} Tell the user why in one short line, and don't ask "
            "them again whether to hand it over."
        )
    # One retry path: the card's Try again. Otto re-sending it too could
    # start the task twice.
    return (
        f"{lead}{reason} Nothing reached them, so it can be sent again: tell "
        "the user in one short line that it didn't start and that Try again on "
        "the card sends it. Don't hand it over yourself, so it can't start twice."
    )
