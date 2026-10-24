"""Start the task a new hire was approved for, in the same confirm.

A hire Otto proposes for a task the user already described carries that task
(``hire_expert``'s ``handoff_title`` / ``handoff_task``), and its card offers
"hire and hand off" beside "hire only". Without this the approval only hired:
the model then asked in prose whether to hand the task over, and the user had
to type a second answer to something they had just approved (SECRT-3138).

Confirming the hand-off id hires and then runs the task as a normal
``delegate_to_expert`` from this chat (``start_delegation``), so the thread,
its permissions, mode and result contract are exactly a delegation's. The
user's click on the card is the approval for both; the hand-off is not put to
them again. The hire stands even when the task cannot start, and the model is
told to retry the delegation rather than ask.
"""

import logging

from backend.copilot.model import ChatSession

from .delegate_to_expert import start_delegation
from .expert_delegation import chain_refusal, resolve_target_expert
from .expert_proposal import hire_message
from .models import (
    ExpertChangeAppliedResponse,
    ExpertHandoffOutcome,
    ExpertHandoffTask,
    SubSessionStatusResponse,
)

logger = logging.getLogger(__name__)

_STARTED = frozenset({"running", "queued", "completed"})


class _NotStarted(Exception):
    """The task did not reach the new hire; carries the model-facing reason."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


async def hand_off_to_new_hire(
    user_id: str,
    session: ChatSession,
    applied: ExpertChangeAppliedResponse,
    task: ExpertHandoffTask,
) -> ExpertChangeAppliedResponse:
    """Start *task* with the expert *applied* just hired."""
    expert = applied.expert
    try:
        started = await _start(user_id, session, expert.id, task)
    except _NotStarted as e:
        return _failed(applied, task, e.reason)
    except Exception:
        logger.warning("Hand-off to a new hire failed to start", exc_info=True)
        return _failed(applied, task, "The task could not be queued.")
    return applied.model_copy(
        update={
            "message": _started_message(expert.name, task, started),
            "handoff": ExpertHandoffOutcome(
                title=task.title,
                status="started",
                sub_session_id=started.sub_session_id,
                link=started.sub_autopilot_session_link,
            ),
        }
    )


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


async def _start(
    user_id: str, session: ChatSession, expert_id: str, task: ExpertHandoffTask
) -> SubSessionStatusResponse:
    """Queue *task* for the new hire without waiting on it."""
    target = await resolve_target_expert(user_id, expert_id)
    if target is None:
        raise _NotStarted("They could not be found on the team right after hiring.")
    refusal = await chain_refusal(user_id, session, target)
    if refusal is not None:
        raise _NotStarted(refusal)
    started = await start_delegation(
        user_id=user_id,
        session=session,
        target=target,
        prompt=task.brief,
        wait_for_result=0,
    )
    if not isinstance(started, SubSessionStatusResponse) or (
        started.status not in _STARTED
    ):
        raise _NotStarted(started.message)
    return started


def _failed(
    applied: ExpertChangeAppliedResponse, task: ExpertHandoffTask, reason: str
) -> ExpertChangeAppliedResponse:
    expert = applied.expert
    return applied.model_copy(
        update={
            "message": _failed_message(expert.name, expert.id, task, reason),
            "handoff": ExpertHandoffOutcome(title=task.title, status="failed"),
        }
    )


def _started_message(
    name: str, task: ExpertHandoffTask, started: SubSessionStatusResponse
) -> str:
    where = (
        f" in their own chat at {started.sub_autopilot_session_link}"
        if started.sub_autopilot_session_link
        else ""
    )
    return (
        f"{name} is hired and has started on {task.title}{where}. Tell the "
        f"user in one short line that {name} is on it and where to follow "
        "along. Don't ask whether to hand it over: they chose that on the "
        "card. Their result comes back here; check it with "
        f"tool:get_sub_session_result (sub_session_id: {started.sub_session_id}) "
        "when you need it. Their workflows and skills finish installing in the "
        "background."
    )


def _failed_message(
    name: str, expert_id: str, task: ExpertHandoffTask, reason: str
) -> str:
    return (
        f"{name} is hired and on the team, but {task.title} did not reach them: "
        f"{reason} The user already approved handing it over on the card, so "
        f"start it with tool:delegate_to_expert (expert_id: {expert_id}) and the "
        "same task instead of asking them again."
    )
