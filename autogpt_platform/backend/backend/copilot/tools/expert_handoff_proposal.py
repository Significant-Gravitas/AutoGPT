"""The task a new teammate is proposed for, shared by both previews.

``hire_expert`` and ``raise_expert`` both take a task the user already
described (``handoff_title`` / ``handoff_task``). A proposal that carries one
is parked twice: the plain change under ``confirmation_id``, and the change
plus its task ("Hire and start" / "Create and start") under
``handoff_confirmation_id``. The card asks one question, so the user's one
click both adds the teammate and starts the work (see ``hire_handoff``); the
plain id stays for a user who asks in chat to add them without starting it.
"""

import uuid
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from backend.data.redis_client import get_redis_async

from .expert_change_messages import NewTeammateKind
from .expert_proposal import ExpertChangeProposal, store_proposal
from .models import ExpertHandoffTask

# Long enough to name a task the way the user would, short enough to sit in
# a button beside the expert's name.
HANDOFF_TITLE_MAX_LENGTH = 80
HANDOFF_TASK_MAX_LENGTH = 4000

HANDOFF_PARAMETERS: dict[str, Any] = {
    "handoff_title": {
        "type": "string",
        "description": "That task in a few words, for the card.",
    },
    "handoff_task": {
        "type": "string",
        "description": "The task, written for the expert.",
    },
}


class _HandoffParams(BaseModel):
    handoff_title: str = Field(default="", max_length=HANDOFF_TITLE_MAX_LENGTH)
    handoff_task: str = Field(default="", max_length=HANDOFF_TASK_MAX_LENGTH)


def parse_handoff(
    handoff_title: str, handoff_task: str
) -> ExpertHandoffTask | None | str:
    """The task the proposal carries, ``None`` for none, or why the two
    fields are refused. The title is collapsed to one line so it fits the
    card's question."""
    try:
        params = _HandoffParams(
            handoff_title=" ".join(handoff_title.split()),
            handoff_task=handoff_task.strip(),
        )
    except ValidationError as e:
        return f"Invalid task — {_validation_detail(e)}"
    if bool(params.handoff_title) != bool(params.handoff_task):
        return "Pass handoff_title and handoff_task together, or neither."
    if not params.handoff_task:
        return None
    return ExpertHandoffTask(title=params.handoff_title, brief=params.handoff_task)


async def park_proposal(proposal: ExpertChangeProposal) -> tuple[str, str | None]:
    """Park *proposal* under a fresh id and, when it carries a task, its
    "and start" twin under a second; returns both ids."""
    redis = await get_redis_async()
    confirmation_id = str(uuid.uuid4())
    if proposal.handoff is None:
        await store_proposal(redis, confirmation_id, proposal)
        return confirmation_id, None
    handoff_confirmation_id = str(uuid.uuid4())
    await store_proposal(
        redis,
        confirmation_id,
        proposal.model_copy(
            update={"sibling_confirmation_id": handoff_confirmation_id}
        ),
    )
    await store_proposal(
        redis,
        handoff_confirmation_id,
        proposal.model_copy(
            update={"hand_off": True, "sibling_confirmation_id": confirmation_id}
        ),
    )
    return confirmation_id, handoff_confirmation_id


def proposed_message(kind: NewTeammateKind, with_handoff: bool) -> str:
    """What Otto is told while the user looks at the card."""
    if kind == "hire":
        lead = "Nothing hired yet. The user is looking at this expert on a card"
        quiet = "do not describe them in text"
        plain = "hire"
    else:
        lead = "Nothing created yet. The user is looking at this charter on a card"
        quiet = "do not repeat any of it in text"
        plain = "create the expert"
    if not with_handoff:
        return (
            f"{lead} with Approve and Decline buttons — {quiet}. Reply with "
            "one short line at most and wait. Only after they explicitly "
            "approve, call tool:confirm_expert_change with this confirmation_id."
        )
    return (
        f"{lead} with two buttons: {kind_verb(kind)} and start the task, or "
        f"not now — {quiet}. Reply with one short line at most and wait. If "
        "they approve, call tool:confirm_expert_change with "
        "handoff_confirmation_id. Use confirmation_id only if they ask to "
        f"{plain} without starting the task. Never ask in text whether to hand "
        "the task over."
    )


def kind_verb(kind: NewTeammateKind) -> str:
    return "hire" if kind == "hire" else "create"


def _validation_detail(error: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}"
        for err in error.errors()
    )
