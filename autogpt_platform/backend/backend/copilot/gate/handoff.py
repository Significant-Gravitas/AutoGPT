"""The hand-off card: who a held ``delegate_to_expert`` goes to, and what.

A generic card shows a hand-off as "Hand a task to a teammate" plus raw
arguments. Under ``payload.handoff`` the card gets the teammate (name, role,
avatar), the brief and the model's reason instead, so the user can judge the
hand-off itself. A hand-off card is editable: the user may rewrite the brief
or pick another teammate, and :func:`edited_args` reads that edit back into
the call that runs.
"""

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

from pydantic import BaseModel, ValidationError

from backend.api.features.experts.models import Expert

logger = logging.getLogger(__name__)

HANDOFF_TOOL = "delegate_to_expert"
# The card holds the whole brief so an edit starts from it, not from the
# clipped copy under ``arguments``; beyond this it is cut, and an unedited cut
# brief is never mistaken for an edit (see ``edited_args``).
BRIEF_MAX_CHARS = 8_000

# The review id of an edited hand-off the user approved on its card. The edit
# changes the arguments, so no approval exists for the call that runs; this
# lets exactly that call through, only while the answered card runs it.
_approved_edit: ContextVar[str | None] = ContextVar("approved_edit", default=None)


class HandoffCard(BaseModel):
    expert_id: str
    expert_name: str
    expert_role: str = ""
    expert_avatar_url: str | None = None
    expert_color: str = ""
    brief: str
    why: str | None = None


async def handoff_card(
    tool_name: str, args: dict[str, Any], user_id: str
) -> HandoffCard | None:
    """The card block for a held hand-off; None for any other call."""
    if tool_name != HANDOFF_TOOL:
        return None
    reference = str(args.get("expert_id") or "").strip()
    expert = await _resolve(user_id, reference) if reference else None
    reason = str(args.get("reason") or "").strip()
    return HandoffCard(
        expert_id=expert.id if expert else reference,
        expert_name=expert.name if expert else reference,
        expert_role=expert.role if expert else "",
        expert_avatar_url=expert.avatar_url if expert else None,
        expert_color=expert.color if expert else "",
        brief=str(args.get("prompt") or "")[:BRIEF_MAX_CHARS],
        why=reason or None,
    )


def edited_args(
    tool_name: str, args: dict[str, Any], payload: dict[str, Any]
) -> dict[str, Any] | None:
    """The call as the user edited it on the card, or None if unedited.

    Only the brief and the teammate can change; every other argument is the
    one the model made.
    """
    if tool_name != HANDOFF_TOOL:
        return None
    try:
        card = HandoffCard.model_validate(payload.get("handoff"))
    except ValidationError:
        return None
    changed: dict[str, str] = {}
    brief = card.brief.strip()
    # A brief cut to the card's cap reads back unchanged, not as an edit.
    if brief and card.brief != str(args.get("prompt") or "")[:BRIEF_MAX_CHARS]:
        changed["prompt"] = brief
    expert_id = card.expert_id.strip()
    if expert_id and expert_id != str(args.get("expert_id") or "").strip():
        changed["expert_id"] = expert_id
    return {**args, **changed} if changed else None


@contextmanager
def approved_edit(review_id: str) -> Iterator[None]:
    """Let the edited call bound to *review_id* past the gate, once."""
    token = _approved_edit.set(review_id)
    try:
        yield
    finally:
        _approved_edit.reset(token)


def is_approved_edit(review_id: str) -> bool:
    return _approved_edit.get() == review_id


async def _resolve(user_id: str, reference: str) -> Expert | None:
    # Deferred: the tools package imports the gate.
    from backend.copilot.tools.expert_delegation import resolve_target_expert

    try:
        return await resolve_target_expert(user_id, reference)
    except Exception:
        logger.warning(f"Hand-off card could not resolve {reference}", exc_info=True)
        return None
