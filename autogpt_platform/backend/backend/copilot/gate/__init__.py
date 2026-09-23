"""AutoPilot approval modes — one gate in front of every tool call.

Ordering is the design, cheapest and most certain first:

1. gate inactive                          -> ALLOW (today's behaviour)
2. an approval for exactly these args     -> ALLOW, consumed single-use
3. the user rejected this tool in chat    -> ASK, in every mode
4. the mode's verdict for the tool's effect: run, ask, or the supervisor

The supervisor is last because it is the least trusted step: it can only turn
a run into a question, never the reverse.
"""

import logging
from typing import Any

from prisma.enums import ReviewStatus
from pydantic import BaseModel, ConfigDict

from backend.copilot.model import ChatSession
from backend.util.feature_flag import Flag, is_feature_enabled

from . import chat_rules, held
from . import review as review_store
from .classifier import classify
from .policy import (
    DEFAULT_MODE,
    AutopilotMode,
    Effect,
    Verdict,
    effect_for,
    verdict_for,
)

logger = logging.getLogger(__name__)

_CONSUMED = (
    "This approval was already used by an identical call that ran. "
    "Do not retry; tell the user what ran."
)
_REJECTED = (
    "The user declined this action. Do not retry it, do not adjust the "
    "arguments and try again, and do not use a different tool to achieve the "
    "same effect. Tell them it was not done and ask what they want instead."
)
_UNRECORDABLE = (
    "This action needs the user's approval, but the approval request could "
    "not be recorded, so nothing ran. Tell the user and stop."
)
_ASK_FIRST = "Ask First is on for this chat, so this action needs your approval."
_OUTWARD = "This action reaches outside the platform, so it needs your approval."
_PARKABLE = frozenset({Effect.SHELL, Effect.PLATFORM, Effect.EXTERNAL})


class Decision(BaseModel):
    """``allowed`` is the only field the caller may act on."""

    model_config = ConfigDict(frozen=True)

    allowed: bool
    reason: str = ""
    review_id: str | None = None


ALLOW = Decision(allowed=True)


async def gate_active(user_id: str | None, session: ChatSession) -> bool:
    """The gate runs only where a signed-in user can answer; sessions nobody
    is watching stay ungated until they get their own path."""
    if not user_id or session.metadata.origin != "interactive":
        return False
    return await is_feature_enabled(Flag.COPILOT_AUTO_MODE, user_id, default=False)


def resolve_mode(session: ChatSession) -> AutopilotMode:
    return session.metadata.autopilot_mode or DEFAULT_MODE


async def active_mode(
    user_id: str | None, session: ChatSession
) -> AutopilotMode | None:
    """The mode the gate enforces on this turn, or None when it is inert."""
    return resolve_mode(session) if await gate_active(user_id, session) else None


async def check_action(
    tool_name: str,
    args: dict[str, Any],
    user_id: str | None,
    session: ChatSession,
    tool_call_id: str = "",
) -> Decision:
    if not await gate_active(user_id, session):
        return ALLOW
    assert user_id is not None

    # Reads, workspace work and the ungated tools run in every mode and can
    # never have been parked, so they skip the review and rule lookups.
    if effect_for(tool_name) not in _PARKABLE:
        return ALLOW

    session_id = session.session_id
    review_id = review_store.review_id_for(session_id, user_id, tool_name, args)

    status = await review_store.find_decision(review_id, user_id, session_id)
    if status == ReviewStatus.APPROVED:
        if await review_store.consume(review_id, user_id):
            return ALLOW
        return Decision(allowed=False, reason=_CONSUMED)
    if status == ReviewStatus.REJECTED:
        await review_store.consume(review_id, user_id)
        await chat_rules.set_ask(session_id, tool_name)
        return Decision(allowed=False, reason=_REJECTED)

    mode = resolve_mode(session)
    verdict = verdict_for(mode, tool_name)
    if await chat_rules.asks(session_id, tool_name):
        reason = "You declined this action earlier in this chat."
    elif verdict is Verdict.RUN:
        return ALLOW
    elif verdict is Verdict.ASK:
        reason = _ASK_FIRST if mode == "ask_first" else _OUTWARD
    else:
        allowed, reason = await classify(
            tool_name=tool_name,
            args=args,
            user_message=_last_user_message(session),
        )
        if allowed:
            return ALLOW
    call = held.HeldCall(
        review_id=review_id, tool_name=tool_name, tool_call_id=tool_call_id, args=args
    )
    return await _park(call, user_id, session, reason)


async def _park(
    call: held.HeldCall, user_id: str, session: ChatSession, reason: str
) -> Decision:
    """Cards queue per chat: the call is kept so its answer can finish it."""
    if not await held.remember(session.session_id, call):
        return Decision(allowed=False, reason=_UNRECORDABLE)
    if not await review_store.open_review(
        call.review_id, user_id, session, call.tool_name, call.args, reason
    ):
        return Decision(allowed=False, reason=_UNRECORDABLE)
    return Decision(allowed=False, reason=reason, review_id=call.review_id)


def refusal_message(reason: str, review_id: str | None) -> str:
    """What the model reads instead of a result."""
    if review_id is None:
        return (
            f"Nothing ran. {reason} Tell the user what you wanted to do and why, "
            "then stop. Do not retry or reach the same effect another way."
        )
    return (
        f"Held for the user's approval (review {review_id}); nothing has run "
        f"yet. {reason} If they approve, the result arrives later as a "
        "<held_call_result> naming this call. Carry on with whatever does not "
        "depend on it. Do not retry it or reach the same effect another way."
    )


def _last_user_message(session: ChatSession) -> str:
    for message in reversed(session.messages):
        if message.role == "user" and message.content:
            return message.content
    return ""


__all__ = [
    "Decision",
    "active_mode",
    "check_action",
    "gate_active",
    "refusal_message",
    "resolve_mode",
]
