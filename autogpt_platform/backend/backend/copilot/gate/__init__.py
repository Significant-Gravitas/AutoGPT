"""AutoPilot auto mode — one permission gate in front of every tool call.

Ordering is the design. Cheap and certain first, and the classifier last and
least trusted, because the static tiers in ``policy.py`` are what have to hold
if it is wrong or compromised:

1. gate inactive                      -> ALLOW (today's behaviour, unchanged)
2. an approval for exactly these args -> ALLOW, consumed single-use
3. tier DEFER                         -> ALLOW; another gate owns this call
4. escalated this session             -> ASK
5. tier ALWAYS_ASK                    -> ASK
6. tainted and effectful              -> ASK, classifier skipped
7. tier READ                          -> ALLOW
8. tier JUDGED                        -> classifier

Step 6 skips the classifier deliberately: that is precisely the case where the
injected text is sitting in the arguments the classifier would be reading.
"""

import logging
from dataclasses import dataclass
from typing import Any

from prisma.enums import ReviewStatus

from backend.copilot.model import ChatSession
from backend.util.feature_flag import Flag, is_feature_enabled

from . import review as review_store
from . import taint
from .classifier import classify
from .policy import Tier, escalates_under_taint, tier_for

logger = logging.getLogger(__name__)

_ALREADY_WAITING = (
    "Another action in this chat is already waiting for the user's approval. "
    "Stop and let them answer that one first; do not retry or try another way."
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


@dataclass(frozen=True)
class Decision:
    """``allowed`` is the only field the caller may act on."""

    allowed: bool
    reason: str = ""
    review_id: str | None = None
    already_waiting: bool = False


ALLOW = Decision(allowed=True)


async def gate_active(user_id: str | None, session: ChatSession) -> bool:
    """Auto mode runs only where a human can actually answer.

    Automation sessions (the scheduler, ``AutoPilotBlock``, ``run_sub_session``)
    and legacy rows with no origin keep today's ungated behaviour. Parking a
    question in a run nobody is watching is a stall, not a safeguard — and
    refusing there instead would silently kill shipped behaviour like the
    weekly "post an update in #standup" follow-up. Unattended work is
    authorized by the interactive act that created it, which is why the
    delegation tools are ALWAYS_ASK and ``schedule_followup`` escalates under
    taint.
    """
    if not user_id or session.metadata.origin != "interactive":
        return False
    if session.metadata.auto_mode is False:
        return False
    return await is_feature_enabled(Flag.COPILOT_AUTO_MODE, user_id, default=False)


async def check_action(
    tool_name: str,
    args: dict[str, Any],
    user_id: str | None,
    session: ChatSession,
    *,
    tool_description: str = "",
) -> Decision:
    if not await gate_active(user_id, session):
        return ALLOW
    assert user_id is not None

    session_id = session.session_id
    review_id = review_store.review_id_for(session_id, user_id, tool_name, args)

    status = await review_store.find_decision(review_id, user_id, session_id)
    if status == ReviewStatus.APPROVED:
        if await review_store.consume(review_id, user_id):
            return ALLOW
        return Decision(allowed=False, reason=_ALREADY_WAITING, already_waiting=True)
    if status == ReviewStatus.REJECTED:
        await review_store.consume(review_id, user_id)
        await taint.escalate(session_id, tool_name)
        return Decision(allowed=False, reason=_REJECTED)

    tier = tier_for(tool_name)
    if tier is Tier.DEFER:
        return ALLOW

    reason = await _verdict(tier, tool_name, tool_description, args, session)
    if reason is None:
        return ALLOW
    return await _park(review_id, user_id, session, tool_name, args, reason)


async def _verdict(
    tier: Tier,
    tool_name: str,
    tool_description: str,
    args: dict[str, Any],
    session: ChatSession,
) -> str | None:
    """The reason this call needs a human, or None to let it through."""
    if await taint.is_escalated(session.session_id, tool_name):
        return "You declined this action earlier in this chat."
    if tier is Tier.ALWAYS_ASK:
        return "This action always needs your approval."
    if escalates_under_taint(tool_name) and await taint.is_tainted(session):
        return (
            "This chat has read content from outside the platform, so actions "
            "with lasting effects need your approval."
        )
    if tier is Tier.READ:
        return None
    allowed, reason = await classify(
        tool_name=tool_name,
        tool_description=tool_description,
        args=args,
        user_message=_last_user_message(session),
        tainted=await taint.is_tainted(session),
    )
    return None if allowed else reason


async def _park(
    review_id: str,
    user_id: str,
    session: ChatSession,
    tool_name: str,
    args: dict[str, Any],
    reason: str,
) -> Decision:
    if await review_store.has_open_review(user_id, session.session_id):
        return Decision(allowed=False, reason=_ALREADY_WAITING, already_waiting=True)
    if not await review_store.open_review(
        review_id, user_id, session, tool_name, args, reason
    ):
        return Decision(allowed=False, reason=_UNRECORDABLE)
    return Decision(allowed=False, reason=reason, review_id=review_id)


def _last_user_message(session: ChatSession) -> str:
    for message in reversed(session.messages):
        if message.role == "user" and message.content:
            return message.content
    return ""


async def note_taint_source(session_id: str, tool_name: str) -> None:
    """Mark the session before a taint source runs, not after it succeeds.

    Parallel dispatch is deliberate here, so a ``bash_exec`` issued alongside a
    ``web_fetch`` would otherwise read the flag before the fetch wrote it.
    """
    await taint.mark_tainted(session_id, tool_name)


__all__ = ["Decision", "check_action", "gate_active", "note_taint_source"]
