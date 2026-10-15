"""AutoPilot approval modes — one gate in front of every tool call.

Ordering is the design, cheapest and most certain first:

1. gate inactive                          -> ALLOW (today's behaviour); in
   shadow mode the steps below run after the call is let through, and their
   verdict is recorded instead of enforced (``shadow.py``)
2. an approval for exactly these args     -> ALLOW, consumed single-use
3. what the call acts on: a block, workflow or MCP tool's own effect, or
   the tool's; a call that runs nothing never asks
4. the user's rule on that subject in this chat -> allow, judge or ask
5. otherwise the mode's verdict for that effect: run, ask, or the supervisor

The supervisor is last because it is the least trusted step: it can only turn
a run into a question, never the reverse.
"""

import logging
from functools import partial
from typing import Any, Awaitable, Callable, Literal

from prisma.enums import ReviewStatus
from pydantic import BaseModel, ConfigDict

from backend.copilot.context import get_current_envelope
from backend.copilot.model import ChatSession
from backend.copilot.tree import raise_ceiling, spent_past_ceiling
from backend.platform_linking.models import Platform
from backend.util.feature_flag import Flag, is_feature_enabled

from . import chat_rules, held
from . import review as review_store
from . import shadow
from .classifier import DecidedBy, supervise
from .headline import Headline
from .policy import (
    DEFAULT_MODE,
    PARKABLE,
    AutopilotMode,
    Effect,
    Verdict,
    effect_for,
    estimate_for,
    verdict_for_effect,
)
from .subject import Subject

logger = logging.getLogger(__name__)

_ALREADY_HELD = "This exact call is already waiting for the user's approval."
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
# Where the supervisor reads the content of the files a command runs.
RUN_FILES_KEY = "files_this_command_runs"
# A chat driven from these runs in Auto, and its cards are answered in the channel.
LINKED_CHAT_PLATFORMS = frozenset({"discord", "slack", "teams", "telegram"})
# Any other linked platform has no buttons to answer a card with, so it runs
# ungated: a held call would strand the chat.
CARDLESS_PLATFORMS = (
    frozenset(p.value.lower() for p in Platform) - LINKED_CHAT_PLATFORMS
)
# One approval of a paid read over the ceiling buys one more dollar.
CEILING_UNIT_MICRODOLLARS = 1_000_000
# Paid steps that otherwise run in every mode; the costliest blocks are workspace.
METERED = frozenset({Effect.READ, Effect.WORKSPACE})
# The user's own word on the subject in this chat outranks the mode's rule.
_RULE_VERDICTS = {
    "allow": Verdict.RUN,
    "judge": Verdict.JUDGE,
    "ask": Verdict.ASK,
    "unreadable": Verdict.ASK,
}


class Decision(BaseModel):
    """``allowed`` is the only field the caller may act on."""

    model_config = ConfigDict(frozen=True)

    allowed: bool
    reason: str = ""
    review_id: str | None = None
    # The held card's headline, ids resolved, for the chat's own row.
    headline: Headline | None = None
    # The user approved this exact call on a card, so nothing downstream asks again.
    approved: bool = False


ALLOW = Decision(allowed=True)

GateMode = Literal["enforce", "shadow", "off"]


class Ruling(BaseModel):
    """What the gate decides about a call, before anything is parked."""

    model_config = ConfigDict(frozen=True)

    allowed: bool
    effect: Effect
    # Why, as a fixed code: the policy step that decided, or the supervisor's code.
    code: str
    reason: str = ""
    reason_kind: review_store.ReasonKind = "mode"
    rule_key: str = ""
    subject: Subject | None = None
    decided_by: DecidedBy | None = None
    spend: dict[str, int] | None = None


async def gate_active(user_id: str | None, session: ChatSession) -> bool:
    return await gate_mode(user_id, session) == "enforce"


async def gate_mode(user_id: str | None, session: ChatSession) -> GateMode:
    """The gate runs only where a signed-in user can answer; sessions nobody
    is watching stay ungated until they get their own path. Enforce wins over
    shadow, which judges and records without holding anything."""
    if not user_id or session.metadata.origin != "interactive":
        return "off"
    if session.metadata.source_platform in CARDLESS_PLATFORMS:
        return "off"
    if await is_feature_enabled(Flag.COPILOT_AUTO_MODE, user_id, default=False):
        return "enforce"
    if await is_feature_enabled(Flag.COPILOT_AUTO_MODE_SHADOW, user_id, default=False):
        return "shadow"
    return "off"


def resolve_mode(session: ChatSession) -> AutopilotMode:
    if session.metadata.source_platform in LINKED_CHAT_PLATFORMS:
        return "auto"
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
    subject_of: Callable[[], Awaitable[Subject | None]] | None = None,
    context_of: Callable[[], Awaitable[dict[str, str | None] | None]] | None = None,
) -> Decision:
    """``subject_of`` resolves what the call acts on; it runs only once no
    approval answers the call, so an approved call is never re-derived.
    ``context_of`` reads the files the call runs, for the supervisor only."""
    gating = await gate_mode(user_id, session)
    if gating == "off":
        return ALLOW
    assert user_id is not None

    # Reads, workspace work and the ungated tools run in every mode and can
    # never have been parked, so they skip the review and rule lookups; a paid
    # step can be parked over the ceiling, so it cannot.
    if effect_for(tool_name) not in PARKABLE and not estimate_for(tool_name):
        return ALLOW

    if gating == "shadow":
        # Nothing is ever parked in shadow, so there is no review to look up.
        shadow.judge_later(
            user_id,
            session,
            kind="action",
            mode=resolve_mode(session),
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            judge=partial(
                _shadow_finding,
                tool_name,
                args,
                user_id,
                session,
                subject_of,
                context_of,
            ),
        )
        return ALLOW

    session_id = session.session_id
    review_id = review_store.review_id_for(session_id, user_id, tool_name, args)

    review = await review_store.find_review(review_id, user_id, session_id)
    if review is not None and review.status == ReviewStatus.APPROVED:
        if await review_store.consume(review_id, user_id):
            if review_store.is_spend_card(review):
                await raise_ceiling(CEILING_UNIT_MICRODOLLARS)
            return Decision(allowed=True, approved=True)
        return Decision(allowed=False, reason=_CONSUMED)
    if review is not None and review.status == ReviewStatus.REJECTED:
        await review_store.consume(review_id, user_id)
        await chat_rules.set_ask(
            session_id,
            await held.rule_key(session_id, review_id, tool_name),
            user_id,
            session.expert_id,
        )
        return Decision(allowed=False, reason=_REJECTED)
    if review is not None and review.status == ReviewStatus.WAITING:
        # The first call's card and stored call stand; re-storing would
        # re-point the late result at the retry's tool call id.
        return Decision(allowed=False, reason=_ALREADY_HELD, review_id=review_id)

    ruling = await _rule(
        tool_name, args, user_id, session, subject_of, context_of, metered=True
    )
    if ruling.allowed:
        return ALLOW
    call = held.HeldCall(
        review_id=review_id,
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        args=args,
        rule_key=ruling.rule_key,
        envelope=get_current_envelope(),
    )
    return await _park(
        call,
        user_id,
        session,
        ruling.reason,
        ruling.reason_kind,
        ruling.subject,
        ruling.decided_by,
        ruling.spend,
    )


async def _rule(
    tool_name: str,
    args: dict[str, Any],
    user_id: str,
    session: ChatSession,
    subject_of: Callable[[], Awaitable[Subject | None]] | None,
    context_of: Callable[[], Awaitable[dict[str, str | None] | None]] | None,
    *,
    metered: bool,
) -> Ruling:
    """Steps 3 to 5: the subject, the user's rule, then the mode's verdict.
    ``metered``: the turn's spend is counted, so a paid step can ask."""
    subject = await subject_of() if subject_of is not None else None
    effect = subject.effect if subject is not None else effect_for(tool_name)
    if effect is Effect.UNGATED:
        return Ruling(allowed=True, effect=effect, code="ungated")
    rule_key = subject.key if subject is not None else tool_name
    mode = resolve_mode(session)
    # Only a subject that can be parked can carry a rule, so reads and
    # workspace work skip the Redis round trip.
    hit = (
        await chat_rules.rule_for(
            session.session_id, rule_key, user_id, session.expert_id
        )
        if effect in PARKABLE
        else None
    )
    rule = hit.rule if hit else None
    # A judge rule covers irreversible subjects too: the user chose the supervisor.
    verdict = _RULE_VERDICTS[rule] if rule else verdict_for_effect(mode, effect)
    estimate = subject.estimate if subject is not None else estimate_for(tool_name)
    held_as = partial(
        Ruling, allowed=False, effect=effect, rule_key=rule_key, subject=subject
    )
    spend = None
    if metered and effect in METERED and estimate > 0 and mode != "unsupervised":
        spend = await spent_past_ceiling(user_id)
    if hit and rule in ("ask", "unreadable"):
        return held_as(reason=hit.reason, reason_kind="rule", code=f"rule_{rule}")
    if spend is not None:
        reason = (
            f"costs about {_dollars(estimate)}, and this chat has spent "
            f"{_dollars(spend[0])} of its {_dollars(spend[1])} ceiling; approving "
            f"adds {_dollars(CEILING_UNIT_MICRODOLLARS)} to it"
        )
        return held_as(
            reason=reason,
            reason_kind="spend",
            code="spend_ceiling",
            spend=_spend_shown(estimate, *spend),
        )
    if verdict is Verdict.RUN:
        code = "rule_allow" if rule == "allow" else "mode_run"
        return Ruling(allowed=True, effect=effect, code=code)
    if verdict is Verdict.ASK and subject is not None and subject.reason:
        return held_as(reason=subject.reason, reason_kind="subject", code="subject_ask")
    if verdict is Verdict.ASK:
        reason = _ASK_FIRST if mode == "ask_first" else _OUTWARD
        return held_as(reason=reason, code="mode_ask")
    files = await context_of() if context_of is not None else None
    if unread := [path for path, text in (files or {}).items() if text is None]:
        reason = f"Could not read {', '.join(unread)}, so this could not be checked."
        return held_as(reason=reason, reason_kind="supervisor", code="files_unreadable")
    judgement = await supervise(
        tool_name=tool_name,
        args=_judged_args(args, files),
        user_message=_last_user_message(session),
    )
    if judgement.allowed:
        return Ruling(
            allowed=True,
            effect=effect,
            code=judgement.code,
            decided_by=judgement.decided_by,
        )
    return held_as(
        reason=judgement.reason,
        reason_kind="supervisor",
        code=judgement.code,
        decided_by=judgement.decided_by,
    )


async def _shadow_finding(
    tool_name: str,
    args: dict[str, Any],
    user_id: str,
    session: ChatSession,
    subject_of: Callable[[], Awaitable[Subject | None]] | None,
    context_of: Callable[[], Awaitable[dict[str, str | None] | None]] | None,
) -> shadow.Finding:
    # Shadow never meters, and reading the ceiling would open the chat's ledger.
    ruling = await _rule(
        tool_name, args, user_id, session, subject_of, context_of, metered=False
    )
    return shadow.action_finding(
        allowed=ruling.allowed,
        code=ruling.code,
        effect=ruling.effect.value,
        decided_by=ruling.decided_by,
    )


async def _park(
    call: held.HeldCall,
    user_id: str,
    session: ChatSession,
    reason: str,
    reason_kind: review_store.ReasonKind,
    subject: Subject | None,
    decided_by: DecidedBy | None,
    spend: dict[str, int] | None = None,
) -> Decision:
    """Cards queue per chat: the call is kept so its answer can finish it."""
    if not await held.remember(session.session_id, call):
        return Decision(allowed=False, reason=_UNRECORDABLE)
    headline = await review_store.open_review(
        call.review_id,
        user_id,
        session,
        call.tool_name,
        call.args,
        reason,
        subject,
        spend=spend,
        reason_kind=reason_kind,
        tool_call_id=call.tool_call_id,
        decided_by=decided_by,
    )
    if headline is None:
        await held.forget(session.session_id, call.review_id)
        return Decision(allowed=False, reason=_UNRECORDABLE)
    return Decision(
        allowed=False, reason=reason, review_id=call.review_id, headline=headline
    )


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


def _spend_shown(estimate: int, spent: int, ceiling: int) -> dict[str, int]:
    """In microdollars: the card formats money itself."""
    return {
        "estimate": estimate,
        "spent": spent,
        "ceiling": ceiling,
        "unit": CEILING_UNIT_MICRODOLLARS,
    }


def _dollars(microdollars: int) -> str:
    return f"${max(microdollars, 0) / 1_000_000:,.2f}"


def _judged_args(
    args: dict[str, Any], files: dict[str, str | None] | None
) -> dict[str, Any]:
    # Dropped first, so a model cannot hand the supervisor a harmless copy.
    judged = {key: value for key, value in args.items() if key != RUN_FILES_KEY}
    if files:
        judged[RUN_FILES_KEY] = files
    return judged


def _last_user_message(session: ChatSession) -> str:
    # Deferred: copilot.service imports the tool registry, which imports this gate.
    from backend.copilot.service import strip_injected_context_for_display

    for message in reversed(session.messages):
        if message.role == "user" and message.content:
            if held.is_answer_row(message):
                continue
            # A first turn's row starts with the server's context blocks.
            return strip_injected_context_for_display(message.content)
    return ""


__all__ = [
    "RUN_FILES_KEY",
    "Decision",
    "GateMode",
    "LINKED_CHAT_PLATFORMS",
    "active_mode",
    "check_action",
    "gate_active",
    "gate_mode",
    "refusal_message",
    "resolve_mode",
]
