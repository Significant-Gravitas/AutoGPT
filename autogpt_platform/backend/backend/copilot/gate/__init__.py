"""AutoPilot approval modes — one gate in front of every tool call.

Ordering is the design, cheapest and most certain first:

1. gate inactive                          -> ALLOW (today's behaviour)
2. an approval for exactly these args     -> ALLOW, consumed single-use;
   a proposal card's Approve of the id a confirm applies counts as one
3. what the call acts on: a block, workflow or MCP tool's own effect, or
   the tool's; a call that runs nothing never asks
4. the user's rule on that subject in this chat -> allow, judge or ask
5. otherwise the mode's verdict for that effect: run, ask, or the supervisor

The supervisor is last because it is the least trusted step: it can only turn
a run into a question, never the reverse.
"""

import logging
from typing import Any, Awaitable, Callable, Literal, TypeGuard

from prisma.enums import ReviewStatus
from pydantic import BaseModel, ConfigDict

from backend.copilot.context import get_current_envelope
from backend.copilot.legacy_first_turn_memory import without_stored_first_turn_memory
from backend.copilot.model import ChatSession
from backend.copilot.tree import raise_ceiling, spent_past_ceiling
from backend.copilot.trial_cost_context import capture_cost_context
from backend.platform_linking.models import Platform
from backend.util.feature_flag import Flag, is_feature_enabled

from . import chat_rules, held
from . import review as review_store
from .card_approval import approved_on_card
from .classifier import DecidedBy, Judgement, supervise
from .headline import Headline
from .policy import (
    DEFAULT_MODE,
    PARKABLE,
    AutopilotMode,
    Effect,
    Verdict,
    effect_for,
    estimate_for,
    strictest_mode,
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
# Where it reads what a judged outward block is: the call names only its id.
RUNS_KEY = "capability_this_call_runs"
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
# Decision 1 (OPEN-3592): the mode's table is final, so Unsupervised runs an
# irreversible block without asking. False makes Unsupervised ask before one.
UNSUPERVISED_RUNS_IRREVERSIBLE_BLOCKS = True


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
    # An active gate reached a verdict on this call: a card, a rule, the mode
    # or the supervisor. With unified approvals on, nothing downstream asks again.
    decided: bool = False


class Ruling(BaseModel):
    """The user's rule, the mode or the supervisor on one subject."""

    model_config = ConfigDict(frozen=True)

    run: bool
    # Why it asks, or what let it run: the audit line on an approved step.
    reason: str = ""
    reason_kind: review_store.ReasonKind = "mode"
    by: Literal["rule", "mode", "supervisor"] = "mode"
    decided_by: DecidedBy | None = None
    spend: dict[str, int] | None = None


ALLOW = Decision(allowed=True)
DECIDED = Decision(allowed=True, decided=True)


async def gate_active(user_id: str | None, session: ChatSession) -> bool:
    if not answerable(user_id, session):
        return False
    return await is_feature_enabled(Flag.COPILOT_AUTO_MODE, user_id, default=False)


def answerable(user_id: str | None, session: ChatSession) -> TypeGuard[str]:
    """Whether a signed-in user can answer a card here; sessions nobody is
    watching stay unchecked until they get their own path."""
    if not user_id or session.metadata.origin != "interactive":
        return False
    return session.metadata.source_platform not in CARDLESS_PLATFORMS


async def unified_approvals(user_id: str) -> bool:
    """On by decision, so a read that fails means on."""
    try:
        return await is_feature_enabled(
            Flag.COPILOT_UNIFIED_APPROVALS, user_id, default=True
        )
    except Exception:
        logger.warning("Unified approvals flag unreadable; on", exc_info=True)
        return True


def resolve_mode(session: ChatSession) -> AutopilotMode:
    if session.metadata.source_platform in LINKED_CHAT_PLATFORMS:
        return "auto"
    return session.metadata.autopilot_mode or DEFAULT_MODE


async def active_mode(
    user_id: str | None, session: ChatSession
) -> AutopilotMode | None:
    """The mode the gate enforces on this turn, or None when it is inert."""
    return resolve_mode(session) if await gate_active(user_id, session) else None


def child_autopilot_mode(
    parent: ChatSession, expert_default: AutopilotMode | None
) -> AutopilotMode:
    """The mode a delegated or handed-off thread starts on: what the parent's
    gate enforces, not what the parent stored, tightened by the expert's own
    default where that is stricter."""
    mode = resolve_mode(parent)
    return mode if expert_default is None else strictest_mode(mode, expert_default)


async def check_action(
    tool_name: str,
    args: dict[str, Any],
    user_id: str | None,
    session: ChatSession,
    tool_call_id: str = "",
    subject_of: Callable[[], Awaitable[Subject | None]] | None = None,
    context_of: Callable[[], Awaitable[dict[str, str | None] | None]] | None = None,
    args_bound: bool = False,
) -> Decision:
    """``subject_of`` resolves what the call acts on; it runs only once no
    approval answers the call, so an approved call is never re-derived.
    ``context_of`` reads the files the call runs, for the supervisor only.
    ``args_bound``: the tool already resolved ``args``, which a held call keeps
    so its replay does not resolve them again."""
    if not await gate_active(user_id, session):
        return ALLOW
    assert user_id is not None

    # Reads, workspace work and the ungated tools run in every mode and can
    # never have been parked, so they skip the review and rule lookups; a paid
    # step can be parked over the ceiling, so it cannot.
    if effect_for(tool_name) not in PARKABLE and not estimate_for(tool_name):
        return ALLOW

    session_id = session.session_id
    review_id = review_store.review_id_for(session_id, user_id, tool_name, args)

    review = await review_store.find_review(review_id, user_id, session_id)
    if review is not None and review.status == ReviewStatus.APPROVED:
        if await review_store.consume(review_id, user_id):
            if review_store.is_spend_card(review):
                await raise_ceiling(CEILING_UNIT_MICRODOLLARS)
            return Decision(allowed=True, approved=True, decided=True)
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
    if await approved_on_card(tool_name, args, user_id, session_id):
        return Decision(allowed=True, approved=True, decided=True)

    subject = await subject_of() if subject_of is not None else None
    effect = subject.effect if subject is not None else effect_for(tool_name)
    if effect is Effect.UNGATED:
        return ALLOW
    rule_key = subject.key if subject is not None else tool_name
    estimate = subject.estimate if subject is not None else estimate_for(tool_name)
    unsupervised = resolve_mode(session) == "unsupervised"

    async def over_ceiling() -> Ruling | None:
        if effect not in METERED or estimate <= 0 or unsupervised:
            return None
        spend = await spent_past_ceiling(user_id)
        if spend is None:
            return None
        return Ruling(
            run=False,
            reason=(
                f"costs about {_dollars(estimate)}, and this chat has spent "
                f"{_dollars(spend[0])} of its {_dollars(spend[1])} ceiling; "
                f"approving adds {_dollars(CEILING_UNIT_MICRODOLLARS)} to it"
            ),
            reason_kind="spend",
            spend=_spend_shown(estimate, *spend),
        )

    async def judge() -> Judgement:
        files = await context_of() if context_of is not None else None
        if unread := [path for path, text in (files or {}).items() if text is None]:
            return Judgement(
                allowed=False,
                reason=(
                    f"Could not read {', '.join(unread)}, so this could not be "
                    "checked."
                ),
            )
        outward_judged = effect is Effect.EXTERNAL_JUDGED
        # A workflow's name is its author's free text; only a block's is vouched for.
        runs = (
            subject.name
            if outward_judged
            and subject is not None
            and subject.key.startswith("block:")
            else None
        )
        return await supervise(
            tool_name=tool_name,
            args=_judged_args(args, files, runs),
            user_messages=_typed_messages(session),
            outward_judged=outward_judged,
        )

    ruling = await rule_on(
        session, user_id, rule_key, effect, subject, judge, over_ceiling
    )
    if ruling.run:
        return DECIDED
    call = held.HeldCall(
        review_id=review_id,
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        args=args,
        args_bound=args_bound,
        rule_key=rule_key,
        envelope=get_current_envelope(),
        cost_context=await capture_cost_context(user_id),
    )
    return await _park(
        call,
        user_id,
        session,
        ruling.reason,
        ruling.reason_kind,
        subject,
        ruling.decided_by,
        ruling.spend,
    )


async def rule_on(
    session: ChatSession,
    user_id: str,
    rule_key: str,
    effect: Effect,
    subject: Subject | None,
    judge: Callable[[], Awaitable[Judgement]],
    over_ceiling: Callable[[], Awaitable[Ruling | None]] | None = None,
    *,
    floor: bool = False,
) -> Ruling:
    """The user's rule on the subject in this chat, else the mode's verdict for
    its effect, the supervisor deciding where either says judge. ``floor``
    turns the mode's run into an ask; a rule still decides."""
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
    if hit and rule in ("ask", "unreadable"):
        return Ruling(run=False, reason=hit.reason, reason_kind="rule", by="rule")
    if over_ceiling is not None and (held_for_spend := await over_ceiling()):
        return held_for_spend
    # No supervisor reads a scheduled run, so a judge rule falls back to the mode.
    if rule == "judge" and subject is not None and subject.schedules:
        rule = None
    # A judge rule covers irreversible subjects too: the user chose the supervisor.
    verdict = _RULE_VERDICTS[rule] if rule else verdict_for_effect(mode, effect)
    if rule is None and verdict is Verdict.RUN and _mode_asks(mode, subject, floor):
        verdict = Verdict.ASK
    by = "rule" if rule else "mode"
    if verdict is Verdict.RUN:
        return Ruling(run=True, by=by)
    if verdict is Verdict.ASK and subject is not None and subject.reason:
        return Ruling(run=False, reason=subject.reason, reason_kind="subject", by=by)
    if verdict is Verdict.ASK:
        reason = _ASK_FIRST if mode == "ask_first" else _OUTWARD
        return Ruling(run=False, reason=reason, by=by)
    judgement = await judge()
    # Only an ask carries the supervisor's words onto a card.
    if judgement.allowed:
        return Ruling(run=True, by="supervisor")
    return Ruling(
        run=False,
        reason=judgement.reason,
        reason_kind="supervisor",
        by="supervisor",
        decided_by=judgement.decided_by,
    )


def _mode_asks(mode: AutopilotMode, subject: Subject | None, floor: bool) -> bool:
    irreversible_block = (
        subject is not None
        and subject.irreversible
        and subject.key.startswith("block:")
    )
    return floor or (
        mode == "unsupervised"
        and irreversible_block
        and not UNSUPERVISED_RUNS_IRREVERSIBLE_BLOCKS
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
    args: dict[str, Any], files: dict[str, str | None] | None, runs: str | None
) -> dict[str, Any]:
    # Dropped first, so a model cannot hand the supervisor a harmless copy.
    judged = {
        key: value
        for key, value in args.items()
        if key not in (RUN_FILES_KEY, RUNS_KEY)
    }
    if files:
        judged[RUN_FILES_KEY] = files
    if runs:
        judged[RUNS_KEY] = runs
    return judged


def _typed_messages(session: ChatSession, upto: int | None = None) -> list[str]:
    """What the user typed, oldest first, among the first ``upto`` messages when
    given; an answered card's rows and a row of nothing but the server's context
    are not the user's words."""
    # Deferred: copilot.service imports the tool registry, which imports this gate.
    from backend.copilot.service import strip_injected_context_for_display

    typed = [
        strip_injected_context_for_display(message.content)
        for message in without_stored_first_turn_memory(session.messages[:upto])
        if message.role == "user"
        and message.content
        and not held.is_answer_row(message)
    ]
    return [message for message in typed if message.strip()]


__all__ = [
    "RUNS_KEY",
    "RUN_FILES_KEY",
    "Decision",
    "Ruling",
    "LINKED_CHAT_PLATFORMS",
    "active_mode",
    "answerable",
    "check_action",
    "child_autopilot_mode",
    "gate_active",
    "refusal_message",
    "resolve_mode",
    "rule_on",
    "unified_approvals",
]
