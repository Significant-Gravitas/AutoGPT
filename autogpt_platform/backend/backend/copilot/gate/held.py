"""Held calls: what a parked call was, so its card's answer can finish it later.

A held call returns to the model at once, so the answer arrives after the turn
that made the call — often after it ended. The call is kept here exactly as
the model made it, and :func:`resolve_answered` runs it (or refuses it) at the
start of the chat's next turn, inside the engine, where the turn's sandbox and
tool bounds exist. Its result reaches the model as a user row naming the
original call. :func:`wake` starts that turn when none is running; a running
turn's end wakes it instead (``stream_registry.mark_session_completed``).
"""

import json
import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Callable, Iterable, Literal

from prisma.enums import ReviewStatus
from pydantic import BaseModel, Field, ValidationError

from backend.copilot import woken_turns
from backend.copilot.constants import COPILOT_NODE_EXEC_ID_SEPARATOR
from backend.copilot.context import envelope_scope, get_current_envelope
from backend.copilot.model import ChatSession
from backend.copilot.pending_messages import PendingMessage
from backend.copilot.session_permissions import resolve_session_permissions
from backend.copilot.tree import (
    MAX_DEPTH,
    TreeRefusal,
    TurnEnvelope,
    narrowed_by,
    readmit_turn,
    root_envelope,
)
from backend.copilot.turn_queue import is_users_own_chat
from backend.data.db_accessors import chat_db, review_db
from backend.data.redis_client import get_redis_async
from backend.util.encryption import JSONCryptor

from . import chat_rules
from . import review as review_store
from .policy import PARKABLE, effect_for

if TYPE_CHECKING:
    from backend.api.features.graph_executions.review.model import (
        PendingHumanReviewModel,
    )
    from backend.copilot.model import ChatMessage
    from backend.copilot.tools.base import BaseTool

logger = logging.getLogger(__name__)

# Outlives any card a user will still answer; the approval's own hour runs
# from the click, not from here.
_TTL_SECONDS = 30 * 24 * 60 * 60
_KEY = "copilot:gate:held:"
# Above the largest result either engine hands the model (the baseline's
# 100,000-character output cap plus the wrapper), so nothing is cut here.
_MAX_RESULT_CHARS = 120_000

Outcome = Literal["approved", "rejected", "expired", "closed", "unknown"]

WAKE_MESSAGE = "I answered an action that was waiting for my approval."
# Metadata on the user rows an answered card writes: the wake, and each result.
_WAKE_KEY = "held_calls_answered"
_RESULT_KEY = "held_call"
_RESEND = (
    "Nothing ran: the approved action's details were lost before it could run. "
    "Tell the user, and ask them to send the request again if it is still needed."
)
# Why an approved call may not run under the envelope it was held under, each
# read after "Nothing ran: ".
_UNRECORDED = "what this action was allowed to do was not recorded."
_UNREADABLE = "what this action was allowed to do could not be read back."
_UNBOUND = "what this action was allowed to do does not match its approval."
_NO_LONGER_OPEN = (
    "Nothing ran from this card: it is no longer open. An identical call may "
    "already have used its approval."
)
# A root's authority, untainted and tainted: all a card can be rebuilt under.
_ROOTS = (root_envelope("card"), root_envelope("card", tainted=True))


def is_answer_row(message: "ChatMessage") -> bool:
    """A user row the gate wrote for an answered card, not something the user typed."""
    return bool(message.metadata) and (
        _WAKE_KEY in message.metadata or _RESULT_KEY in message.metadata
    )


class HeldResult(PendingMessage):
    """A late result is capped by the engine, as a direct one is, not by the
    32,000-character limit on a typed follow-up."""

    content: str = Field(min_length=1, max_length=_MAX_RESULT_CHARS)


class HeldCall(BaseModel):
    review_id: str
    tool_name: str
    tool_call_id: str
    args: dict[str, Any]
    # What a rejection sets to ask for the rest of the chat; the tool when None.
    rule_key: str | None = None
    held_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # Rebuilt from a card whose stored copy of the arguments no longer binds.
    lost: bool = False
    # The turn that held it. Its wake starts under this, re-checked against the
    # tree; a call rebuilt from its card has none and is not woken.
    envelope: TurnEnvelope | None = None


async def remember(session_id: str, call: HeldCall) -> bool:
    """Record the call unless a record already stands under its review id,
    which is never overwritten: the id binds the call, its arguments and the
    authority it was held under, so a standing record is this call's own, and
    a write over it could only swap the authority its approval runs under.
    False means the call cannot be finished on approval, so it must not park."""
    try:
        redis = await get_redis_async()
        async with redis.pipeline(transaction=True) as pipe:
            # Encrypted: the arguments can hold a credential, and must stay exact
            # to match the approval, so they cannot be redacted like the card's.
            pipe.hsetnx(
                _key(session_id),
                call.review_id,
                JSONCryptor().encrypt(call.model_dump(mode="json")),
            )
            pipe.expire(_key(session_id), _TTL_SECONDS)
            await pipe.execute()
        return True
    except Exception:
        logger.warning(
            f"Gate could not store held call {call.review_id}", exc_info=True
        )
        return False


async def rule_key(session_id: str, review_id: str, tool_name: str) -> str:
    """What a rejection of this card sets to ask: its subject, else its tool."""
    call = (await _held(session_id)).get(review_id)
    return (call.rule_key if call else None) or tool_name


async def subject_keys(session_id: str, review_ids: list[str]) -> dict[str, str]:
    """The key each held card can set a rule on; a held read or a money card
    has none."""
    if not review_ids:
        return {}
    try:
        held = await _held(session_id)
    except Exception:
        # The approval still lands; only the rule is lost.
        logger.warning(
            f"Held calls unreadable for session {session_id}; approving without a rule",
            exc_info=True,
        )
        return {}
    return {
        review_id: call.rule_key
        for review_id in review_ids
        if (call := held.get(review_id))
        and call.rule_key
        and (call.rule_key != call.tool_name or effect_for(call.tool_name) in PARKABLE)
    }


async def forget(session_id: str, review_id: str) -> None:
    """Drop a call whose card was never opened; nothing can answer it."""
    await _claim(session_id, review_id)


async def answered(user_id: str, session_id: str) -> list[HeldCall]:
    """Held calls whose card has been answered, oldest first."""
    held = await _held(session_id)
    if not held:
        return []
    try:
        rows = await review_db().get_reviews_by_node_exec_ids(list(held), user_id)
    except Exception:
        logger.warning(
            f"Gate could not read held calls for session {session_id}", exc_info=True
        )
        return []
    return sorted(
        (
            held[review_id]
            for review_id, row in rows.items()
            if row.session_id == session_id and row.status != ReviewStatus.WAITING
        ),
        key=lambda call: call.held_at,
    )


async def resolve_answered(
    user_id: str | None,
    session: ChatSession,
    cap: Callable[[str], str] = lambda text: text,
) -> list[PendingMessage]:
    """Run every answered held call once and return its result as a user row.

    Call only once the turn's execution context is set. ``cap`` is the
    engine's own last step on a direct tool result, so a late one reads the
    same. The HDEL claims the call, so two turns racing over one card cannot
    both run it; the gate's own consume stays the second lock behind it.
    """
    if not user_id:
        return []
    delivered: list[PendingMessage] = []
    for call in await answered(user_id, session.session_id):
        if not await _claim(session.session_id, call.review_id):
            continue
        try:
            delivered.append(await _deliver(user_id, session, call, cap))
        except Exception:
            logger.warning(f"Held call {call.review_id} not delivered", exc_info=True)
            delivered.extend(await _recover(user_id, session.session_id, call))
    return delivered


async def wake(
    user_id: str,
    session_id: str,
    answered_rows: "Iterable[PendingHumanReviewModel]" = (),
) -> None:
    """Start a turn to carry answered cards, if the chat is idle.

    Best effort: a running turn's end calls this again, and any later turn
    folds the results in anyway. ``answered_rows`` are the cards just
    answered; one whose held call is gone from Redis is restored from its row.
    """
    # Deferred: the executor utilities import the tool registry.
    from backend.copilot.active_turns import (
        ConcurrentTurnLimitError,
        acquire_turn_slot,
        get_delegated_turn_limit,
        get_inflight_turn_limit,
    )
    from backend.copilot.executor.utils import dispatch_turn
    from backend.copilot.model import ChatMessage, append_and_save_message
    from backend.copilot.turn_queue import (
        UNRECORDED_WAKE,
        WAKE_LATER,
        InflightCapExceeded,
        post_refusal,
        try_enqueue_turn,
        wakes_sub_work,
    )

    try:
        await _restore(user_id, session_id, answered_rows)
        calls = await answered(user_id, session_id)
        if not calls:
            return
        # One wake per set of answered cards: a turn that died before its fold
        # must not be woken again for the same cards, turn after failed turn.
        wake_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{session_id}:" + ",".join(sorted(c.review_id for c in calls)),
            )
        )
        info = await chat_db().get_chat_session_metadata(session_id)
        if info is None or info.user_id != user_id:
            return
        refusal_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{wake_id}:refused"))
        # Bounds only the turn that reads the results: each call runs under
        # the envelope it was held under (``_own_envelope``), not this one.
        newest = max(calls, key=lambda c: c.held_at).envelope
        envelope = newest if newest is not None and _well_formed(newest) else None
        if envelope is None and not is_users_own_chat(info):
            # Deriving one from the turn that just ended would run the approved
            # action under that turn's limits, not the ones it was held under.
            await post_refusal(session_id, UNRECORDED_WAKE, message_id=refusal_id)
            return
        permissions = resolve_session_permissions(info)
        metadata = {_WAKE_KEY: True}
        capacity = get_delegated_turn_limit() if wakes_sub_work(info) else None
        try:
            async with acquire_turn_slot(
                user_id, session_id, capacity=capacity
            ) as slot:
                # Not admitted: a turn is already running, and its end wakes us.
                if not slot.admitted:
                    return
                if (
                    await append_and_save_message(
                        session_id,
                        ChatMessage(
                            id=wake_id,
                            role="user",
                            content=WAKE_MESSAGE,
                            metadata=metadata,
                        ),
                    )
                    is None
                ):
                    return
                turn_id = str(uuid.uuid4())
                await dispatch_turn(
                    slot,
                    session_id=session_id,
                    user_id=user_id,
                    turn_id=turn_id,
                    message=WAKE_MESSAGE,
                    organization_id=info.organization_id,
                    team_id=info.team_id,
                    llm_auth_provider=info.metadata.llm_auth_provider,
                    llm_credential_id=info.metadata.llm_credential_id,
                    permissions=permissions,
                    message_metadata=metadata,
                    envelope=envelope,
                    # Lost in the user's own chat: a root, as their message would be.
                    root=envelope is None,
                )
                # A channel that answered one of these follows this turn.
                await woken_turns.record(
                    session_id, [c.review_id for c in calls], turn_id
                )
        except ConcurrentTurnLimitError:
            await try_enqueue_turn(
                user_id=user_id,
                inflight_cap=get_inflight_turn_limit(),
                session_id=session_id,
                message=WAKE_MESSAGE,
                message_id=wake_id,
                message_metadata=metadata,
                llm_auth_provider=info.metadata.llm_auth_provider,
                llm_credential_id=info.metadata.llm_credential_id,
                permissions=(
                    permissions.model_dump(exclude_none=True) if permissions else None
                ),
                envelope=envelope,
            )
        except TreeRefusal as refused:
            await post_refusal(
                session_id, f"{refused.message} {WAKE_LATER}", message_id=refusal_id
            )
    except InflightCapExceeded:
        logger.info(f"Held calls in {session_id} wait for the user's next turn")
    except Exception:
        logger.warning(f"Could not wake session {session_id}", exc_info=True)


async def _deliver(
    user_id: str, session: ChatSession, call: HeldCall, cap: Callable[[str], str]
) -> PendingMessage:
    from backend.copilot.tools import get_tool

    rows = await review_db().get_reviews_by_node_exec_ids([call.review_id], user_id)
    card = rows.get(call.review_id)
    outcome, output = await _outcome(
        user_id, session, call, get_tool(call.tool_name), card
    )
    shown = _card_copy(card)
    try:
        return _result_row(call, cap(output), outcome, shown)
    except Exception:
        # The outcome is known and may be a refusal; only the engine's cut
        # failed, so deliver it trimmed rather than let recovery guess.
        logger.warning(f"Could not cap held result {call.review_id}", exc_info=True)
        return _result_row(call, output[: _MAX_RESULT_CHARS // 2], outcome, shown)


async def _recover(
    user_id: str, session_id: str, call: HeldCall
) -> list[PendingMessage]:
    """``_outcome`` failed. A card still open goes back for the next turn; a
    spent approval means the call reached the gate, so it may have run."""
    try:
        rows = await review_db().get_reviews_by_node_exec_ids([call.review_id], user_id)
        spent = call.review_id not in rows
    except Exception:
        spent = False
    if not spent:
        await remember(session_id, call)
        return []
    return [
        _result_row(
            call,
            "The approved action may have run, but its result was lost before it "
            "reached you. Tell the user, and check the outcome before relying "
            "on it.",
            "unknown",
        )
    ]


def _result_row(
    call: HeldCall,
    output: str,
    outcome: Outcome,
    shown: dict[str, Any] | None = None,
) -> PendingMessage:
    return HeldResult(
        content=(
            f'<held_call_result tool="{call.tool_name}" '
            f'tool_call_id="{call.tool_call_id}" review_id="{call.review_id}">\n'
            f"{output}\n</held_call_result>"
        ),
        metadata={
            _RESULT_KEY: {
                "review_id": call.review_id,
                "tool_name": call.tool_name,
                "tool_call_id": call.tool_call_id,
                # The chain row shows the answer without parsing the text.
                "outcome": outcome,
                **(shown or {}),
            }
        },
    )


def _card_copy(card: "PendingHumanReviewModel | None") -> dict[str, Any]:
    """The arguments as the card showed them, redacted and clipped by
    ``review_payload``, for the chain row once the card itself is consumed."""
    payload = card.payload if card and isinstance(card.payload, dict) else {}
    return {k: payload[k] for k in ("arguments", "fields", "clipped") if k in payload}


async def _outcome(
    user_id: str,
    session: ChatSession,
    call: HeldCall,
    tool: "BaseTool | None",
    row: "PendingHumanReviewModel | None",
) -> tuple[Outcome, str]:
    if row is None or row.status == ReviewStatus.WAITING:
        return "closed", _NO_LONGER_OPEN
    # Deferred: reads imports this package's __init__, which imports this module.
    from .reads import answered_read, is_held_read

    if is_held_read(call.review_id):
        return await answered_read(user_id, row)
    if row.status == ReviewStatus.REJECTED:
        await review_store.consume(call.review_id, user_id)
        await chat_rules.set_ask(
            session.session_id,
            call.rule_key or call.tool_name,
            user_id,
            session.expert_id,
        )
        return "rejected", (
            "Nothing ran: the user declined this action. Do not retry it or "
            "reach the same effect another way."
        )
    approved_at = row.reviewed_at or row.updated_at or row.created_at
    if datetime.now(UTC) - approved_at > review_store.APPROVAL_TTL:
        return await _invite_again(
            user_id,
            call,
            "expired",
            "Nothing ran: the approval expired an hour after it was given. "
            "Propose the call again if it is still needed.",
        )
    if call.lost:
        return await _invite_again(user_id, call, "closed", _RESEND)
    if tool is None:
        await review_store.consume(call.review_id, user_id)
        return "closed", "Nothing ran: this tool no longer exists."
    try:
        envelope = await _own_envelope(user_id, session, call)
    except TreeRefusal as refused:
        return await _invite_again(
            user_id,
            call,
            "closed",
            f"Nothing ran: {refused.message} Propose it again if it is still needed.",
        )
    with envelope_scope(envelope):
        # The gate finds the approval for exactly these arguments and spends it.
        result = await tool.execute(user_id, session, call.tool_call_id, **call.args)
    # With the flag switched off since, the gate ran it without spending the
    # approval; spend it here so no later identical call rides on it.
    await review_store.consume(call.review_id, user_id)
    if isinstance(result.output, str):
        return "approved", result.output
    return "approved", json.dumps(result.output, default=str)


async def _invite_again(
    user_id: str, call: HeldCall, outcome: Outcome, refusal: str
) -> tuple[Outcome, str]:
    """``refusal`` invites the call again, so it goes out only once this
    spent the approval: one still live would be ridden by the call proposed
    again. Gone already, an identical call may have used it, and nothing is
    invited. When neither is known it raises, and the call waits for the next
    turn."""
    if not await review_store.invalidate(call.review_id, user_id):
        return "closed", _NO_LONGER_OPEN
    return outcome, refusal


async def _own_envelope(
    user_id: str, session: ChatSession, call: HeldCall
) -> TurnEnvelope:
    """The envelope an approved call runs under: the one it was held under,
    re-checked now and narrowed by the session's permissions now, never the
    one of whichever turn folds it. Raises :class:`TreeRefusal`."""
    envelope = call.envelope
    if envelope is None:
        # Held before envelopes were kept, or rebuilt from its card. Every turn
        # in the user's own chat is a root, so a root stands in there for the
        # one that held it; anywhere else nothing says what it was allowed.
        running = get_current_envelope()
        if not is_users_own_chat(session) or (
            running is not None and running.depth != 0
        ):
            raise TreeRefusal(_UNRECORDED)
        envelope = running or root_envelope(
            call.review_id, session_id=session.session_id
        )
    else:
        await _readmitted(user_id, envelope)
    envelope = narrowed_by(envelope, resolve_session_permissions(session))
    if not envelope.permits(call.tool_name):
        raise TreeRefusal(
            f"{call.tool_name} is no longer allowed where it was proposed."
        )
    # The card was opened under the authority it was given for; a call that
    # would run under another cannot spend it, a card from before approvals
    # were bound among them.
    bound = review_store.action_review_id(
        session.session_id, user_id, call.tool_name, call.args, envelope
    )
    if bound != call.review_id:
        raise TreeRefusal(_UNBOUND)
    return envelope


async def cannot_act(user_id: str, envelope: TurnEnvelope | None) -> str | None:
    """Why an approval may not be spent under ``envelope`` now, if so: the
    gate asks this of every approved call, wherever it comes from."""
    if envelope is None:
        return None
    try:
        await _readmitted(user_id, envelope)
    except TreeRefusal as refused:
        return f"The approval cannot be used now: {refused.message}"
    return None


async def _readmitted(user_id: str, envelope: TurnEnvelope) -> None:
    """A recorded envelope that may still act: well formed, before its
    deadline, its tree open and its ceiling unspent. Raises
    :class:`TreeRefusal`."""
    if not _well_formed(envelope):
        raise TreeRefusal(_UNREADABLE)
    if envelope.deadline_at is not None and envelope.deadline_at <= datetime.now(UTC):
        raise TreeRefusal("the task that proposed it is past its deadline.")
    if envelope.depth == 0:
        # The per-user rate limit gates a root, never its tree, and admitting
        # one would open its ledgers with every approved call.
        return
    try:
        await readmit_turn(envelope, user_id=user_id)
    except TreeRefusal as refused:
        raise TreeRefusal(
            f"the task that proposed it may no longer act. Its limits: {refused.message}"
        ) from refused


def _well_formed(envelope: TurnEnvelope) -> bool:
    """Only what a root or a derived child can be: a tree, a depth within the
    bound, a concrete tool set below the root, and a deadline, if any, that
    can be compared with now."""
    return (
        bool(envelope.tree_id)
        and 0 <= envelope.depth <= MAX_DEPTH
        and (envelope.depth == 0 or envelope.tools is not None)
        and (envelope.deadline_at is None or envelope.deadline_at.tzinfo is not None)
    )


async def _restore(
    user_id: str, session_id: str, rows: "Iterable[PendingHumanReviewModel]"
) -> None:
    gate_prefix = review_store.node_id_for("")
    rows = [r for r in rows if r.node_exec_id.startswith(gate_prefix)]
    if not rows:
        return
    try:
        held, unreadable = await _read_held(session_id)
    except Exception:
        # Unread, a record that is there looks gone, and one that no longer
        # reads back would be rebuilt as runnable: rebuild nothing.
        logger.warning(
            f"Gate could not list held calls for session {session_id}", exc_info=True
        )
        return
    for row in rows:
        if row.node_exec_id in held:
            continue
        logger.warning(f"Held call {row.node_exec_id} restored from its card")
        call = _from_row(user_id, session_id, row)
        if row.node_exec_id in unreadable:
            # It may have bounded the call more than its card can say, so what
            # replaces it never runs. Records are never written over, so the
            # unreadable one goes first; a record written meanwhile stands.
            call.lost = True
            await _claim(session_id, row.node_exec_id)
        await remember(session_id, call)


def _from_row(
    user_id: str, session_id: str, row: "PendingHumanReviewModel"
) -> HeldCall:
    """The card stores a redacted, clipped copy; the review id is a hash of the
    exact arguments and the authority they were held under, so only a copy
    that hashes back to it may run. No card records a delegated turn's tree,
    so only a root's call can be rebuilt; the rest are lost."""
    tool_name = row.node_exec_id.removeprefix(review_store.node_id_for("")).rsplit(
        COPILOT_NODE_EXEC_ID_SEPARATOR, 1
    )[0]
    payload = row.payload if isinstance(row.payload, dict) else {}
    args = payload.get("arguments")
    if not isinstance(args, dict):
        args = {}
    intact = any(
        review_store.action_review_id(session_id, user_id, tool_name, args, root)
        == row.node_exec_id
        for root in _ROOTS
    )
    return HeldCall(
        review_id=row.node_exec_id,
        tool_name=tool_name,
        # The card keeps the call's id, so the late result still finds its row.
        tool_call_id=str(payload.get("tool_call_id") or ""),
        args=args if intact else {},
        held_at=row.created_at,
        lost=not intact,
    )


async def _held(session_id: str) -> dict[str, HeldCall]:
    try:
        held, _ = await _read_held(session_id)
    except Exception:
        logger.warning(
            f"Gate could not list held calls for session {session_id}", exc_info=True
        )
        return {}
    return held


async def _read_held(session_id: str) -> tuple[dict[str, HeldCall], set[str]]:
    """The held calls that read back, and the ids of those that do not. Raises
    when Redis cannot be read."""
    redis = await get_redis_async()
    raw = await redis.hgetall(_key(session_id))
    held: dict[str, HeldCall] = {}
    unreadable: set[str] = set()
    for key, value in raw.items():
        review_id = key.decode() if isinstance(key, bytes) else key
        stored = value.decode() if isinstance(value, bytes) else value
        try:
            held[review_id] = HeldCall.model_validate(JSONCryptor().decrypt(stored))
        except ValidationError:
            # The answer rebuilds it from its card, and it does not run.
            logger.warning(f"Gate could not read held call {review_id}")
            unreadable.add(review_id)
    return held, unreadable


async def _claim(session_id: str, review_id: str) -> bool:
    try:
        redis = await get_redis_async()
        return await redis.hdel(_key(session_id), review_id) == 1
    except Exception:
        logger.warning(f"Gate could not claim held call {review_id}", exc_info=True)
        return False


def _key(session_id: str) -> str:
    return f"{_KEY}{session_id}"
