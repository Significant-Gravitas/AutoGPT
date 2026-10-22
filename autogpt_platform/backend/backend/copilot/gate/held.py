"""Held calls: what a parked call was, so its card's answer can finish it later.

A held call returns to the model at once, so the answer arrives after the turn
that made the call — often after it ended. The call is kept here exactly as
the model made it, and :func:`resolve_answered` runs it (or refuses it) at the
start of the chat's next turn, inside the engine, where the turn's sandbox and
tool bounds exist. Its result reaches the model as a user row naming the
original call. :func:`wake` starts that turn when none is running; a running
turn's end wakes it instead, once that turn has left its executor
(``stream_registry.mark_session_completed``). An answer on an idle chat waits
for the chat's last card, so a burst of answers starts one turn.
"""

import json
import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Iterable, Literal

from prisma.enums import ReviewStatus
from pydantic import BaseModel, Field, ValidationError

from backend.copilot import woken_turns
from backend.copilot.constants import (
    COPILOT_NODE_EXEC_ID_SEPARATOR,
    COPILOT_STEP_CARD_PREFIX,
)
from backend.copilot.google_data_redaction import (
    is_google_call,
    is_google_output,
    remember_google_output,
)
from backend.copilot.model import CHAT_STATUS_RUNNING, ChatSession
from backend.copilot.pending_messages import PendingMessage
from backend.copilot.response_model import StreamStatus
from backend.copilot.tool_image import HeldImage, ToolImage, parse_held_image
from backend.copilot.tree import TreeRefusal, TurnEnvelope
from backend.copilot.trial_cost_context import TrialCostContext, restore_cost_context
from backend.data.db_accessors import chat_db, review_db
from backend.data.redis_client import get_redis_async
from backend.util.encryption import JSONCryptor

from . import chat_rules
from . import review as review_store
from .headline import headline_for
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


def is_answer_row(message: "ChatMessage") -> bool:
    """A user row the gate wrote for an answered card, not something the user typed."""
    return bool(message.metadata) and (
        _WAKE_KEY in message.metadata or _RESULT_KEY in message.metadata
    )


class HeldResult(PendingMessage):
    """A late result is capped by the engine, as a direct one is, not by the
    32,000-character limit on a typed follow-up."""

    content: str = Field(min_length=1, max_length=_MAX_RESULT_CHARS)

    @staticmethod
    def carried_by(data: dict[str, Any]) -> bool:
        """Whether a buffered message's data is a held call's result."""
        return _RESULT_KEY in (data.get("metadata") or {})


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
    cost_context: TrialCostContext | None = None


async def remember(session_id: str, call: HeldCall) -> bool:
    """False means the call cannot be finished on approval, so it must not park."""
    try:
        redis = await get_redis_async()
        async with redis.pipeline(transaction=True) as pipe:
            # Encrypted: the arguments can hold a credential, and must stay exact
            # to match the approval, so they cannot be redacted like the card's.
            pipe.hset(
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
    return [
        held[review_id] for review_id, _ in await _answered(user_id, session_id, held)
    ]


async def answered_cards(
    user_id: str, session_id: str
) -> "list[PendingHumanReviewModel]":
    """The answered cards whose held calls have not run yet, oldest first."""
    held = await _held(session_id)
    return [row for _, row in await _answered(user_id, session_id, held)]


async def _answered(
    user_id: str, session_id: str, held: dict[str, HeldCall]
) -> "list[tuple[str, PendingHumanReviewModel]]":
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
            (review_id, row)
            for review_id, row in rows.items()
            if row.session_id == session_id and row.status != ReviewStatus.WAITING
        ),
        key=lambda answered: held[answered[0]].held_at,
    )


async def resolve_answered(
    user_id: str | None,
    session: ChatSession,
    deliver: Callable[[PendingMessage], None],
    cap: Callable[[str], str] = lambda text: text,
) -> AsyncIterator[StreamStatus]:
    """Run every answered held call once and ``deliver`` its result as a user row.

    Yields a status before an approved call runs, which the engine streams:
    nothing else reaches the chat while it does. Call only once the turn's
    execution context is set. ``cap`` is the engine's own last step on a
    direct tool result, so a late one reads the same. The HDEL claims the
    call, so two turns racing over one card cannot both run it; the gate's own
    consume stays the second lock behind it.
    """
    if not user_id:
        return
    for call in await answered(user_id, session.session_id):
        try:
            rows = await review_db().get_reviews_by_node_exec_ids(
                [call.review_id], user_id
            )
        except Exception:
            logger.warning(f"Held call {call.review_id} unreadable", exc_info=True)
            continue
        row = rows.get(call.review_id)
        tool = _due(call, row, user_id)
        # Before the claim: a stream closed at this yield leaves the call held.
        if tool:
            yield StreamStatus(
                message=f"Running the action you approved: {_headline(call, row)}"
            )
        if not await _claim(session.session_id, call.review_id):
            continue
        try:
            await _resolve(user_id, session, call, row, tool, deliver, cap)
        except Exception:
            logger.warning(f"Held call {call.review_id} not delivered", exc_info=True)
            for result in await _recover(user_id, session.session_id, call):
                deliver(result)


async def wake(
    user_id: str,
    session_id: str,
    answered_rows: "Iterable[PendingHumanReviewModel]" = (),
    *,
    until_last: bool = False,
) -> bool:
    """Start a turn to carry answered cards, if the chat is idle.

    Best effort: a running turn's end calls this again, and any later turn
    folds the results in anyway. ``answered_rows`` are the cards just
    answered; one whose held call is gone from Redis is restored from its row.
    ``until_last`` is an answer's wake: on an idle chat it starts nothing while
    another card still waits, and returns False.
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
    from backend.copilot.session_permissions import resolve_session_permissions
    from backend.copilot.turn_queue import (
        UNRECORDED_WAKE,
        WAKE_LATER,
        InflightCapExceeded,
        is_users_own_chat,
        post_refusal,
        try_enqueue_turn,
        wakes_sub_work,
    )

    try:
        await _restore(user_id, session_id, answered_rows)
        calls = await answered(user_id, session_id)
        if not calls:
            return True
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
            return True
        waiting = await _waiting(user_id, session_id)
        # A running turn's end carries the answers whatever still waits.
        if until_last and waiting and info.chat_status != CHAT_STATUS_RUNNING:
            logger.info(f"Held calls in {session_id} wait for {waiting} more answers")
            return False
        # A turn now would run over the question; the answer's turn runs these
        # cards first.
        if info.metadata.pending_question is not None:
            logger.info(f"Held calls in {session_id} wait for its open question")
            return True
        message = _wake_message(waiting)
        refusal_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{wake_id}:refused"))
        latest = max(calls, key=lambda c: c.held_at)
        envelope = latest.envelope
        if envelope is None and not is_users_own_chat(info):
            # Deriving one from the turn that just ended would run the approved
            # action under that turn's limits, not the ones it was held under.
            await post_refusal(session_id, UNRECORDED_WAKE, message_id=refusal_id)
            return True
        if latest.cost_context is None or latest.cost_context.user_id != user_id:
            await post_refusal(session_id, _RESEND, message_id=refusal_id)
            return True
        permissions = resolve_session_permissions(info)
        metadata = {_WAKE_KEY: True}
        capacity = get_delegated_turn_limit() if wakes_sub_work(info) else None
        with restore_cost_context(user_id, latest.cost_context):
            try:
                async with acquire_turn_slot(
                    user_id, session_id, capacity=capacity
                ) as slot:
                    # Not admitted: a turn is already running, and its end wakes us.
                    if not slot.admitted:
                        return True
                    if (
                        await append_and_save_message(
                            session_id,
                            ChatMessage(
                                id=wake_id,
                                role="user",
                                content=message,
                                metadata=metadata,
                            ),
                        )
                        is None
                    ):
                        return True
                    turn_id = str(uuid.uuid4())
                    await dispatch_turn(
                        slot,
                        session_id=session_id,
                        user_id=user_id,
                        turn_id=turn_id,
                        message=message,
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
                    message=message,
                    message_id=wake_id,
                    message_metadata=metadata,
                    llm_auth_provider=info.metadata.llm_auth_provider,
                    llm_credential_id=info.metadata.llm_credential_id,
                    permissions=(
                        permissions.model_dump(exclude_none=True)
                        if permissions
                        else None
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
    return True


def _wake_message(waiting: int) -> str:
    """The wake row, which tells the model how many of its cards are unanswered."""
    if not waiting:
        return WAKE_MESSAGE
    more = "1 more still waits" if waiting == 1 else f"{waiting} more still wait"
    return f"{WAKE_MESSAGE} {more} for my answer."


async def _waiting(user_id: str, session_id: str) -> int:
    """The chat's held calls still waiting for an answer."""
    try:
        rows = await review_db().get_pending_reviews_for_chat_session(
            session_id, user_id
        )
    except Exception:
        # Unreadable counts as none, so an answer still wakes its turn.
        logger.warning(f"Waiting cards in {session_id} unreadable", exc_info=True)
        return 0
    gate_prefix = review_store.node_id_for("")
    # A workflow step's card resumes its run, not this chat, and leaves by the
    # run page or a stopped run, neither of which wakes the chat.
    return sum(
        1
        for row in rows
        if row.node_exec_id.startswith(gate_prefix)
        and not row.node_exec_id.startswith(COPILOT_STEP_CARD_PREFIX)
    )


async def _resolve(
    user_id: str,
    session: ChatSession,
    call: HeldCall,
    row: "PendingHumanReviewModel | None",
    tool: "BaseTool | None",
    deliver: Callable[[PendingMessage], None],
    cap: Callable[[str], str],
) -> None:
    if call.cost_context is None or call.cost_context.user_id != user_id:
        await review_store.consume(call.review_id, user_id)
        deliver(_result_row(call, _RESEND, "closed"))
        return
    with restore_cost_context(user_id, call.cost_context):
        if tool:
            outcome, output = await _run(user_id, session, call, tool)
        else:
            outcome, output = await _settled(user_id, session, call, row)
    shown = _card_copy(row)
    google_data = is_google_call(call.args) or is_google_output(output)
    image_result = parse_held_image(output) if outcome == "approved" else None
    if image_result is not None:
        output = image_result.text
    try:
        deliver(
            _result_row(
                call,
                cap(output),
                outcome,
                shown,
                google_data,
                image=image_result.image if image_result else None,
            )
        )
    except Exception:
        # The outcome is known and may be a refusal; only the engine's cut
        # failed, so deliver it trimmed rather than let recovery guess.
        logger.warning(f"Could not cap held result {call.review_id}", exc_info=True)
        deliver(
            _result_row(
                call,
                output[: _MAX_RESULT_CHARS // 2],
                outcome,
                shown,
                google_data,
                image=image_result.image if image_result else None,
            )
        )


async def _recover(
    user_id: str, session_id: str, call: HeldCall
) -> list[PendingMessage]:
    """``_resolve`` failed. A card still open goes back for the next turn; a
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
    google_data: bool = False,
    image: ToolImage | None = None,
) -> PendingMessage:
    content = (
        f'<held_call_result tool="{call.tool_name}" '
        f'tool_call_id="{call.tool_call_id}" review_id="{call.review_id}">\n'
        f"{output}\n</held_call_result>"
    )
    # A user row, not a tool result, so the traces cannot tie it to a Google
    # call, and its cut may no longer parse: remember it as delivered.
    if google_data:
        remember_google_output(content)
    return HeldResult(
        model_image=image,
        content=content,
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


def _due(
    call: HeldCall, row: "PendingHumanReviewModel | None", user_id: str
) -> "BaseTool | None":
    """The tool an answered card's approved call runs with; None when it runs
    nothing, which :func:`_resolve` then says."""
    from backend.copilot.tools import get_tool

    from .reads import is_held_read

    if (
        row is None
        or row.status != ReviewStatus.APPROVED
        or is_held_read(call.review_id)
        or _expired(row)
        or call.lost
        or call.cost_context is None
        or call.cost_context.user_id != user_id
    ):
        return None
    return get_tool(call.tool_name)


async def _settled(
    user_id: str,
    session: ChatSession,
    call: HeldCall,
    row: "PendingHumanReviewModel | None",
) -> tuple[Outcome, str]:
    """What an answered card delivers when its call does not run."""
    # Deferred: reads imports this package's __init__, which imports this module.
    from .reads import answered_read, is_held_read

    if row is None or row.status == ReviewStatus.WAITING:
        return "closed", "Nothing ran: this card is no longer open."
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
    await review_store.consume(call.review_id, user_id)
    if _expired(row):
        return "expired", (
            "Nothing ran: the approval expired an hour after it was given. "
            "Propose the call again if it is still needed."
        )
    if call.lost:
        return "closed", _RESEND
    return "closed", "Nothing ran: this tool no longer exists."


def _expired(row: "PendingHumanReviewModel") -> bool:
    approved_at = row.reviewed_at or row.updated_at or row.created_at
    return datetime.now(UTC) - approved_at > review_store.APPROVAL_TTL


def _headline(call: HeldCall, row: "PendingHumanReviewModel | None") -> str:
    """The card's own headline, which the user approved."""
    payload = row.payload if row and isinstance(row.payload, dict) else {}
    if "headline" in payload:
        return review_store.payload_headline(payload)
    return headline_for(call.tool_name, call.args).text


async def _run(
    user_id: str, session: ChatSession, call: HeldCall, tool: "BaseTool"
) -> tuple[Outcome, str]:
    # The gate finds the approval for exactly these arguments and spends it.
    result = await tool.execute(user_id, session, call.tool_call_id, **call.args)
    # With the flag switched off since, the gate ran it without spending the
    # approval; spend it here so no later identical call rides on it.
    await review_store.consume(call.review_id, user_id)
    if result.image is not None:
        text = (
            result.output
            if isinstance(result.output, str)
            else json.dumps(result.output, default=str)
        )
        return "approved", HeldImage(text=text, image=result.image).model_dump_json()
    if isinstance(result.output, str):
        return "approved", result.output
    return "approved", json.dumps(result.output, default=str)


async def _restore(
    user_id: str, session_id: str, rows: "Iterable[PendingHumanReviewModel]"
) -> None:
    gate_prefix = review_store.node_id_for("")
    rows = [r for r in rows if r.node_exec_id.startswith(gate_prefix)]
    if not rows:
        return
    held = await _held(session_id)
    for row in rows:
        if row.node_exec_id not in held:
            logger.warning(f"Held call {row.node_exec_id} restored from its card")
            await remember(session_id, _from_row(user_id, session_id, row))


def _from_row(
    user_id: str, session_id: str, row: "PendingHumanReviewModel"
) -> HeldCall:
    """The card stores a redacted, clipped copy; the review id is a hash of the
    exact arguments, so only a copy that hashes back to it may run."""
    tool_name = row.node_exec_id.removeprefix(review_store.node_id_for("")).rsplit(
        COPILOT_NODE_EXEC_ID_SEPARATOR, 1
    )[0]
    payload = row.payload if isinstance(row.payload, dict) else {}
    args = payload.get("arguments")
    if not isinstance(args, dict):
        args = {}
    intact = (
        review_store.review_id_for(session_id, user_id, tool_name, args)
        == row.node_exec_id
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
        redis = await get_redis_async()
        raw = await redis.hgetall(_key(session_id))
    except Exception:
        logger.warning(
            f"Gate could not list held calls for session {session_id}", exc_info=True
        )
        return {}
    held: dict[str, HeldCall] = {}
    for key, value in raw.items():
        review_id = key.decode() if isinstance(key, bytes) else key
        stored = value.decode() if isinstance(value, bytes) else value
        try:
            held[review_id] = HeldCall.model_validate(JSONCryptor().decrypt(stored))
        except ValidationError:
            # Unreadable is the same as gone: the answer restores it from its card.
            logger.warning(f"Gate could not read held call {review_id}")
    return held


async def _claim(session_id: str, review_id: str) -> bool:
    try:
        redis = await get_redis_async()
        return await redis.hdel(_key(session_id), review_id) == 1
    except Exception:
        logger.warning(f"Gate could not claim held call {review_id}", exc_info=True)
        return False


def _key(session_id: str) -> str:
    return f"{_KEY}{session_id}"
