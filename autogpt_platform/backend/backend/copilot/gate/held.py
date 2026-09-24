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
from pydantic import BaseModel, Field

from backend.copilot.constants import COPILOT_NODE_EXEC_ID_SEPARATOR
from backend.copilot.model import ChatSession
from backend.copilot.pending_messages import PendingMessage
from backend.data.db_accessors import chat_db, review_db
from backend.data.redis_client import get_redis_async

from . import chat_rules
from . import review as review_store

if TYPE_CHECKING:
    from backend.api.features.graph_executions.review.model import (
        PendingHumanReviewModel,
    )
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
_RESEND = (
    "Nothing ran: the approved action's details were lost before it could run. "
    "Tell the user, and ask them to send the request again if it is still needed."
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


async def remember(session_id: str, call: HeldCall) -> bool:
    """False means the call cannot be finished on approval, so it must not park."""
    try:
        redis = await get_redis_async()
        async with redis.pipeline(transaction=True) as pipe:
            pipe.hset(_key(session_id), call.review_id, call.model_dump_json())
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
        get_inflight_turn_limit,
    )
    from backend.copilot.executor.utils import dispatch_turn
    from backend.copilot.model import ChatMessage, append_and_save_message
    from backend.copilot.session_permissions import resolve_session_permissions
    from backend.copilot.turn_queue import InflightCapExceeded, try_enqueue_turn

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
        permissions = resolve_session_permissions(info)
        metadata = {"held_calls_answered": True}
        try:
            async with acquire_turn_slot(user_id, session_id) as slot:
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
                await dispatch_turn(
                    slot,
                    session_id=session_id,
                    user_id=user_id,
                    turn_id=str(uuid.uuid4()),
                    message=WAKE_MESSAGE,
                    organization_id=info.organization_id,
                    team_id=info.team_id,
                    llm_auth_provider=info.metadata.llm_auth_provider,
                    llm_credential_id=info.metadata.llm_credential_id,
                    permissions=permissions,
                    message_metadata=metadata,
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
            )
    except InflightCapExceeded:
        logger.info(f"Held calls in {session_id} wait for the user's next turn")
    except Exception:
        logger.warning(f"Could not wake session {session_id}", exc_info=True)


async def _deliver(
    user_id: str, session: ChatSession, call: HeldCall, cap: Callable[[str], str]
) -> PendingMessage:
    from backend.copilot.tools import get_tool

    outcome, output = await _outcome(user_id, session, call, get_tool(call.tool_name))
    try:
        return _result_row(call, cap(output), outcome)
    except Exception:
        # The outcome is known and may be a refusal; only the engine's cut
        # failed, so deliver it trimmed rather than let recovery guess.
        logger.warning(f"Could not cap held result {call.review_id}", exc_info=True)
        return _result_row(call, output[: _MAX_RESULT_CHARS // 2], outcome)


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


def _result_row(call: HeldCall, output: str, outcome: Outcome) -> PendingMessage:
    return HeldResult(
        content=(
            f'<held_call_result tool="{call.tool_name}" '
            f'tool_call_id="{call.tool_call_id}" review_id="{call.review_id}">\n'
            f"{output}\n</held_call_result>"
        ),
        metadata={
            "held_call": {
                "review_id": call.review_id,
                "tool_name": call.tool_name,
                "tool_call_id": call.tool_call_id,
                # The chain row shows the answer without parsing the text.
                "outcome": outcome,
            }
        },
    )


async def _outcome(
    user_id: str, session: ChatSession, call: HeldCall, tool: "BaseTool | None"
) -> tuple[Outcome, str]:
    rows = await review_db().get_reviews_by_node_exec_ids([call.review_id], user_id)
    row = rows.get(call.review_id)
    if row is None or row.status == ReviewStatus.WAITING:
        return "closed", "Nothing ran: this card is no longer open."
    # Deferred: reads imports this package's __init__, which imports this module.
    from .reads import answered_read, is_held_read

    if is_held_read(call.review_id):
        return await answered_read(user_id, row)
    if row.status == ReviewStatus.REJECTED:
        await review_store.consume(call.review_id, user_id)
        await chat_rules.set_ask(session.session_id, call.rule_key or call.tool_name)
        return "rejected", (
            "Nothing ran: the user declined this action. Do not retry it or "
            "reach the same effect another way."
        )
    approved_at = row.reviewed_at or row.updated_at or row.created_at
    if datetime.now(UTC) - approved_at > review_store.APPROVAL_TTL:
        await review_store.consume(call.review_id, user_id)
        return "expired", (
            "Nothing ran: the approval expired an hour after it was given. "
            "Propose the call again if it is still needed."
        )
    if call.lost:
        await review_store.consume(call.review_id, user_id)
        return "closed", _RESEND
    if tool is None:
        await review_store.consume(call.review_id, user_id)
        return "closed", "Nothing ran: this tool no longer exists."
    # The gate finds the approval for exactly these arguments and spends it.
    result = await tool.execute(user_id, session, call.tool_call_id, **call.args)
    # With the flag switched off since, the gate ran it without spending the
    # approval; spend it here so no later identical call rides on it.
    await review_store.consume(call.review_id, user_id)
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
        held[review_id] = HeldCall.model_validate_json(value)
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
