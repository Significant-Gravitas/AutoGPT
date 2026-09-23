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
from typing import TYPE_CHECKING, Any, Callable

from prisma.enums import ReviewStatus
from pydantic import BaseModel, Field

from backend.copilot.model import ChatSession
from backend.copilot.pending_messages import PendingMessage
from backend.data.db_accessors import chat_db, review_db
from backend.data.redis_client import get_redis_async

from . import chat_rules
from . import review as review_store

if TYPE_CHECKING:
    from backend.copilot.tools.base import BaseTool

logger = logging.getLogger(__name__)

# Outlives any card a user will still answer; the approval's own hour runs
# from the click, not from here.
_TTL_SECONDS = 30 * 24 * 60 * 60
_KEY = "copilot:gate:held:"
# Above the largest result either engine hands the model (the baseline's
# 100,000-character output cap plus the wrapper), so nothing is cut here.
_MAX_RESULT_CHARS = 120_000

WAKE_MESSAGE = "I answered an action that was waiting for my approval."


class HeldResult(PendingMessage):
    """A late result is capped by the engine, as a direct one is, not by the
    32,000-character limit on a typed follow-up."""

    content: str = Field(min_length=1, max_length=_MAX_RESULT_CHARS)


class HeldCall(BaseModel):
    review_id: str
    tool_name: str
    tool_call_id: str
    args: dict[str, Any]
    held_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


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
    exec_id = review_store.session_exec_id(session_id)
    return sorted(
        (
            held[review_id]
            for review_id, row in rows.items()
            if row.graph_exec_id == exec_id and row.status != ReviewStatus.WAITING
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
            # Put it back for the next turn; the gate's consume still stops a
            # second run if the call got as far as running.
            logger.warning(f"Held call {call.review_id} not delivered", exc_info=True)
            await remember(session.session_id, call)
    return delivered


async def wake(user_id: str, session_id: str) -> None:
    """Start a turn to carry answered cards, if the chat is idle.

    Best effort: a running turn's end calls this again, and any later turn
    folds the results in anyway.
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

    output = cap(await _outcome(user_id, session, call, get_tool(call.tool_name)))
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
            }
        },
    )


async def _outcome(
    user_id: str, session: ChatSession, call: HeldCall, tool: "BaseTool | None"
) -> str:
    rows = await review_db().get_reviews_by_node_exec_ids([call.review_id], user_id)
    row = rows.get(call.review_id)
    if row is None or row.status == ReviewStatus.WAITING:
        return "Nothing ran: this card is no longer open."
    if row.status == ReviewStatus.REJECTED:
        await review_store.consume(call.review_id, user_id)
        await chat_rules.set_ask(session.session_id, call.tool_name)
        return (
            "Nothing ran: the user declined this action. Do not retry it or "
            "reach the same effect another way."
        )
    approved_at = row.reviewed_at or row.updated_at or row.created_at
    if datetime.now(UTC) - approved_at > review_store.APPROVAL_TTL:
        await review_store.consume(call.review_id, user_id)
        return (
            "Nothing ran: the approval expired an hour after it was given. "
            "Propose the call again if it is still needed."
        )
    if tool is None:
        await review_store.consume(call.review_id, user_id)
        return "Nothing ran: this tool no longer exists."
    # The gate finds the approval for exactly these arguments and spends it.
    result = await tool.execute(user_id, session, call.tool_call_id, **call.args)
    if isinstance(result.output, str):
        return result.output
    return json.dumps(result.output, default=str)


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
