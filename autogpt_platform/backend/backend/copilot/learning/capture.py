"""Turn-end capture hook for the chat services.

Cheap by design: one flag read and one upsert, fired in the background
after the reply has streamed. Recording a source never blocks a reply and
a failure here never changes the conversation.
"""

from __future__ import annotations

import asyncio
import logging

from backend.copilot.dream.scheduling import ensure_dream_system_scheduled
from backend.copilot.model import ChatMessage, ChatSession
from backend.data.skill_learning import LearningSourceRecord
from backend.util.feature_flag import Flag, is_feature_enabled

from .chat_source import record_chat_turn
from .nightly import run_skill_learning_pass

logger = logging.getLogger(__name__)

_background_tasks: set[asyncio.Task] = set()


def turn_with_initiating_user_row(
    session: ChatSession, generated: list[ChatMessage], user_message: str
) -> list[ChatMessage]:
    """The turn's messages including the persisted user row that started it.

    Both chat engines hand the capture hook only the rows generated during
    the turn: the SDK slices ``session.messages`` after the attempt started
    and the baseline passes its per-turn state, while the initiating user
    message was appended (and persisted) before either slice begins. The
    user's own confirmation or acceptance lives on that row, so it is
    located by content among the persisted user rows and prepended. A row
    without a sequence is not persisted and is never referenced.
    """
    if any(
        m.role == "user" and (m.content or "").strip() == user_message.strip()
        for m in generated
    ):
        return list(generated)
    present = {id(m) for m in generated}
    for message in reversed(session.messages):
        if id(message) in present:
            continue
        if message.role != "user" or message.sequence is None:
            continue
        if (message.content or "").strip() == user_message.strip():
            return [message, *generated]
    return list(generated)


async def capture_chat_turn(
    user_id: str,
    session: ChatSession,
    turn_messages: list[ChatMessage],
    user_message: str,
) -> LearningSourceRecord | None:
    """Record this turn as a learning source revision. Never raises.

    An explicit "learn this" request is labelled ``requested`` and runs the
    same validation right away instead of waiting for the night; the
    per-user cron is registered lazily the first time a source lands.
    """
    try:
        if not await is_feature_enabled(Flag.DREAM_SKILL_LEARNING_ENABLED, user_id):
            return None
        messages = turn_with_initiating_user_row(session, turn_messages, user_message)
        record = await record_chat_turn(user_id, session, messages, user_message)
        _spawn(
            ensure_dream_system_scheduled(user_id),
            f"skill-learning-register-{user_id[:12]}",
        )
        if record.origin == "requested" and record.has_unprocessed_revision:
            _spawn(
                run_skill_learning_pass(
                    user_id, source_ids=[record.id], trigger="requested"
                ),
                f"skill-learning-requested-{record.id[:12]}",
            )
        return record
    except Exception:
        logger.warning(
            "Skill learning capture failed for user %s — turn unaffected",
            user_id[:12],
            exc_info=True,
        )
        return None


def _spawn(coro, name: str) -> None:
    task = asyncio.create_task(coro, name=name)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
