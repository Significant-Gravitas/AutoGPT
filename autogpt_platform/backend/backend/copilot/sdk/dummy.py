"""Dummy SDK service for testing copilot streaming.

Returns mock streaming responses without calling Claude Agent SDK.
Enable via CHAT_TEST_MODE=true in .env (ChatConfig.test_mode).

WARNING: This is for testing only. Do not use in production.

Magic keywords (case-insensitive, anywhere in message):
    __test_transient_error__   — Simulate a transient Anthropic API error
                                 (ECONNRESET).  Streams partial text, then
                                 yields StreamError with retryable prefix.
    __test_fatal_error__       — Simulate a non-retryable SDK error.
    __test_slow_response__     — Simulate a slow response (2s per word).
    (no keyword)               — Normal dummy response.
"""

import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from ..constants import (
    COPILOT_ERROR_PREFIX,
    COPILOT_RETRYABLE_ERROR_PREFIX,
    FRIENDLY_TRANSIENT_MSG,
)
from ..model import ChatMessage, ChatSession, get_chat_session, upsert_chat_session
from ..response_model import (
    StreamBaseResponse,
    StreamError,
    StreamFinish,
    StreamFinishStep,
    StreamStart,
    StreamStartStep,
    StreamTextDelta,
    StreamTextEnd,
    StreamTextStart,
)

logger = logging.getLogger(__name__)


async def _safe_upsert(session: ChatSession) -> None:
    """Best-effort session persist — skip silently if DB is unavailable."""
    try:
        await upsert_chat_session(session)
    except Exception:
        logger.debug("[TEST MODE] Could not persist session (DB unavailable)")


def _has_keyword(message: str | None, keyword: str) -> bool:
    return keyword in (message or "").lower()


async def stream_chat_completion_dummy(
    session_id: str,
    message: str | None = None,
    tool_call_response: str | None = None,
    is_user_message: bool = True,
    user_id: str | None = None,
    retry_count: int = 0,
    session: ChatSession | None = None,
    context: dict[str, str] | None = None,
    **_kwargs: Any,
) -> AsyncGenerator[StreamBaseResponse, None]:
    """Stream dummy chat completion for testing.

    Returns a simple streaming response with text deltas to test:
    - Streaming infrastructure works
    - No timeout occurs
    - Text arrives in chunks
    - StreamFinish is sent by mark_session_completed

    See module docstring for magic keywords that trigger error scenarios.
    """
    logger.warning(
        f"[TEST MODE] Using dummy copilot streaming for session {session_id}"
    )

    # Load session from DB (matches SDK service behaviour) so error markers
    # and the assistant reply are persisted and survive page refresh.
    # Best-effort: skip if DB is unavailable (e.g. unit tests).
    if session is None:
        try:
            session = await get_chat_session(session_id, user_id)
        except Exception:
            logger.debug("[TEST MODE] Could not load session (DB unavailable)")
            session = None

    message_id = str(uuid.uuid4())
    text_block_id = str(uuid.uuid4())

    # Start the stream (matches baseline: StreamStart → StreamStartStep)
    yield StreamStart(messageId=message_id, sessionId=session_id)
    yield StreamStartStep()

    # --- Magic keyword: transient error (retryable) -------------------------
    if _has_keyword(message, "__test_transient_error__"):
        # Stream some partial text first (simulates mid-stream failure)
        yield StreamTextStart(id=text_block_id)
        for word in ["Working", "on", "it..."]:
            yield StreamTextDelta(id=text_block_id, delta=f"{word} ")
            await asyncio.sleep(0.1)
        yield StreamTextEnd(id=text_block_id)
        yield StreamFinishStep()
        # Persist retryable marker so "Try Again" button shows after refresh
        if session:
            session.messages.append(
                ChatMessage(
                    role="assistant",
                    content=f"{COPILOT_RETRYABLE_ERROR_PREFIX} {FRIENDLY_TRANSIENT_MSG}",
                )
            )
            await _safe_upsert(session)
        yield StreamError(
            errorText=FRIENDLY_TRANSIENT_MSG,
            code="transient_api_error",
        )
        return

    # --- Magic keyword: fatal error (non-retryable) -------------------------
    if _has_keyword(message, "__test_fatal_error__"):
        yield StreamFinishStep()
        error_msg = "Internal SDK error: model refused to respond"
        # Persist non-retryable error marker
        if session:
            session.messages.append(
                ChatMessage(
                    role="assistant",
                    content=f"{COPILOT_ERROR_PREFIX} {error_msg}",
                )
            )
            await _safe_upsert(session)
        yield StreamError(errorText=error_msg, code="sdk_error")
        return

    # --- Magic keyword: slow response ---------------------------------------
    delay = 2.0 if _has_keyword(message, "__test_slow_response__") else 0.1

    # --- Normal dummy response ----------------------------------------------
    dummy_response = "I counted: 1... 2... 3. All done!"
    words = dummy_response.split()

    yield StreamTextStart(id=text_block_id)
    for i, word in enumerate(words):
        # Add space except for last word
        text = word if i == len(words) - 1 else f"{word} "
        yield StreamTextDelta(id=text_block_id, delta=text)
        await asyncio.sleep(delay)
    yield StreamTextEnd(id=text_block_id)

    # Persist the assistant reply so it survives page refresh
    if session:
        session.messages.append(ChatMessage(role="assistant", content=dummy_response))
        await _safe_upsert(session)

    yield StreamFinishStep()
    yield StreamFinish()
