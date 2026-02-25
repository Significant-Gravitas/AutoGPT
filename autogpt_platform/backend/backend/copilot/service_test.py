import asyncio
import logging
from os import getenv

import pytest

from . import service as chat_service
from .model import create_chat_session, get_chat_session, upsert_chat_session
from .response_model import StreamError, StreamTextDelta, StreamToolOutputAvailable
from .sdk import service as sdk_service
from .sdk.transcript import download_transcript

logger = logging.getLogger(__name__)


@pytest.mark.asyncio(loop_scope="session")
async def test_stream_chat_completion(setup_test_user, test_user_id):
    """
    Test the stream_chat_completion function.
    """
    api_key: str | None = getenv("OPEN_ROUTER_API_KEY")
    if not api_key:
        return pytest.skip("OPEN_ROUTER_API_KEY is not set, skipping test")

    session = await create_chat_session(test_user_id)

    has_errors = False
    assistant_message = ""
    async for chunk in chat_service.stream_chat_completion(
        session.session_id, "Hello, how are you?", user_id=session.user_id
    ):
        logger.info(chunk)
        if isinstance(chunk, StreamError):
            has_errors = True
        if isinstance(chunk, StreamTextDelta):
            assistant_message += chunk.delta

    # StreamFinish is published by mark_session_completed (processor layer),
    # not by the service. The generator completing means the stream ended.
    assert not has_errors, "Error occurred while streaming chat completion"
    assert assistant_message, "Assistant message is empty"


@pytest.mark.asyncio(loop_scope="session")
async def test_stream_chat_completion_with_tool_calls(setup_test_user, test_user_id):
    """
    Test the stream_chat_completion function.
    """
    api_key: str | None = getenv("OPEN_ROUTER_API_KEY")
    if not api_key:
        return pytest.skip("OPEN_ROUTER_API_KEY is not set, skipping test")

    session = await create_chat_session(test_user_id)
    session = await upsert_chat_session(session)

    has_errors = False
    had_tool_calls = False
    async for chunk in chat_service.stream_chat_completion(
        session.session_id,
        "Please find me an agent that can help me with my business. Use the query 'moneny printing agent'",
        user_id=session.user_id,
    ):
        logger.info(chunk)
        if isinstance(chunk, StreamError):
            has_errors = True
        if isinstance(chunk, StreamToolOutputAvailable):
            had_tool_calls = True

    assert not has_errors, "Error occurred while streaming chat completion"
    assert had_tool_calls, "Tool calls did not occur"
    session = await get_chat_session(session.session_id)
    assert session, "Session not found"
    assert session.usage, "Usage is empty"


@pytest.mark.asyncio(loop_scope="session")
async def test_sdk_resume_multi_turn(setup_test_user, test_user_id):
    """Test that the SDK --resume path captures and uses transcripts across turns.

    Turn 1: Send a message containing a unique keyword.
    Turn 2: Ask the model to recall that keyword — proving the transcript was
    persisted and restored via --resume.
    """
    api_key: str | None = getenv("OPEN_ROUTER_API_KEY")
    if not api_key:
        return pytest.skip("OPEN_ROUTER_API_KEY is not set, skipping test")

    from .config import ChatConfig

    cfg = ChatConfig()
    if not cfg.claude_agent_use_resume:
        return pytest.skip("CLAUDE_AGENT_USE_RESUME is not enabled, skipping test")

    session = await create_chat_session(test_user_id)
    session = await upsert_chat_session(session)

    # --- Turn 1: send a message with a unique keyword ---
    keyword = "ZEPHYR42"
    turn1_msg = (
        f"Please remember this special keyword: {keyword}. "
        "Just confirm you've noted it, keep your response brief."
    )
    turn1_text = ""
    turn1_errors: list[str] = []

    async for chunk in sdk_service.stream_chat_completion_sdk(
        session.session_id,
        turn1_msg,
        user_id=test_user_id,
    ):
        if isinstance(chunk, StreamTextDelta):
            turn1_text += chunk.delta
        elif isinstance(chunk, StreamError):
            turn1_errors.append(chunk.errorText)

    assert not turn1_errors, f"Turn 1 errors: {turn1_errors}"
    assert turn1_text, "Turn 1 produced no text"

    # Wait for background upload task to complete (retry up to 5s).
    # The CLI may not produce a usable transcript for very short
    # conversations (only metadata entries) — this is environment-dependent
    # (CLI version, platform).  When that happens, multi-turn still works
    # via conversation compression (non-resume path), but we can't test
    # the --resume round-trip.
    transcript = None
    for _ in range(10):
        await asyncio.sleep(0.5)
        transcript = await download_transcript(test_user_id, session.session_id)
        if transcript:
            break
    if not transcript:
        return pytest.skip(
            "CLI did not produce a usable transcript — "
            "cannot test --resume round-trip in this environment"
        )
    logger.info(f"Turn 1 transcript uploaded: {len(transcript.content)} bytes")

    # Reload session for turn 2
    session = await get_chat_session(session.session_id, test_user_id)
    assert session, "Session not found after turn 1"

    # --- Turn 2: ask model to recall the keyword ---
    turn2_msg = "What was the special keyword I asked you to remember?"
    turn2_text = ""
    turn2_errors: list[str] = []

    async for chunk in sdk_service.stream_chat_completion_sdk(
        session.session_id,
        turn2_msg,
        user_id=test_user_id,
        session=session,
    ):
        if isinstance(chunk, StreamTextDelta):
            turn2_text += chunk.delta
        elif isinstance(chunk, StreamError):
            turn2_errors.append(chunk.errorText)

    assert not turn2_errors, f"Turn 2 errors: {turn2_errors}"
    assert turn2_text, "Turn 2 produced no text"
    assert keyword in turn2_text, (
        f"Model did not recall keyword '{keyword}' in turn 2. "
        f"Response: {turn2_text[:200]}"
    )
    logger.info(f"Turn 2 recalled keyword successfully: {turn2_text[:100]}")
