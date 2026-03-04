import logging
from os import getenv

import pytest

from backend.copilot.baseline import stream_chat_completion_baseline
from backend.copilot.model import (
    create_chat_session,
    get_chat_session,
    upsert_chat_session,
)
from backend.copilot.response_model import (
    StreamError,
    StreamFinish,
    StreamStart,
    StreamTextDelta,
)

logger = logging.getLogger(__name__)


@pytest.mark.asyncio(loop_scope="session")
async def test_baseline_multi_turn(setup_test_user, test_user_id):
    """Test that the baseline LLM path streams responses and maintains history.

    Turn 1: Send a message with a unique keyword.
    Turn 2: Ask the model to recall the keyword — proving conversation history
    is correctly passed to the single-call LLM.
    """
    api_key: str | None = getenv("OPEN_ROUTER_API_KEY")
    if not api_key:
        return pytest.skip("OPEN_ROUTER_API_KEY is not set, skipping test")

    session = await create_chat_session(test_user_id)
    session = await upsert_chat_session(session)

    # --- Turn 1: send a message with a unique keyword ---
    keyword = "QUASAR99"
    turn1_msg = (
        f"Please remember this special keyword: {keyword}. "
        "Just confirm you've noted it, keep your response brief."
    )
    turn1_text = ""
    turn1_errors: list[str] = []
    got_start = False
    got_finish = False

    async for chunk in stream_chat_completion_baseline(
        session.session_id,
        turn1_msg,
        user_id=test_user_id,
    ):
        if isinstance(chunk, StreamStart):
            got_start = True
        elif isinstance(chunk, StreamTextDelta):
            turn1_text += chunk.delta
        elif isinstance(chunk, StreamError):
            turn1_errors.append(chunk.errorText)
        elif isinstance(chunk, StreamFinish):
            got_finish = True

    assert got_start, "Turn 1 did not yield StreamStart"
    assert got_finish, "Turn 1 did not yield StreamFinish"
    assert not turn1_errors, f"Turn 1 errors: {turn1_errors}"
    assert turn1_text, "Turn 1 produced no text"
    logger.info(f"Turn 1 response: {turn1_text[:100]}")

    # Reload session for turn 2
    session = await get_chat_session(session.session_id, test_user_id)
    assert session, "Session not found after turn 1"

    # Verify messages were persisted (user + assistant)
    assert (
        len(session.messages) >= 2
    ), f"Expected at least 2 messages after turn 1, got {len(session.messages)}"

    # --- Turn 2: ask model to recall the keyword ---
    turn2_msg = "What was the special keyword I asked you to remember?"
    turn2_text = ""
    turn2_errors: list[str] = []

    async for chunk in stream_chat_completion_baseline(
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
