"""Dummy SDK service for testing copilot streaming.

Returns mock streaming responses without calling Claude Agent SDK.
Enable via COPILOT_TEST_MODE=true environment variable.

WARNING: This is for testing only. Do not use in production.
"""

import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator

from ..model import ChatSession
from ..response_model import StreamBaseResponse, StreamStart, StreamTextDelta

logger = logging.getLogger(__name__)


async def stream_chat_completion_dummy(
    session_id: str,
    message: str | None = None,
    tool_call_response: str | None = None,
    is_user_message: bool = True,
    user_id: str | None = None,
    retry_count: int = 0,
    session: ChatSession | None = None,
    context: dict[str, str] | None = None,
) -> AsyncGenerator[StreamBaseResponse, None]:
    """Stream dummy chat completion for testing.

    Returns a simple streaming response with text deltas to test:
    - Streaming infrastructure works
    - No timeout occurs
    - Text arrives in chunks
    - StreamFinish is sent by mark_session_completed
    """
    logger.warning(
        f"[TEST MODE] Using dummy copilot streaming for session {session_id}"
    )

    message_id = str(uuid.uuid4())
    text_block_id = str(uuid.uuid4())

    # Start the stream
    yield StreamStart(messageId=message_id, sessionId=session_id)

    # Simulate streaming text response with delays
    dummy_response = "I counted: 1... 2... 3. All done!"
    words = dummy_response.split()

    for i, word in enumerate(words):
        # Add space except for last word
        text = word if i == len(words) - 1 else f"{word} "
        yield StreamTextDelta(id=text_block_id, delta=text)
        # Small delay to simulate real streaming
        await asyncio.sleep(0.1)
