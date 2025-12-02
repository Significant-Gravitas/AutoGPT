import logging
from os import getenv

import pytest

import backend.server.v2.chat.service as chat_service
from backend.server.v2.chat.response_model import (
    StreamEnd,
    StreamError,
    StreamTextChunk,
    StreamToolExecutionResult,
)

logger = logging.getLogger(__name__)


@pytest.mark.asyncio(loop_scope="session")
async def test_stream_chat_completion():
    """
    Test the stream_chat_completion function.
    """
    api_key: str | None = getenv("OPEN_ROUTER_API_KEY")
    if not api_key:
        return pytest.skip("OPEN_ROUTER_API_KEY is not set, skipping test")

    session = await chat_service.create_chat_session()

    has_errors = False
    has_ended = False
    assistant_message = ""
    async for chunk in chat_service.stream_chat_completion(
        session.session_id, "Hello, how are you?", user_id=session.user_id
    ):
        logger.info(chunk)
        if isinstance(chunk, StreamError):
            has_errors = True
        if isinstance(chunk, StreamTextChunk):
            assistant_message += chunk.content
        if isinstance(chunk, StreamEnd):
            has_ended = True

    assert has_ended, "Chat completion did not end"
    assert not has_errors, "Error occurred while streaming chat completion"
    assert assistant_message, "Assistant message is empty"


@pytest.mark.asyncio(loop_scope="session")
async def test_stream_chat_completion_with_tool_calls():
    """
    Test the stream_chat_completion function.
    """
    api_key: str | None = getenv("OPEN_ROUTER_API_KEY")
    if not api_key:
        return pytest.skip("OPEN_ROUTER_API_KEY is not set, skipping test")

    session = await chat_service.create_chat_session()
    session = await chat_service.upsert_chat_session(session)

    has_errors = False
    has_ended = False
    had_tool_calls = False
    async for chunk in chat_service.stream_chat_completion(
        session.session_id,
        "Please find me an agent that can help me with my business. Use the query 'moneny printing agent'",
        user_id=session.user_id,
    ):
        logger.info(chunk)
        if isinstance(chunk, StreamError):
            has_errors = True

        if isinstance(chunk, StreamEnd):
            has_ended = True
        if isinstance(chunk, StreamToolExecutionResult):
            had_tool_calls = True

    assert has_ended, "Chat completion did not end"
    assert not has_errors, "Error occurred while streaming chat completion"
    assert had_tool_calls, "Tool calls did not occur"
    session = await chat_service.get_session(session.session_id)
    assert session, "Session not found"
    assert session.usage, "Usage is empty"
