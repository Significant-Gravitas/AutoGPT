"""Tests for SuggestNextStepsTool."""

from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.model import ChatSession
from backend.copilot.response_model import MAX_SUGGESTIONS, StreamSuggestions
from backend.copilot.tools.models import ErrorResponse, SuggestNextStepsResponse
from backend.copilot.tools.suggest_next_steps import SuggestNextStepsTool


@pytest.fixture()
def tool() -> SuggestNextStepsTool:
    return SuggestNextStepsTool()


@pytest.fixture()
def session() -> ChatSession:
    return ChatSession.new(user_id="test-user", dry_run=False)


class _ActiveTurn:
    def __init__(self, turn_id: str) -> None:
        self.turn_id = turn_id


@pytest.fixture()
def publish():
    with patch(
        "backend.copilot.tools.suggest_next_steps.publish_chunk",
        new=AsyncMock(),
    ) as publish_chunk, patch(
        "backend.copilot.tools.suggest_next_steps.get_session",
        new=AsyncMock(return_value=_ActiveTurn("turn-1")),
    ):
        yield publish_chunk


@pytest.mark.asyncio
async def test_publishes_chips_onto_the_turn_stream(
    tool: SuggestNextStepsTool, session: ChatSession, publish: AsyncMock
):
    result = await tool._execute(
        user_id="test-user",
        session=session,
        suggestions=["Email the report", "Post on r/SaaS"],
    )

    assert isinstance(result, SuggestNextStepsResponse)
    assert result.suggestions == ["Email the report", "Post on r/SaaS"]

    publish.assert_awaited_once()
    turn_id, event = publish.await_args.args
    assert turn_id == "turn-1"
    assert isinstance(event, StreamSuggestions)
    assert event.suggestions == ["Email the report", "Post on r/SaaS"]


@pytest.mark.asyncio
async def test_extra_suggestions_are_dropped(
    tool: SuggestNextStepsTool, session: ChatSession, publish: AsyncMock
):
    result = await tool._execute(
        user_id="test-user",
        session=session,
        suggestions=["One", "Two", "Three", "Four"],
    )

    assert isinstance(result, SuggestNextStepsResponse)
    assert result.suggestions == ["One", "Two", "Three"]
    assert len(result.suggestions) == MAX_SUGGESTIONS


@pytest.mark.asyncio
async def test_non_list_returns_error_without_publishing(
    tool: SuggestNextStepsTool, session: ChatSession, publish: AsyncMock
):
    result = await tool._execute(
        user_id="test-user", session=session, suggestions="Email the report"
    )

    assert isinstance(result, ErrorResponse)
    publish.assert_not_awaited()


@pytest.mark.asyncio
async def test_non_string_entries_return_error(
    tool: SuggestNextStepsTool, session: ChatSession, publish: AsyncMock
):
    result = await tool._execute(
        user_id="test-user", session=session, suggestions=["Ok", 7]
    )

    assert isinstance(result, ErrorResponse)
    publish.assert_not_awaited()


@pytest.mark.asyncio
async def test_all_blank_suggestions_return_error(
    tool: SuggestNextStepsTool, session: ChatSession, publish: AsyncMock
):
    result = await tool._execute(
        user_id="test-user", session=session, suggestions=["", "   "]
    )

    assert isinstance(result, ErrorResponse)
    publish.assert_not_awaited()


@pytest.mark.asyncio
async def test_publish_failure_does_not_fail_the_turn(
    tool: SuggestNextStepsTool, session: ChatSession
):
    """Chips are a convenience surface — a Redis hiccup must not surface as
    a tool error the model then has to apologise for."""
    with patch(
        "backend.copilot.tools.suggest_next_steps.get_session",
        new=AsyncMock(side_effect=RuntimeError("redis down")),
    ):
        result = await tool._execute(
            user_id="test-user", session=session, suggestions=["Email the report"]
        )

    assert isinstance(result, SuggestNextStepsResponse)
    assert result.suggestions == ["Email the report"]


def test_openai_schema_shape(tool: SuggestNextStepsTool):
    schema = tool.as_openai_tool()
    assert schema["function"]["name"] == "suggest_next_steps"
    params = schema["function"]["parameters"]
    assert params["required"] == ["suggestions"]
    assert params["properties"]["suggestions"]["maxItems"] == MAX_SUGGESTIONS
