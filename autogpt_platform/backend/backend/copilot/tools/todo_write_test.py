"""Tests for TodoWriteTool."""

import pytest

from backend.copilot.model import ChatSession
from backend.copilot.tools.models import ErrorResponse, TodoItem, TodoWriteResponse
from backend.copilot.tools.todo_write import TodoWriteTool


@pytest.fixture()
def tool() -> TodoWriteTool:
    return TodoWriteTool()


@pytest.fixture()
def session() -> ChatSession:
    return ChatSession.new(user_id="test-user", dry_run=False)


@pytest.mark.asyncio
async def test_valid_todo_list(tool: TodoWriteTool, session: ChatSession):
    result = await tool._execute(
        user_id=None,
        session=session,
        todos=[
            {
                "content": "Write tests",
                "activeForm": "Writing tests",
                "status": "pending",
            },
            {
                "content": "Ship PR",
                "activeForm": "Shipping PR",
                "status": "in_progress",
            },
        ],
    )

    assert isinstance(result, TodoWriteResponse)
    assert result.session_id == session.session_id
    assert len(result.todos) == 2
    assert result.todos[0] == TodoItem(
        content="Write tests",
        activeForm="Writing tests",
        status="pending",
    )
    assert result.todos[1].status == "in_progress"


@pytest.mark.asyncio
async def test_default_status_is_pending(tool: TodoWriteTool, session: ChatSession):
    result = await tool._execute(
        user_id=None,
        session=session,
        todos=[{"content": "Write tests", "activeForm": "Writing tests"}],
    )

    assert isinstance(result, TodoWriteResponse)
    assert result.todos[0].status == "pending"


@pytest.mark.asyncio
async def test_missing_todos_returns_error(tool: TodoWriteTool, session: ChatSession):
    result = await tool._execute(user_id=None, session=session)

    assert isinstance(result, ErrorResponse)
    assert "todos" in result.message.lower()


@pytest.mark.asyncio
async def test_non_list_todos_returns_error(tool: TodoWriteTool, session: ChatSession):
    result = await tool._execute(user_id=None, session=session, todos="not a list")

    assert isinstance(result, ErrorResponse)


@pytest.mark.asyncio
async def test_invalid_item_returns_error(tool: TodoWriteTool, session: ChatSession):
    # Missing required `activeForm` field.
    result = await tool._execute(
        user_id=None,
        session=session,
        todos=[{"content": "Missing active form"}],
    )

    assert isinstance(result, ErrorResponse)


@pytest.mark.asyncio
async def test_multiple_in_progress_rejected(tool: TodoWriteTool, session: ChatSession):
    """Exactly one item should be in_progress at a time — SDK parity rule."""
    result = await tool._execute(
        user_id=None,
        session=session,
        todos=[
            {
                "content": "A",
                "activeForm": "Doing A",
                "status": "in_progress",
            },
            {
                "content": "B",
                "activeForm": "Doing B",
                "status": "in_progress",
            },
        ],
    )

    assert isinstance(result, ErrorResponse)
    assert "in_progress" in result.message


def test_openai_schema_shape(tool: TodoWriteTool):
    schema = tool.as_openai_tool()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "TodoWrite"
    params = schema["function"]["parameters"]
    assert params["required"] == ["todos"]
    items = params["properties"]["todos"]["items"]
    assert items["required"] == ["content", "activeForm"]
    assert items["properties"]["status"]["enum"] == [
        "pending",
        "in_progress",
        "completed",
    ]
