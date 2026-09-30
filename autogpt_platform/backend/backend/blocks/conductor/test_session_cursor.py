import pytest

from backend.blocks.conductor.get_session import ConductorGetSessionBlock
from backend.blocks.conductor.test_fixtures import (
    TEST_CREDENTIALS_INPUT,
    collect,
    mock_block,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("messages", [[], [{"content": "missing row ID"}]])
async def test_incremental_poll_preserves_cursor_without_a_new_row_id(messages: list):
    block = ConductorGetSessionBlock()
    mock_block(
        block,
        {"_fetch": lambda *args, **kwargs: {"messages": {"data": messages}}},
    )

    result = await collect(
        block,
        {
            "credentials": TEST_CREDENTIALS_INPUT,
            "session_id": "s1",
            "after": "last-seen-row",
        },
    )

    assert result["messages"] == messages
    assert result["next_after"] == "last-seen-row"
