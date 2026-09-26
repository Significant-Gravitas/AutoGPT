import pytest

from backend.blocks.conductor.send_message import ConductorSendMessageBlock
from backend.blocks.conductor.test_fixtures import (
    TEST_CREDENTIALS_INPUT,
    collect,
    mock_block,
)
from backend.util.exceptions import BlockExecutionError


@pytest.mark.asyncio
@pytest.mark.parametrize("receipt", [{}, {"messageId": None}, {"messageId": ""}])
async def test_missing_receipt_fails_before_waiting(receipt: dict):
    block = ConductorSendMessageBlock()
    mock_block(
        block,
        {
            "_send": lambda *args, **kwargs: receipt,
            "_wait": lambda *args, **kwargs: pytest.fail(
                "must not poll without a receipt"
            ),
        },
    )

    with pytest.raises(BlockExecutionError, match="messageId"):
        await collect(
            block,
            {
                "credentials": TEST_CREDENTIALS_INPUT,
                "session_id": "s1",
                "message": "go",
            },
        )
