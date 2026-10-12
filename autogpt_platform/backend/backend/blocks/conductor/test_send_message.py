import pytest

from backend.blocks.conductor.send_message import ConductorSendMessageBlock
from backend.blocks.conductor.test_fixtures import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    mock_block,
)
from backend.util.exceptions import BlockExecutionError


@pytest.mark.asyncio
@pytest.mark.parametrize("wait_for_reply", [True, False])
@pytest.mark.parametrize("receipt", [{}, {"messageId": None}, {"messageId": ""}])
async def test_missing_receipt_fails_before_emitting_outputs(
    receipt: dict, wait_for_reply: bool
):
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
    input_data = block.Input.model_validate(
        {
            "credentials": TEST_CREDENTIALS_INPUT,
            "session_id": "s1",
            "message": "go",
            "wait_for_reply": wait_for_reply,
        }
    )
    outputs = []

    with pytest.raises(BlockExecutionError, match="messageId"):
        async for output in block.run(input_data, credentials=TEST_CREDENTIALS):
            outputs.append(output)

    assert outputs == []
