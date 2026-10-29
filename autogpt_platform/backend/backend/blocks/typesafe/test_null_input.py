from unittest.mock import AsyncMock

import pytest

from backend.data.execution import ExecutionContext

from ._config import TEST_CREDENTIALS
from .route import JevRouteBlock
from .yes_no import JevYesNoBlock


@pytest.mark.parametrize(
    "block_type,pin", [(JevRouteBlock, "option_1"), (JevYesNoBlock, "yes")]
)
async def test_null_state_and_data_survive_platform_execution(
    block_type, pin, monkeypatch
):
    block = block_type()
    mock = AsyncMock(side_effect=block.test_mock["call_jev"])
    monkeypatch.setattr(block, "call_jev", mock)
    inputs = {**block.test_input, "state": None, "data": None}
    outputs = dict(
        [
            item
            async for item in block.execute(
                inputs,
                credentials=TEST_CREDENTIALS,
                execution_context=ExecutionContext(),
            )
        ]
    )
    assert outputs[pin] is None
    assert mock.call_args.args[1] is None
