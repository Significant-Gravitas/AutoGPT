from unittest.mock import AsyncMock, patch

import pytest

from backend.blocks.slant3d._api import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.slant3d.slicing import Slant3DSlicerBlock
from backend.data.execution import ExecutionContext


@pytest.mark.parametrize(
    "requested_quantity,data",
    [
        (10, {"total": 1.37, "quantity": 1}),
        (10, {"total": 1.37}),
        (1, {"total": 10.2, "quantity": 2}),
    ],
)
async def test_slicer_rejects_prices_for_a_different_quantity(requested_quantity, data):
    block = Slant3DSlicerBlock()
    with patch.object(
        block,
        "_make_request",
        AsyncMock(return_value={"message": "File Price Estimated", "data": data}),
    ):
        outputs = block.run(
            block.Input(
                credentials=TEST_CREDENTIALS_INPUT,
                file_id="file-1",
                quantity=requested_quantity,
            ),
            credentials=TEST_CREDENTIALS,
            execution_context=ExecutionContext(user_id="user-1", graph_exec_id="run-1"),
        )
        with pytest.raises(
            ValueError, match=rf"print.*{requested_quantity} were requested"
        ):
            await anext(outputs)
