"""Tests that every SpecStream block needs the user's own API key."""

from unittest.mock import patch

import pytest
from pydantic import SecretStr

from backend.blocks.specstream._config import TEST_CREDENTIALS_INPUT
from backend.blocks.specstream._test_helpers import approve_inputs
from backend.blocks.specstream._testdata import TEST_FIND_ID
from backend.blocks.specstream.approve_and_run import SpecStreamApproveAndRunBlock
from backend.blocks.specstream.dataset_request import (
    SpecStreamCreateDatasetRequestBlock,
)
from backend.blocks.specstream.list_rows import SpecStreamListRowsBlock
from backend.blocks.specstream.run_status import (
    SpecStreamGetRunStatusBlock,
    SpecStreamWaitForRunBlock,
)
from backend.data.model import APIKeyCredentials
from backend.util.exceptions import BlockExecutionError

PLATFORM_KEY = APIKeyCredentials(
    id="specstream-default",
    provider="specstream",
    api_key=SecretStr("platform-key"),
    title="SpecStream API Key",
)
BLOCK_INPUTS = [
    (SpecStreamCreateDatasetRequestBlock, {"request": "anything"}),
    (SpecStreamApproveAndRunBlock, approve_inputs()),
    (SpecStreamGetRunStatusBlock, {"run_id": TEST_FIND_ID}),
    (SpecStreamWaitForRunBlock, {"run_id": TEST_FIND_ID}),
    (SpecStreamListRowsBlock, {"run_id": TEST_FIND_ID}),
]


@pytest.mark.parametrize("block_class, inputs", BLOCK_INPUTS)
@pytest.mark.asyncio
async def test_every_block_refuses_the_platform_key_before_any_call(
    block_class, inputs
):
    block = block_class()
    with (
        patch("backend.blocks.specstream._api.Requests") as requests,
        pytest.raises(BlockExecutionError, match="own SpecStream API key"),
    ):
        async for _ in block.run(
            block.input_schema(credentials=TEST_CREDENTIALS_INPUT, **inputs),
            credentials=PLATFORM_KEY,
        ):
            pass
    requests.assert_not_called()
