from typing import cast

import pytest

from backend.blocks import get_blocks
from backend.blocks.google import docs
from backend.blocks.google._auth import TEST_CREDENTIALS
from backend.data.block import BlockInput

_DOCUMENT_BLOCKS = [
    block
    for block in get_blocks().values()
    if block.__module__ == docs.__name__
    and "document" in block().input_schema.model_fields
]


@pytest.mark.parametrize("block_cls", _DOCUMENT_BLOCKS, ids=lambda b: b.__name__)
async def test_a_block_with_no_document_selected_yields_only_that_error(block_cls):
    block = block_cls()
    test_input = cast(BlockInput, block.test_input)
    inputs = {k: v for k, v in test_input.items() if k != "document"}

    outputs = [
        output
        async for output in block.run(
            block.input_schema(**inputs), credentials=TEST_CREDENTIALS
        )
    ]

    assert outputs == [("error", "No document selected")]
