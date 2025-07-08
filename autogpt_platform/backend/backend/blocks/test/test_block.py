from typing import Type

import pytest

from backend.data.block import Block, get_blocks
from backend.util.test import execute_block_test


@pytest.mark.parametrize("block", get_blocks().values(), ids=lambda b: b.name)
async def test_available_blocks(block: Type[Block]):
    await execute_block_test(block())
