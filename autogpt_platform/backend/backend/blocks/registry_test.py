from unittest.mock import patch

import backend.blocks as blocks
from backend.blocks._base import BlockType


class _OutputBlock:
    block_type = BlockType.OUTPUT


class _InputBlock:
    block_type = BlockType.INPUT


class _OtherBlock:
    block_type = BlockType.STANDARD


def test_get_output_block_ids_returns_exactly_output_blocks():
    blocks.get_output_block_ids.cache_clear()
    try:
        with patch.object(
            blocks,
            "get_blocks",
            return_value={
                "output-1": _OutputBlock,
                "input-1": _InputBlock,
                "other-1": _OtherBlock,
            },
        ):
            assert list(blocks.get_output_block_ids()) == ["output-1"]
    finally:
        blocks.get_output_block_ids.cache_clear()
