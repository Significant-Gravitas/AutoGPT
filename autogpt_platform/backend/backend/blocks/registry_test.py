from unittest.mock import patch

import pytest

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


@pytest.mark.parametrize(
    "file_name, is_test",
    [
        ("test_block.py", True),
        ("examples_test.py", True),
        ("_client_test.py", True),
        ("_test.py", True),
        ("conftest.py", True),
        ("llm.py", False),
        ("testing.py", False),
        ("latest.py", False),
    ],
)
def test_is_test_module(file_name: str, is_test: bool):
    assert blocks._is_test_module(file_name) is is_test


def test_load_all_blocks_skips_test_modules():
    blocks.load_all_blocks.cache_clear()
    try:
        with (
            patch.object(blocks, "importlib") as importlib,
            patch.object(blocks, "_all_subclasses", return_value=[]),
        ):
            blocks.load_all_blocks()
        imported = {call.args[0] for call in importlib.import_module.call_args_list}
    finally:
        blocks.load_all_blocks.cache_clear()

    assert ".llm" in imported
    assert ".slant3d.webhook" in imported
    assert ".slant3d.examples_test" not in imported
    assert ".slant3d.conftest" not in imported
    assert ".typesafe._client_test" not in imported
