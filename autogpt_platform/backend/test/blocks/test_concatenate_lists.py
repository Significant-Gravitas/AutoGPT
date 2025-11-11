from autogpt_platform.backend.backend.blocks.concatenate_lists import (
    ConcatenateListsBlock,
)


def test_concatenate_lists_basic():
    block = ConcatenateListsBlock()
    result = block.run([[1, 2], [3, 4]])
    assert result["result"] == [1, 2, 3, 4]


def test_concatenate_lists_empty_lists():
    block = ConcatenateListsBlock()
    result = block.run([[], [], []])
    assert result["result"] == []


def test_concatenate_lists_invalid_input():
    block = ConcatenateListsBlock()
    out = block.run("invalid")
    assert "error" in out
