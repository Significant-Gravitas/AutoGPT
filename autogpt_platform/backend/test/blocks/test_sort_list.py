from typing import Any

import pytest

from backend.blocks.sort_list import SortListBlock
from backend.data.execution import ExecutionContext
from backend.util.exceptions import BlockExecutionError
from backend.util.test import execute_block_test


async def _collect_outputs(input_data: dict[str, Any]) -> list[tuple[str, Any]]:
    block = SortListBlock()
    return [
        (name, value)
        async for name, value in block.execute(
            input_data,
            execution_context=ExecutionContext(),
        )
    ]


async def test_builtin_block_cases():
    await execute_block_test(SortListBlock())


def test_error_output_keeps_default_value():
    error_field = SortListBlock.Output.model_fields["error"]

    assert not error_field.is_required()
    assert error_field.default == ""


async def test_sorts_numbers_naturally():
    outputs = await _collect_outputs({"list": [3, 1, 2]})

    assert outputs == [("sorted_list", [1, 2, 3]), ("length", 3)]


async def test_sorts_in_reverse_order():
    outputs = await _collect_outputs({"list": [3, 1, 2], "reverse": True})

    assert outputs == [("sorted_list", [3, 2, 1]), ("length", 3)]


async def test_empty_key_sorts_items_directly():
    outputs = await _collect_outputs({"list": [3, 1, 2], "key": ""})

    assert outputs == [("sorted_list", [1, 2, 3]), ("length", 3)]


async def test_sorts_dictionaries_by_key():
    outputs = await _collect_outputs(
        {
            "list": [
                {"name": "b", "score": 2},
                {"name": "a", "score": 1},
            ],
            "key": "score",
        }
    )

    assert outputs == [
        (
            "sorted_list",
            [
                {"name": "a", "score": 1},
                {"name": "b", "score": 2},
            ],
        ),
        ("length", 2),
    ]


async def test_does_not_mutate_original_list():
    original = [{"name": "b", "score": 2}, {"name": "a", "score": 1}]

    await _collect_outputs({"list": original, "key": "score"})

    assert original == [{"name": "b", "score": 2}, {"name": "a", "score": 1}]


async def test_returns_error_for_mixed_unsortable_values():
    with pytest.raises(BlockExecutionError, match="sort"):
        await _collect_outputs({"list": [1, "two"]})


async def test_returns_error_when_key_sort_item_is_not_dictionary():
    with pytest.raises(BlockExecutionError, match="dictionary"):
        await _collect_outputs({"list": [{"score": 1}, 2], "key": "score"})


async def test_returns_error_when_key_is_missing():
    with pytest.raises(BlockExecutionError, match="score"):
        await _collect_outputs({"list": [{"score": 1}, {"name": "a"}], "key": "score"})
