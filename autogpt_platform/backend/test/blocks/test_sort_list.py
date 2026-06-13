from typing import Any

from backend.blocks.sort_list import SortListBlock


async def _collect_outputs(input_data: dict[str, Any]) -> list[tuple[str, Any]]:
    block = SortListBlock()
    model = SortListBlock.Input(**input_data)
    return [(name, value) async for name, value in block.run(model)]


async def test_builtin_block_cases():
    block = SortListBlock()
    assert isinstance(block.test_input, list)
    assert isinstance(block.test_output, list)
    outputs: list[tuple[str, Any]] = []

    for input_data in block.test_input:
        outputs.extend(await _collect_outputs(input_data))

    assert outputs == block.test_output


async def test_sorts_numbers_naturally():
    outputs = await _collect_outputs({"list": [3, 1, 2]})

    assert outputs == [("sorted_list", [1, 2, 3]), ("length", 3)]


async def test_sorts_in_reverse_order():
    outputs = await _collect_outputs({"list": [3, 1, 2], "reverse": True})

    assert outputs == [("sorted_list", [3, 2, 1]), ("length", 3)]


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
    outputs = await _collect_outputs({"list": [1, "two"]})

    assert outputs[0][0] == "error"
    assert "sort" in outputs[0][1].lower()


async def test_returns_error_when_key_sort_item_is_not_dictionary():
    outputs = await _collect_outputs({"list": [{"score": 1}, 2], "key": "score"})

    assert outputs[0][0] == "error"
    assert "dictionary" in outputs[0][1].lower()


async def test_returns_error_when_key_is_missing():
    outputs = await _collect_outputs(
        {"list": [{"score": 1}, {"name": "a"}], "key": "score"}
    )

    assert outputs[0][0] == "error"
    assert "score" in outputs[0][1]
