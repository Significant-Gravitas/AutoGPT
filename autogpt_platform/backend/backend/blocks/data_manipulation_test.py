from typing import Any

import pytest

from backend.blocks.data_manipulation import FindInDictionaryBlock


async def _find(obj: Any, key: Any) -> list[tuple[str, Any]]:
    block = FindInDictionaryBlock()
    return [
        output
        async for output in block.run(FindInDictionaryBlock.Input(input=obj, key=key))
    ]


@pytest.mark.parametrize(
    "obj, key, expected",
    [
        # The builder sends a `str | int` field as a string by default.
        ([10, 20, 30], "1", [("output", 20)]),
        ([10, 20, 30], " 2 ", [("output", 30)]),
        ("[10, 20, 30]", "0", [("output", 10)]),
        (["a", "b"], "5", [("missing", ["a", "b"])]),
        ([], "0", [("missing", [])]),
        ([10, 20, 30], 5, [("missing", [10, 20, 30])]),
    ],
)
async def test_list_lookup_by_numeric_string_key(obj, key, expected):
    assert await _find(obj, key) == expected


@pytest.mark.parametrize("key", ["²", "٣", "1²"])
async def test_non_ascii_digit_keys_are_not_list_indexes(key):
    # These pass str.isdigit(); they must not be used as indexes (int("²")
    # raises) and fall back to the attribute lookup as before.
    assert await _find([10, 20, 30], key) == [("output", [])]


async def test_very_long_digit_key_is_missing_not_an_error():
    # int() rejects digit strings over the interpreter's limit (4300 by
    # default since Python 3.11); such a key can't be a valid index anyway.
    assert await _find([10, 20, 30], "1" * 5000) == [("missing", [10, 20, 30])]


async def test_non_index_key_on_empty_list_unchanged():
    assert await _find([], "name") == [("output", [])]


async def test_list_of_dicts_still_plucks_matching_key():
    obj = [{"1": "a"}, {"2": "b"}, {"1": "c"}]
    assert await _find(obj, "1") == [("output", ["a", "c"])]


async def test_list_lookup_by_int_key_unchanged():
    assert await _find([10, 20, 30], 1) == [("output", 20)]
