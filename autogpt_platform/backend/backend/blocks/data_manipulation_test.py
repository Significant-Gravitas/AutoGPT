import json
from types import SimpleNamespace
from typing import Any

import pytest

from backend.blocks.data_manipulation import FindInDictionaryBlock
from backend.util.mock import MockObject


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
        # Same range rule as an int key: negatives are missing.
        ([1, 2, 3], "-1", [("missing", [1, 2, 3])]),
        ([1, 2, 3], -1, [("missing", [1, 2, 3])]),
        ([10, 20, 30], 5, [("missing", [10, 20, 30])]),
    ],
)
async def test_list_lookup_by_numeric_string_key(obj, key, expected):
    assert await _find(obj, key) == expected


@pytest.mark.parametrize("key", ["²", "٣", "1²"])
async def test_non_ascii_digit_keys_are_not_list_indexes(key):
    # These pass str.isdigit(); they must not be used as indexes (int("²")
    # raises) and fall back to the attribute lookup, which finds nothing.
    assert await _find([10, 20, 30], key) == [("missing", [10, 20, 30])]


async def test_very_long_digit_key_is_missing_not_an_error():
    # int() rejects digit strings over the interpreter's limit (4300 by
    # default since Python 3.11); such a key can't be a valid index anyway.
    assert await _find([10, 20, 30], "1" * 5000) == [("missing", [10, 20, 30])]


async def test_non_index_key_on_empty_list_unchanged():
    assert await _find([], "name") == [("output", [])]


async def test_list_of_dicts_still_plucks_matching_key():
    obj = [{"1": "a"}, {"2": "b"}, {"1": "c"}]
    assert await _find(obj, "1") == [("output", ["a", "c"])]


async def test_digit_key_on_list_of_dicts_never_picks_a_row():
    # Whether "1" plucks a column must not depend on which dict comes first.
    assert await _find([{"2": "b"}, {"1": "a"}], "1") == [("output", ["a"])]


async def test_list_lookup_by_int_key_unchanged():
    assert await _find([10, 20, 30], 1) == [("output", 20)]


SPARSE = [{"order": "A"}, {"tracking": "T1"}, {"tracking": "T2"}]


async def test_sparse_list_of_dicts_plucks_from_every_dict():
    # The first dict lacks the key; the others still count.
    assert await _find(SPARSE, "tracking") == [("output", ["T1", "T2"])]


async def test_sparse_list_of_dicts_as_json_string():
    assert await _find(json.dumps(SPARSE), "tracking") == [("output", ["T1", "T2"])]


async def test_list_of_dicts_without_the_key_is_missing():
    obj = [{"a": 1}, {"b": 2}]
    assert await _find(obj, "zzz") == [("missing", obj)]


async def test_non_dict_items_in_a_list_of_dicts_are_skipped():
    # A str item used to be substring-tested and then indexed (TypeError).
    assert await _find([{"a": 1}, "xa", {"a": 2}], "a") == [("output", [1, 2])]


async def test_dict_lookup_unchanged():
    assert await _find({"x": 1}, "x") == [("output", 1)]
    assert await _find({"x": 1}, "y") == [("missing", {"x": 1})]


async def test_object_lookup_unchanged():
    assert await _find(MockObject(value="!!", key="key"), "key") == [("output", "key")]


async def test_list_of_objects_plucks_attribute_or_is_missing():
    objs = [SimpleNamespace(value="a"), SimpleNamespace(value="b")]
    assert await _find(objs, "value") == [("output", ["a", "b"])]
    assert await _find(objs, "nope") == [("missing", objs)]
