import pytest
from pydantic import BaseModel

from backend.util.type import coerce_inputs_to_schema, convert


@pytest.mark.parametrize(
    "value, expected",
    [
        (" \tTrUe\n", True),
        ("\r1\t", True),
        ("\u2003true\u2003", True),
        (" false ", False),
        (" 0 ", False),
        (" \t\n", False),
        ("", False),
        (" truthy ", False),
    ],
)
def test_boolean_strings_ignore_surrounding_whitespace(value: str, expected: bool):
    assert convert(value, bool) is expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (" \t[1, 2, 1]\n", {1, 2}),
        ("[]", set()),
        (" [not json] ", {"[not json]"}),
        (" [1,] ", {"[1,]"}),
        ("[1, 2", {"[1, 2"}),
        (" hello world ", {"hello world"}),
        ("   ", {""}),
        ('{"key": 1}', {'{"key": 1}'}),
    ],
)
def test_set_strings_parse_arrays_or_preserve_one_value(value: str, expected: set):
    assert convert(value, set) == expected


@pytest.mark.parametrize("value", ["[[1, 2]]", '[{"key": 1}]', "[1, [2]]"])
def test_unhashable_json_set_elements_preserve_the_original_string(value: str):
    assert convert(value, set) == {value}


@pytest.mark.parametrize(
    "value, expected",
    [
        (" \t[1, 2, 1]\n", (1, 2, 1)),
        ("[]", ()),
        (" [not json] ", ("[not json]",)),
        (" [1,] ", ("[1,]",)),
        ("[1, 2", ("[1, 2",)),
        (" hello world ", ("hello world",)),
        ("   ", ("",)),
        ('[[1, 2], {"key": 3}]', ([1, 2], {"key": 3})),
    ],
)
def test_tuple_strings_parse_arrays_or_preserve_one_value(value: str, expected: tuple):
    assert convert(value, tuple) == expected


@pytest.mark.parametrize("value", ['["1", "2", "3"]', ["1", "2", "3"], ("1", "2", "3")])
def test_variadic_tuples_convert_every_element(value: object):
    assert convert(value, tuple[int, ...]) == (1, 2, 3)


@pytest.mark.parametrize("value", ['["5", 6]', ["5", 6], ("5", 6)])
def test_fixed_length_tuples_keep_their_per_element_types(value: object):
    assert convert(value, tuple[int, str]) == (5, "6")


@pytest.mark.parametrize(
    "value, expected",
    [
        ([1, 2, 1], {1, 2}),
        ((1, 2, 1), {1, 2}),
        ({1, 2}, {1, 2}),
        ({"key": 1}, {("key", 1)}),
        (7, {7}),
    ],
)
def test_set_non_string_conversion_is_unchanged(value: object, expected: set):
    assert convert(value, set) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        ([1, 2, 1], (1, 2, 1)),
        ((1, 2, 1), (1, 2, 1)),
        ({1}, (1,)),
        ({"key": 1}, (("key", 1),)),
        (7, (7,)),
        (None, (None,)),
    ],
)
def test_tuple_non_string_conversion_is_unchanged(value: object, expected: tuple):
    assert convert(value, tuple) == expected


@pytest.mark.parametrize(
    "value, expected", [(True, True), (False, False), (1, True), (0, False)]
)
def test_boolean_non_string_conversion_is_unchanged(value: object, expected: bool):
    assert convert(value, bool) is expected


class CollectionInputs(BaseModel):
    enabled: bool
    tags: set[str]
    values: tuple[int, ...]


def test_schema_coercion_preserves_a_json_array_for_variadic_tuple_inputs():
    values = {
        "enabled": " true ",
        "tags": ' ["a", "b", "a"] ',
        "values": '["1", "2", "3"]',
    }
    coerce_inputs_to_schema(values, CollectionInputs)
    result = CollectionInputs.model_validate(values)
    assert result.enabled is True
    assert result.tags == {"a", "b"}
    assert result.values == (1, 2, 3)
