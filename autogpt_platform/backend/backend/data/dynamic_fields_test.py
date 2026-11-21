from typing import Any

import pytest
from pydantic import BaseModel

from backend.data.dynamic_fields import (
    DICT_SPLIT,
    MAX_DYNAMIC_LIST_INDEX,
    MAX_DYNAMIC_PATH_LENGTH,
    MAX_DYNAMIC_PATH_TOKENS,
    assert_safe_pin_name,
    get_dynamic_field_description,
    get_dynamic_field_schema,
    is_safe_object_attribute,
    merge_execution_input,
    parse_execution_output,
)


@pytest.mark.parametrize(
    "ident,safe",
    [
        ("attr", True),
        ("attr1", True),
        ("snake_case_attr", True),
        ("_private", False),
        ("__dict__", False),
        ("__class__", False),
        ("attr-with-dash", False),
        ("1leading_digit", False),
        ("has space", False),
        ("", False),
    ],
)
def test_is_safe_object_attribute(ident: str, safe: bool):
    assert is_safe_object_attribute(ident) is safe


def test_assert_safe_pin_name_allows_plain_and_legitimate_dynamic_names():
    # A plain field name and every legitimate dynamic shape must pass unchanged.
    for name in [
        "plain_field",
        "obj_@_attr1",
        "values_#_key",
        "items_$_0",
        "nested_@_items_$_1_#_key",
        "data_#___proto__",  # a dunder dict KEY is data, not an attribute
    ]:
        assert_safe_pin_name(name)


@pytest.mark.parametrize(
    "name",
    [
        "obj_@___dict__",
        "obj_@___class__",
        "obj_@__private",
        "nested_#_key_@___dict__",
        "nested_$_0_@___class__",
    ],
)
def test_assert_safe_pin_name_rejects_unsafe_object_attribute(name: str):
    with pytest.raises(ValueError, match="unsafe object attribute"):
        assert_safe_pin_name(name)


def test_assert_safe_pin_name_rejects_out_of_range_list_index():
    assert_safe_pin_name(f"items_$_{MAX_DYNAMIC_LIST_INDEX}")  # boundary is allowed
    with pytest.raises(ValueError, match="out of range"):
        assert_safe_pin_name(f"items_$_{MAX_DYNAMIC_LIST_INDEX + 1}")


def test_merge_execution_input_refuses_unsafe_object_attribute():
    # Reaching an attribute through _@_ must raise, not silently build the object.
    with pytest.raises(ValueError, match="unsafe object attribute"):
        merge_execution_input({"obj_@___dict__": "x"})


def test_merge_execution_input_refuses_out_of_range_list_index():
    with pytest.raises(ValueError, match="out of range"):
        merge_execution_input({"items_$_" + str(MAX_DYNAMIC_LIST_INDEX + 1): "x"})


def test_merge_execution_input_allows_safe_attribute():
    result = merge_execution_input({"obj_@_attr": "value"})
    assert result["obj"].attr == "value"


class _Target:
    def __init__(self):
        self.public = "visible"


def test_parse_execution_output_reads_safe_attribute():
    assert parse_execution_output(("out", _Target()), "out_@_public") == "visible"


@pytest.mark.parametrize("selector", ["out_@___dict__", "out_@___class__"])
def test_parse_execution_output_refuses_unsafe_attribute(selector: str):
    # The attribute exists on the object, but the guard must still refuse it.
    assert parse_execution_output(("out", _Target()), selector) is None


def test_merge_execution_input_cannot_reach_interpreter_state():
    """A chain of object-attribute tokens must not escape the value being built.

    Without the guard, the chain below walks off the placeholder object into a
    live module's namespace and writes there, so the assertion on `mock_module`
    is what makes this test able to fail.
    """
    import backend.util.mock as mock_module

    marker = "_pin_name_guard_marker"
    chain = "x_@___class___@___init___@___globals___#_" + marker

    with pytest.raises(ValueError, match="unsafe object attribute"):
        merge_execution_input({chain: "value"})

    assert marker not in vars(mock_module)


@pytest.mark.parametrize(
    "at_cap,over_cap,error",
    [
        (
            ["k" * (MAX_DYNAMIC_PATH_LENGTH - len(DICT_SPLIT))],
            ["k" * (MAX_DYNAMIC_PATH_LENGTH - len(DICT_SPLIT) + 1)],
            "longer than",
        ),
        (
            ["k"] * MAX_DYNAMIC_PATH_TOKENS,
            ["k"] * (MAX_DYNAMIC_PATH_TOKENS + 1),
            "deeper than",
        ),
    ],
    ids=["length", "depth"],
)
def test_dynamic_path_size_is_capped(
    at_cap: list[str], over_cap: list[str], error: str
):
    assert_safe_pin_name(_dict_path(at_cap))
    with pytest.raises(ValueError, match=error):
        assert_safe_pin_name(_dict_path(over_cap))
    with pytest.raises(ValueError, match=error):
        merge_execution_input({_dict_path(over_cap): "x"})
    # The path resolves against this output without the cap, so None is the cap.
    output = ("out", _nested(over_cap))
    assert parse_execution_output(output, _dict_path(over_cap)) is None


def test_merge_execution_input_padding_budget_fits_one_maximal_index():
    # Assigning into slots an earlier key already padded costs nothing.
    merged = merge_execution_input({f"a_$_{MAX_DYNAMIC_LIST_INDEX}": 1, "a_$_0": 2})
    assert len(merged["a"]) == MAX_DYNAMIC_LIST_INDEX + 1
    assert merged["a"][0] == 2


@pytest.mark.parametrize(
    "data",
    [
        {f"a_$_{MAX_DYNAMIC_LIST_INDEX}": 1, "b_$_0": 2},
        {f"a_$_{MAX_DYNAMIC_LIST_INDEX}_$_0": 1},
    ],
    ids=["second-list", "nested-list"],
)
def test_merge_execution_input_caps_total_list_padding(data: dict[str, Any]):
    with pytest.raises(ValueError, match="padding exceeds"):
        merge_execution_input(data)


def test_parse_execution_output_reads_instance_data_only():
    class Model(BaseModel):
        field: str = "visible"

    def generator():
        yield 1

    assert parse_execution_output(("out", Model()), "out_@_field") == "visible"
    # A public name that is not instance data: the generator's live frame.
    assert parse_execution_output(("out", generator()), "out_@_gi_frame") is None


def _dict_path(keys: list[str]) -> str:
    return "out" + "".join(DICT_SPLIT + key for key in keys)


def _nested(keys: list[str]) -> Any:
    value: Any = "leaf"
    for key in reversed(keys):
        value = {key: value}
    return value


@pytest.mark.parametrize(
    "field_name,expected",
    [
        (
            "timing_#_start_datetime",
            "Dictionary field 'start_datetime' for base field 'timing'"
            " (timing['start_datetime'])",
        ),
        ("items_$_0", "List item 0 for base field 'items' (items[0])"),
        (
            "user_@_email_address",
            "Object attribute 'email_address' for base field 'user'"
            " (user.email_address)",
        ),
        (
            "rows_$_2_#_first_name",
            "Dictionary field 'first_name' for base field 'rows'"
            " (rows[2]['first_name'])",
        ),
        ("regular_field", "Value for regular_field"),
    ],
)
def test_dynamic_field_description_names_the_full_key(field_name, expected):
    assert get_dynamic_field_description(field_name) == expected


_TIMING_SCHEMA = {
    "oneOf": [
        {
            "type": "object",
            "properties": {
                "discriminator": {"const": "exact_timing", "type": "string"},
                "start_datetime": {"type": "string", "format": "date-time"},
                "end_datetime": {"type": "string", "format": "date-time"},
            },
        },
        {
            "type": "object",
            "properties": {
                "discriminator": {"const": "duration_timing", "type": "string"},
                "start_datetime": {"type": "string", "format": "date-time"},
                "duration_minutes": {"type": "integer"},
            },
        },
    ]
}


def test_dynamic_field_schema_finds_a_key_in_one_union_variant():
    assert get_dynamic_field_schema(_TIMING_SCHEMA, "timing_#_duration_minutes") == {
        "type": "integer"
    }


def test_dynamic_field_schema_merges_a_key_shared_by_union_variants():
    assert get_dynamic_field_schema(_TIMING_SCHEMA, "timing_#_start_datetime") == {
        "type": "string",
        "format": "date-time",
    }
    assert get_dynamic_field_schema(_TIMING_SCHEMA, "timing_#_discriminator") == {
        "anyOf": [
            {"const": "exact_timing", "type": "string"},
            {"const": "duration_timing", "type": "string"},
        ]
    }


def test_dynamic_field_schema_follows_lists_and_typed_dicts():
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": {"type": "number"},
        },
    }
    assert get_dynamic_field_schema(schema, "scores_$_0_#_math") == {"type": "number"}


def test_dynamic_field_schema_keeps_every_all_of_constraint():
    schema = {
        "allOf": [
            {"properties": {"x": {"type": "integer", "minimum": 0}}},
            {"properties": {"x": {"type": "integer", "maximum": 10}}},
        ]
    }
    assert get_dynamic_field_schema(schema, "data_#_x") == {
        "allOf": [
            {"type": "integer", "minimum": 0},
            {"type": "integer", "maximum": 10},
        ]
    }


def test_dynamic_field_schema_keeps_sibling_any_of_and_one_of_apart():
    schema = {
        "anyOf": [{"properties": {"x": {"type": "integer", "minimum": 0}}}],
        "oneOf": [{"properties": {"x": {"type": "integer", "maximum": 10}}}],
    }
    assert get_dynamic_field_schema(schema, "data_#_x") == {
        "allOf": [
            {"type": "integer", "minimum": 0},
            {"type": "integer", "maximum": 10},
        ]
    }


def test_dynamic_field_schema_reads_boolean_prefix_items():
    schema = {"type": "array", "prefixItems": [True, {"type": "integer"}, False]}
    assert get_dynamic_field_schema(schema, "items_$_0") is None
    assert get_dynamic_field_schema(schema, "items_$_1") == {"type": "integer"}
    assert get_dynamic_field_schema(schema, "items_$_2") is None


@pytest.mark.parametrize(
    "schema,field_name",
    [
        # a union with an untyped variant accepts any value
        (
            {
                "anyOf": [
                    {"type": "object", "additionalProperties": {"type": "integer"}},
                    {"type": "object", "additionalProperties": {}},
                ]
            },
            "values_#_name",
        ),
        # dict[str, Any]: the value is unconstrained
        ({"type": "object", "additionalProperties": True}, "values_#_name"),
        ({"type": "object", "additionalProperties": {}}, "values_#_name"),
        # list[Any]
        ({"type": "array", "items": {}}, "entries_$_0"),
        # the key isn't in the schema
        (_TIMING_SCHEMA, "timing_#_unknown"),
        # an index the executor can't merge
        ({"type": "array", "items": {"type": "integer"}}, "entries_$_first"),
        # Any
        ({}, "data_@_user_name"),
        ({"type": "object"}, "plain_field"),
    ],
)
def test_dynamic_field_schema_is_none_without_a_typed_leaf(schema, field_name):
    assert get_dynamic_field_schema(schema, field_name) is None


def test_dynamic_field_schema_returns_a_copy():
    leaf = get_dynamic_field_schema(_TIMING_SCHEMA, "timing_#_duration_minutes")
    assert leaf is not None
    leaf["description"] = "changed"
    assert (
        "description"
        not in _TIMING_SCHEMA["oneOf"][1]["properties"]["duration_minutes"]
    )
