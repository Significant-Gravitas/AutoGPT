from typing import Any

import pytest
from pydantic import BaseModel

from backend.data.dynamic_fields import (
    DICT_SPLIT,
    MAX_DYNAMIC_LIST_INDEX,
    MAX_DYNAMIC_PATH_LENGTH,
    MAX_DYNAMIC_PATH_TOKENS,
    assert_safe_pin_name,
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
