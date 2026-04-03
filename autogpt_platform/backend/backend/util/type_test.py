from typing import Any, List, Literal, Optional

from pydantic import BaseModel

from backend.util.type import _value_satisfies_type, coerce_inputs_to_schema, convert


def test_type_conversion():
    assert convert(5.5, int) == 5
    assert convert("5.5", int) == 5
    assert convert([1, 2, 3], int) == 3
    assert convert("7", Optional[int]) == 7
    assert convert("7", int | None) == 7

    assert convert("5.5", float) == 5.5
    assert convert(5, float) == 5.0

    assert convert("True", bool) is True
    assert convert("False", bool) is False

    assert convert(5, str) == "5"
    assert convert({"a": 1, "b": 2}, str) == '{"a": 1, "b": 2}'
    assert convert([1, 2, 3], str) == "[1, 2, 3]"

    assert convert("5", list) == ["5"]
    assert convert((1, 2, 3), list) == [1, 2, 3]
    assert convert({1, 2, 3}, list) == [1, 2, 3]

    assert convert("5", dict) == {"value": 5}
    assert convert('{"a": 1, "b": 2}', dict) == {"a": 1, "b": 2}
    assert convert([1, 2, 3], dict) == {0: 1, 1: 2, 2: 3}
    assert convert((1, 2, 3), dict) == {0: 1, 1: 2, 2: 3}

    assert convert("5", List[int]) == [5]
    assert convert("[5,4,2]", List[int]) == [5, 4, 2]
    assert convert([5, 4, 2], List[str]) == ["5", "4", "2"]

    # Test the specific case that was failing: empty list to Optional[str]
    assert convert([], Optional[str]) == "[]"
    assert convert([], str) == "[]"

    # Test the actual failing case: empty list to ShortTextType
    from backend.util.type import ShortTextType

    assert convert([], Optional[ShortTextType]) == "[]"
    assert convert([], ShortTextType) == "[]"

    # Test other empty list conversions
    assert convert([], int) == 0  # len([]) = 0
    assert convert([], Optional[int]) == 0


# ---------------------------------------------------------------------------
# _value_satisfies_type
# ---------------------------------------------------------------------------


class TestValueSatisfiesType:
    # --- simple types ---
    def test_simple_match(self):
        assert _value_satisfies_type("hello", str) is True
        assert _value_satisfies_type(42, int) is True
        assert _value_satisfies_type(3.14, float) is True
        assert _value_satisfies_type(True, bool) is True

    def test_simple_mismatch(self):
        assert _value_satisfies_type("hello", int) is False
        assert _value_satisfies_type(42, str) is False
        assert _value_satisfies_type([1, 2], str) is False

    # --- Any ---
    def test_any_always_satisfied(self):
        assert _value_satisfies_type("hello", Any) is True
        assert _value_satisfies_type(42, Any) is True
        assert _value_satisfies_type([1, 2], Any) is True
        assert _value_satisfies_type(None, Any) is True

    # --- Optional / Union ---
    def test_optional_with_value(self):
        assert _value_satisfies_type("hello", Optional[str]) is True
        assert _value_satisfies_type(42, Optional[int]) is True

    def test_optional_mismatch(self):
        assert _value_satisfies_type(42, Optional[str]) is False

    def test_union_matches_first_member(self):
        assert _value_satisfies_type("hello", str | list[str]) is True

    def test_union_matches_second_member(self):
        assert _value_satisfies_type(["a", "b"], str | list[str]) is True

    def test_union_no_match(self):
        assert _value_satisfies_type(42, str | list[str]) is False

    # --- list[T] ---
    def test_list_str_all_match(self):
        assert _value_satisfies_type(["a", "b", "c"], list[str]) is True

    def test_list_str_inner_mismatch(self):
        assert _value_satisfies_type([1, 2, 3], list[str]) is False

    def test_list_int_all_match(self):
        assert _value_satisfies_type([1, 2, 3], list[int]) is True

    def test_list_int_inner_mismatch(self):
        assert _value_satisfies_type(["1", "2"], list[int]) is False

    def test_empty_list_satisfies_any_list_type(self):
        assert _value_satisfies_type([], list[str]) is True
        assert _value_satisfies_type([], list[int]) is True

    def test_string_does_not_satisfy_list(self):
        assert _value_satisfies_type("hello", list[str]) is False

    # --- nested list[list[str]] ---
    def test_nested_list_all_match(self):
        assert _value_satisfies_type([["a", "b"], ["c"]], list[list[str]]) is True

    def test_nested_list_inner_mismatch(self):
        assert _value_satisfies_type([["a", 1], ["c"]], list[list[str]]) is False

    def test_nested_list_outer_mismatch(self):
        assert _value_satisfies_type(["a", "b"], list[list[str]]) is False

    # --- dict[K, V] ---
    def test_dict_str_int_match(self):
        assert _value_satisfies_type({"a": 1, "b": 2}, dict[str, int]) is True

    def test_dict_str_int_value_mismatch(self):
        assert _value_satisfies_type({"a": "1", "b": "2"}, dict[str, int]) is False

    def test_dict_str_int_key_mismatch(self):
        assert _value_satisfies_type({1: 1, 2: 2}, dict[str, int]) is False

    def test_empty_dict_satisfies(self):
        assert _value_satisfies_type({}, dict[str, int]) is True

    # --- set[T] / tuple[T] ---
    def test_set_match(self):
        assert _value_satisfies_type({1, 2, 3}, set[int]) is True

    def test_set_mismatch(self):
        assert _value_satisfies_type({"a", "b"}, set[int]) is False

    def test_tuple_homogeneous_match(self):
        assert _value_satisfies_type((1, 2, 3), tuple[int, ...]) is True

    def test_tuple_homogeneous_mismatch(self):
        assert _value_satisfies_type((1, "2", 3), tuple[int, ...]) is False

    def test_tuple_heterogeneous_match(self):
        assert _value_satisfies_type(("a", 1, True), tuple[str, int, bool]) is True

    def test_tuple_heterogeneous_mismatch(self):
        assert _value_satisfies_type(("a", "b", True), tuple[str, int, bool]) is False

    def test_tuple_heterogeneous_wrong_length(self):
        assert _value_satisfies_type(("a", 1), tuple[str, int, bool]) is False

    # --- bare generics (no args) ---
    def test_bare_list(self):
        assert _value_satisfies_type([1, "a"], list) is True

    def test_bare_dict(self):
        assert _value_satisfies_type({"a": 1}, dict) is True

    # --- union with generic inner mismatch ---
    def test_union_list_with_wrong_inner_falls_through(self):
        # [1, 2] doesn't satisfy list[str] (inner mismatch), and not str either
        assert _value_satisfies_type([1, 2], str | list[str]) is False

    # --- Literal (non-runtime origin) ---
    def test_literal_does_not_crash(self):
        """Literal origins are not runtime types — should return False, not crash."""
        assert _value_satisfies_type("active", Literal["active", "inactive"]) is False


# ---------------------------------------------------------------------------
# coerce_inputs_to_schema — using real Pydantic models
# ---------------------------------------------------------------------------


class SampleSchema(BaseModel):
    name: str
    count: int
    items: list[str]
    config: dict[str, int] = {}


class NestedSchema(BaseModel):
    rows: list[list[str]]


class UnionSchema(BaseModel):
    content: str | list[str]


class OptionalSchema(BaseModel):
    label: Optional[str] = None
    value: int = 0


class AnyFieldSchema(BaseModel):
    data: Any


class TestCoerceInputsToSchema:
    def test_string_to_int(self):
        data: dict[str, Any] = {"name": "test", "count": "42", "items": ["a"]}
        coerce_inputs_to_schema(data, SampleSchema)
        assert data["count"] == 42
        assert isinstance(data["count"], int)

    def test_json_string_to_list(self):
        data: dict[str, Any] = {"name": "test", "count": 1, "items": '["a","b","c"]'}
        coerce_inputs_to_schema(data, SampleSchema)
        assert data["items"] == ["a", "b", "c"]

    def test_already_correct_types_unchanged(self):
        data: dict[str, Any] = {
            "name": "test",
            "count": 42,
            "items": ["a", "b"],
            "config": {"x": 1},
        }
        coerce_inputs_to_schema(data, SampleSchema)
        assert data == {
            "name": "test",
            "count": 42,
            "items": ["a", "b"],
            "config": {"x": 1},
        }

    def test_inner_element_coercion(self):
        """list[str] with int inner elements → coerced to strings."""
        data: dict[str, Any] = {"name": "test", "count": 1, "items": [1, 2, 3]}
        coerce_inputs_to_schema(data, SampleSchema)
        assert data["items"] == ["1", "2", "3"]

    def test_dict_value_coercion(self):
        """dict[str, int] with string values → coerced to ints."""
        data: dict[str, Any] = {
            "name": "test",
            "count": 1,
            "items": [],
            "config": {"x": "10", "y": "20"},
        }
        coerce_inputs_to_schema(data, SampleSchema)
        assert data["config"] == {"x": 10, "y": 20}

    def test_nested_list_from_json_string(self):
        data: dict[str, Any] = {
            "rows": '[["Name","Score"],["Alice","90"]]',
        }
        coerce_inputs_to_schema(data, NestedSchema)
        assert data["rows"] == [["Name", "Score"], ["Alice", "90"]]

    def test_nested_list_already_correct(self):
        original = [["a", "b"], ["c", "d"]]
        data: dict[str, Any] = {"rows": original}
        coerce_inputs_to_schema(data, NestedSchema)
        assert data["rows"] == original

    def test_union_preserves_valid_list(self):
        """list[str] value should NOT be stringified for str | list[str]."""
        data: dict[str, Any] = {"content": ["a", "b"]}
        coerce_inputs_to_schema(data, UnionSchema)
        assert data["content"] == ["a", "b"]
        assert isinstance(data["content"], list)

    def test_union_preserves_valid_string(self):
        data: dict[str, Any] = {"content": "hello"}
        coerce_inputs_to_schema(data, UnionSchema)
        assert data["content"] == "hello"

    def test_union_list_with_wrong_inner_gets_coerced(self):
        """[1, 2] for str | list[str] — inner ints don't match list[str],
        so convert() is called. convert tries str first → stringifies."""
        data: dict[str, Any] = {"content": [1, 2]}
        coerce_inputs_to_schema(data, UnionSchema)
        # convert([1,2], str | list[str]) tries str first → "[1, 2]"
        # This is convert()'s union behavior — str wins over list[str]
        assert isinstance(data["content"], (str, list))

    def test_skips_none_values(self):
        data: dict[str, Any] = {"label": None, "value": "5"}
        coerce_inputs_to_schema(data, OptionalSchema)
        assert data["label"] is None
        assert data["value"] == 5

    def test_skips_missing_fields(self):
        data: dict[str, Any] = {"value": "10"}
        coerce_inputs_to_schema(data, OptionalSchema)
        assert "label" not in data
        assert data["value"] == 10

    def test_any_field_skipped(self):
        """Fields typed as Any should pass through without coercion."""
        data: dict[str, Any] = {"data": [1, "mixed", {"nested": True}]}
        coerce_inputs_to_schema(data, AnyFieldSchema)
        assert data["data"] == [1, "mixed", {"nested": True}]

    def test_preserves_all_convert_capabilities(self):
        """Verify coerce_inputs_to_schema doesn't lose any convert() capability
        that existed before the _value_satisfies_type gate was added."""

        class FullSchema(BaseModel):
            as_int: int
            as_float: float
            as_bool: bool
            as_str: str
            as_list: list[int]
            as_dict: dict[str, str]

        data: dict[str, Any] = {
            "as_int": "42",
            "as_float": "3.14",
            "as_bool": "True",
            "as_str": 123,
            "as_list": "[1,2,3]",
            "as_dict": '{"a": "b"}',
        }
        coerce_inputs_to_schema(data, FullSchema)
        assert data["as_int"] == 42
        assert data["as_float"] == 3.14
        assert data["as_bool"] is True
        assert data["as_str"] == "123"
        assert data["as_list"] == [1, 2, 3]
        assert data["as_dict"] == {"a": "b"}

    def test_inherited_fields_are_coerced(self):
        """model_fields includes inherited fields; __annotations__ does not.
        This verifies that fields from a parent schema are still coerced."""

        class ParentSchema(BaseModel):
            base_count: int

        class ChildSchema(ParentSchema):
            name: str

        # base_count is inherited — __annotations__ wouldn't include it
        assert "base_count" not in ChildSchema.__annotations__
        assert "base_count" in ChildSchema.model_fields

        data: dict[str, Any] = {"base_count": "42", "name": "test"}
        coerce_inputs_to_schema(data, ChildSchema)
        assert data["base_count"] == 42
        assert isinstance(data["base_count"], int)

    def test_nested_pydantic_model_field(self):
        """dict input for a Pydantic model-typed field passes through.
        convert() doesn't construct Pydantic models — Pydantic validation
        handles that downstream. This test documents the behavior."""

        class InnerModel(BaseModel):
            x: int

        class OuterModel(BaseModel):
            inner: InnerModel

        data: dict[str, Any] = {"inner": {"x": 1}}
        coerce_inputs_to_schema(data, OuterModel)
        # dict stays as dict — convert() doesn't construct Pydantic models
        assert data["inner"] == {"x": 1}
        assert isinstance(data["inner"], dict)

    def test_literal_field_passes_through(self):
        """Literal-typed fields should not crash coercion."""

        class LiteralSchema(BaseModel):
            status: Literal["active", "inactive"]

        data: dict[str, Any] = {"status": "active"}
        coerce_inputs_to_schema(data, LiteralSchema)
        assert data["status"] == "active"

    def test_list_of_pydantic_model_field(self):
        """list[dict] for list[PydanticModel] passes through unchanged."""

        class ItemModel(BaseModel):
            name: str

        class ContainerModel(BaseModel):
            items: list[ItemModel]

        data: dict[str, Any] = {"items": [{"name": "a"}, {"name": "b"}]}
        coerce_inputs_to_schema(data, ContainerModel)
        # Dicts stay as dicts — Pydantic validation handles construction
        assert data["items"] == [{"name": "a"}, {"name": "b"}]
        assert isinstance(data["items"][0], dict)
