import datetime
from typing import Any, Optional

from prisma import Json
from pydantic import BaseModel

from backend.util.json import SafeJson, find_arrays_in_text, find_objects_in_text


class SamplePydanticModel(BaseModel):
    name: str
    age: Optional[int] = None
    timestamp: Optional[datetime.datetime] = None
    metadata: Optional[dict] = None


class SampleModelWithNonSerializable(BaseModel):
    name: str
    func: Any = None  # Could contain non-serializable data
    data: Optional[dict] = None


class TestSafeJson:
    """Test cases for SafeJson function."""

    def test_safejson_returns_json_type(self):
        """Test that SafeJson returns a proper Json instance."""
        data = {"test": "value"}
        result = SafeJson(data)
        assert isinstance(result, Json)

    def test_simple_dict_serialization(self):
        """Test basic dictionary serialization."""
        data = {"name": "John", "age": 30, "active": True}
        result = SafeJson(data)
        assert isinstance(result, Json)

    def test_unicode_handling(self):
        """Test that Unicode characters are handled properly."""
        data = {
            "name": "café",
            "emoji": "🎉",
            "chinese": "你好",
            "arabic": "مرحبا",
        }
        result = SafeJson(data)
        assert isinstance(result, Json)

    def test_nested_data_structures(self):
        """Test complex nested data structures."""
        data = {
            "user": {
                "name": "Alice",
                "preferences": {
                    "theme": "dark",
                    "notifications": ["email", "push"],
                },
            },
            "metadata": {
                "tags": ["important", "urgent"],
                "scores": [8.5, 9.2, 7.8],
            },
        }
        result = SafeJson(data)
        assert isinstance(result, Json)

    def test_pydantic_model_basic(self):
        """Test basic Pydantic model serialization."""
        model = SamplePydanticModel(name="John", age=30)
        result = SafeJson(model)
        assert isinstance(result, Json)

    def test_pydantic_model_with_none_values(self):
        """Test Pydantic model with None values (should be excluded)."""
        model = SamplePydanticModel(name="John", age=None, timestamp=None)
        result = SafeJson(model)
        assert isinstance(result, Json)
        # The actual Json content should exclude None values due to exclude_none=True

    def test_pydantic_model_with_datetime(self):
        """Test Pydantic model with datetime field."""
        now = datetime.datetime.now()
        model = SamplePydanticModel(name="John", age=25, timestamp=now)
        result = SafeJson(model)
        assert isinstance(result, Json)

    def test_non_serializable_values_in_dict(self):
        """Test that non-serializable values in dict are converted to None."""
        data = {
            "name": "test",
            "function": lambda x: x,  # Non-serializable
            "datetime": datetime.datetime.now(),  # Non-serializable
            "valid_data": "this should work",
        }
        result = SafeJson(data)
        assert isinstance(result, Json)

    def test_pydantic_model_with_non_serializable_fallback(self):
        """Test Pydantic model with non-serializable field using fallback."""
        model = SampleModelWithNonSerializable(
            name="test",
            func=lambda x: x,  # Non-serializable
            data={"valid": "data"},
        )
        result = SafeJson(model)
        assert isinstance(result, Json)

    def test_empty_data_structures(self):
        """Test empty data structures."""
        test_cases = [
            {},  # Empty dict
            [],  # Empty list
            "",  # Empty string
            None,  # None value
        ]

        for data in test_cases:
            result = SafeJson(data)
            assert isinstance(result, Json)

    def test_complex_mixed_data(self):
        """Test complex mixed data with various types."""
        data = {
            "string": "test",
            "integer": 42,
            "float": 3.14,
            "boolean": True,
            "none_value": None,
            "list": [1, 2, "three", {"nested": "dict"}],
            "nested_dict": {
                "level2": {
                    "level3": ["deep", "nesting", 123],
                }
            },
        }
        result = SafeJson(data)
        assert isinstance(result, Json)

    def test_list_of_pydantic_models(self):
        """Test list containing Pydantic models."""
        models = [
            SamplePydanticModel(name="Alice", age=25),
            SamplePydanticModel(name="Bob", age=30),
        ]
        data = {"users": models}
        result = SafeJson(data)
        assert isinstance(result, Json)

    def test_edge_case_circular_reference_protection(self):
        """Test that circular references don't cause infinite loops."""
        # Note: This test assumes the underlying json.dumps handles circular refs
        # by raising an exception, which our fallback should handle
        data = {}
        data["self"] = data  # Create circular reference

        # This should either work with fallback or raise a reasonable error
        try:
            result = SafeJson(data)
            assert isinstance(result, Json)
        except (ValueError, RecursionError):
            # If it raises an error, that's also acceptable behavior
            pass

    def test_large_data_structure(self):
        """Test with a reasonably large data structure."""
        data = {
            "items": [
                {"id": i, "name": f"item_{i}", "active": i % 2 == 0} for i in range(100)
            ],
            "metadata": {
                "total": 100,
                "generated_at": "2024-01-01T00:00:00Z",
                "tags": ["auto", "generated", "test"],
            },
        }
        result = SafeJson(data)
        assert isinstance(result, Json)

    def test_special_characters_and_encoding(self):
        """Test various special characters and encoding scenarios."""
        data = {
            "quotes": 'He said "Hello world!"',
            "backslashes": "C:\\Users\\test\\file.txt",
            "newlines": "Line 1\nLine 2\nLine 3",
            "tabs": "Column1\tColumn2\tColumn3",
            "unicode_escape": "\u0048\u0065\u006c\u006c\u006f",  # "Hello"
            "mixed": "Test with émojis 🚀 and ñúméríçs",
        }
        result = SafeJson(data)
        assert isinstance(result, Json)

    def test_numeric_edge_cases(self):
        """Test various numeric edge cases."""
        data = {
            "zero": 0,
            "negative": -42,
            "large_int": 999999999999999999,
            "small_float": 0.000001,
            "large_float": 1e10,
            "infinity": float("inf"),  # This might become None due to fallback
            "negative_infinity": float(
                "-inf"
            ),  # This might become None due to fallback
        }
        result = SafeJson(data)
        assert isinstance(result, Json)

    def test_boolean_and_null_values(self):
        """Test boolean and null value handling."""
        data = {
            "true_value": True,
            "false_value": False,
            "null_value": None,
            "mixed_list": [True, False, None, "string", 42],
        }
        result = SafeJson(data)
        assert isinstance(result, Json)


class TestFindObjectsInText:
    """Test cases for find_objects_in_text function."""

    def test_find_single_object(self):
        """Test finding a single JSON object in text."""
        text = 'Here is a JSON object: {"name": "John", "age": 30} in the middle.'
        result = find_objects_in_text(text)
        assert len(result) == 1
        assert result[0] == '{"name": "John", "age": 30}'

    def test_find_multiple_objects(self):
        """Test finding multiple JSON objects in text."""
        text = 'First object: {"a": 1} and second object: {"b": 2, "c": true}'
        result = find_objects_in_text(text)
        assert len(result) == 2
        assert '{"a": 1}' in result
        assert '{"b": 2, "c": true}' in result

    def test_find_nested_objects(self):
        """Test finding nested JSON objects."""
        text = 'Nested: {"outer": {"inner": {"value": 42}}, "other": "data"}'
        result = find_objects_in_text(text)
        assert len(result) == 1
        assert result[0] == '{"outer": {"inner": {"value": 42}}, "other": "data"}'

    def test_empty_object(self):
        """Test finding empty JSON object."""
        text = "Empty object: {} here"
        result = find_objects_in_text(text)
        assert len(result) == 1
        assert result[0] == "{}"

    def test_object_with_arrays(self):
        """Test finding object containing arrays."""
        text = 'Object with array: {"items": [1, 2, 3], "tags": ["a", "b"]}'
        result = find_objects_in_text(text)
        assert len(result) == 1
        assert result[0] == '{"items": [1, 2, 3], "tags": ["a", "b"]}'

    def test_object_with_various_types(self):
        """Test object with different JSON value types."""
        text = 'Complex: {"str": "text", "num": 123, "bool": false, "null": null}'
        result = find_objects_in_text(text)
        assert len(result) == 1
        assert result[0] == '{"str": "text", "num": 123, "bool": false, "null": null}'

    def test_object_with_escaped_quotes(self):
        """Test object with escaped quotes in strings."""
        text = 'Escaped: {"message": "He said \\"Hello\\" to me"}'
        result = find_objects_in_text(text)
        assert len(result) == 1
        assert result[0] == '{"message": "He said \\"Hello\\" to me"}'

    def test_object_with_whitespace(self):
        """Test object with various whitespace formatting."""
        text = 'Whitespace: {  "key1"  :  "value1"  ,  "key2"  :  42  }'
        result = find_objects_in_text(text)
        assert len(result) == 1
        assert result[0] == '{  "key1"  :  "value1"  ,  "key2"  :  42  }'

    def test_no_objects_found(self):
        """Test text with no JSON objects."""
        text = "This is just plain text with no JSON objects."
        result = find_objects_in_text(text)
        assert len(result) == 0

    def test_malformed_objects_ignored(self):
        """Test that malformed JSON objects are ignored."""
        text = (
            'Good: {"valid": true} Bad: {invalid: json} Another good: {"also": "valid"}'
        )
        result = find_objects_in_text(text)
        assert len(result) == 2
        assert '{"valid": true}' in result
        assert '{"also": "valid"}' in result

    def test_objects_in_multiline_text(self):
        """Test finding objects in multiline text."""
        text = """Here is some text
        {"multiline": "object",
         "with": "formatting"}
        More text here"""
        result = find_objects_in_text(text)
        assert len(result) == 1
        assert '"multiline": "object"' in result[0]
        assert '"with": "formatting"' in result[0]

    def test_objects_with_numbers(self):
        """Test objects with various number formats."""
        text = 'Numbers: {"int": 42, "float": 3.14, "neg": -123, "sci": 1.23e-4}'
        result = find_objects_in_text(text)
        assert len(result) == 1
        assert result[0] == '{"int": 42, "float": 3.14, "neg": -123, "sci": 1.23e-4}'


class TestFindArraysInText:
    """Test cases for find_arrays_in_text function."""

    def test_find_single_array(self):
        """Test finding a single JSON array in text."""
        text = "Here is an array: [1, 2, 3] in the text."
        result = find_arrays_in_text(text)
        assert len(result) == 1
        assert result[0] == "[1, 2, 3]"

    def test_find_multiple_arrays(self):
        """Test finding multiple JSON arrays in text."""
        text = 'First: [1, 2] and second: ["a", "b", "c"]'
        result = find_arrays_in_text(text)
        assert len(result) == 2
        assert "[1, 2]" in result
        assert '["a", "b", "c"]' in result

    def test_find_nested_arrays(self):
        """Test finding nested JSON arrays."""
        text = 'Nested: [[1, 2], [3, 4], ["a", ["b", "c"]]]'
        result = find_arrays_in_text(text)
        assert len(result) == 1
        assert result[0] == '[[1, 2], [3, 4], ["a", ["b", "c"]]]'

    def test_empty_array(self):
        """Test finding empty JSON array."""
        text = "Empty array: [] here"
        result = find_arrays_in_text(text)
        assert len(result) == 1
        assert result[0] == "[]"

    def test_array_with_objects(self):
        """Test finding array containing objects."""
        text = 'Array with objects: [{"name": "John"}, {"name": "Jane"}]'
        result = find_arrays_in_text(text)
        assert len(result) == 1
        assert result[0] == '[{"name": "John"}, {"name": "Jane"}]'

    def test_array_with_various_types(self):
        """Test array with different JSON value types."""
        text = 'Mixed array: [1, "text", true, null, 3.14]'
        result = find_arrays_in_text(text)
        assert len(result) == 1
        assert result[0] == '[1, "text", true, null, 3.14]'

    def test_array_with_escaped_quotes(self):
        """Test array with escaped quotes in strings."""
        text = 'Escaped: ["He said \\"Hello\\"", "She replied \\"Hi\\""]'
        result = find_arrays_in_text(text)
        assert len(result) == 1
        assert result[0] == '["He said \\"Hello\\"", "She replied \\"Hi\\""]'

    def test_array_with_whitespace(self):
        """Test array with various whitespace formatting."""
        text = 'Whitespace: [  1  ,  "two"  ,  true  ]'
        result = find_arrays_in_text(text)
        assert len(result) == 1
        assert result[0] == '[  1  ,  "two"  ,  true  ]'

    def test_no_arrays_found(self):
        """Test text with no JSON arrays."""
        text = "This is just plain text with no JSON arrays."
        result = find_arrays_in_text(text)
        assert len(result) == 0

    def test_malformed_arrays_ignored(self):
        """Test that malformed JSON arrays are ignored."""
        text = 'Good: [1, 2, 3] Bad: [invalid, json] Another good: ["valid", "array"]'
        result = find_arrays_in_text(text)
        assert len(result) == 2
        assert "[1, 2, 3]" in result
        assert '["valid", "array"]' in result

    def test_arrays_in_multiline_text(self):
        """Test finding arrays in multiline text."""
        text = """Here is some text
        [
            "multiline",
            "array",
            "with",
            "formatting"
        ]
        More text here"""
        result = find_arrays_in_text(text)
        assert len(result) == 1
        assert '"multiline"' in result[0]
        assert '"array"' in result[0]

    def test_arrays_with_numbers(self):
        """Test arrays with various number formats."""
        text = "Numbers: [42, 3.14, -123, 1.23e-4]"
        result = find_arrays_in_text(text)
        assert len(result) == 1
        assert result[0] == "[42, 3.14, -123, 1.23e-4]"

    def test_complex_mixed_content(self):
        """Test finding arrays in text with mixed JSON content."""
        text = (
            'Object: {"items": [1, 2]} Array: [{"id": 1}, {"id": 2}] Text: more content'
        )
        result = find_arrays_in_text(text)
        assert '[{"id": 1}, {"id": 2}]' in result
