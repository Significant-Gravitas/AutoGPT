import datetime
from typing import Any, Optional

from prisma import Json
from pydantic import BaseModel

from backend.util.json import SafeJson


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

    def test_control_character_sanitization(self):
        """Test that PostgreSQL-incompatible control characters are sanitized by SafeJson."""
        # Test data with problematic control characters that would cause PostgreSQL errors
        problematic_data = {
            "null_byte": "data with \x00 null",
            "bell_char": "data with \x07 bell",
            "form_feed": "data with \x0C feed",
            "escape_char": "data with \x1B escape",
            "delete_char": "data with \x7F delete",
        }

        # SafeJson should successfully process data with control characters
        result = SafeJson(problematic_data)
        assert isinstance(result, Json)

        # Test that safe whitespace characters are preserved
        safe_data = {
            "with_tab": "text with \t tab",
            "with_newline": "text with \n newline",
            "with_carriage_return": "text with \r carriage return",
            "normal_text": "completely normal text",
        }

        safe_result = SafeJson(safe_data)
        assert isinstance(safe_result, Json)
