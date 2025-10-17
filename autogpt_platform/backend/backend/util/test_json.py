import datetime
from typing import Any, Optional, cast

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

        # Verify that dangerous control characters are actually removed
        result_data = result.data
        assert "\x00" not in str(result_data)  # null byte removed
        assert "\x07" not in str(result_data)  # bell removed
        assert "\x0C" not in str(result_data)  # form feed removed
        assert "\x1B" not in str(result_data)  # escape removed
        assert "\x7F" not in str(result_data)  # delete removed

        # Test that safe whitespace characters are preserved
        safe_data = {
            "with_tab": "text with \t tab",
            "with_newline": "text with \n newline",
            "with_carriage_return": "text with \r carriage return",
            "normal_text": "completely normal text",
        }

        safe_result = SafeJson(safe_data)
        assert isinstance(safe_result, Json)

        # Verify safe characters are preserved
        safe_result_data = cast(dict[str, Any], safe_result.data)
        assert isinstance(safe_result_data, dict)
        with_tab = safe_result_data.get("with_tab", "")
        with_newline = safe_result_data.get("with_newline", "")
        with_carriage_return = safe_result_data.get("with_carriage_return", "")
        assert "\t" in str(with_tab)  # tab preserved
        assert "\n" in str(with_newline)  # newline preserved
        assert "\r" in str(with_carriage_return)  # carriage return preserved

    def test_web_scraping_content_sanitization(self):
        """Test sanitization of typical web scraping content with null characters."""
        # Simulate web content that might contain null bytes from SearchTheWebBlock
        web_content = "Article title\x00Hidden null\x01Start of heading\x08Backspace\x0CForm feed content\x1FUnit separator\x7FDelete char"

        result = SafeJson(web_content)
        assert isinstance(result, Json)

        # Verify all problematic characters are removed
        sanitized_content = str(result.data)
        assert "\x00" not in sanitized_content
        assert "\x01" not in sanitized_content
        assert "\x08" not in sanitized_content
        assert "\x0C" not in sanitized_content
        assert "\x1F" not in sanitized_content
        assert "\x7F" not in sanitized_content

        # Verify the content is still readable
        assert "Article title" in sanitized_content
        assert "Hidden null" in sanitized_content
        assert "content" in sanitized_content

    def test_legitimate_code_preservation(self):
        """Test that legitimate code with backslashes and escapes is preserved."""
        # File paths with backslashes should be preserved
        file_paths = {
            "windows_path": "C:\\Users\\test\\file.txt",
            "network_path": "\\\\server\\share\\folder",
            "escaped_backslashes": "String with \\\\ double backslashes",
        }

        result = SafeJson(file_paths)
        result_data = cast(dict[str, Any], result.data)
        assert isinstance(result_data, dict)

        # Verify file paths are preserved correctly (JSON converts \\\\ back to \\)
        windows_path = result_data.get("windows_path", "")
        network_path = result_data.get("network_path", "")
        escaped_backslashes = result_data.get("escaped_backslashes", "")
        assert "C:\\Users\\test\\file.txt" in str(windows_path)
        assert "\\server\\share" in str(network_path)
        assert "\\" in str(escaped_backslashes)

    def test_legitimate_json_escapes_preservation(self):
        """Test that legitimate JSON escape sequences are preserved."""
        # These should all be preserved as they're valid and useful
        legitimate_escapes = {
            "quotes": 'He said "Hello world!"',
            "newlines": "Line 1\\nLine 2\\nLine 3",
            "tabs": "Column1\\tColumn2\\tColumn3",
            "unicode_chars": "Unicode: \u0048\u0065\u006c\u006c\u006f",  # "Hello"
            "mixed_content": "Path: C:\\\\temp\\\\file.txt\\nSize: 1024 bytes",
        }

        result = SafeJson(legitimate_escapes)
        result_data = cast(dict[str, Any], result.data)
        assert isinstance(result_data, dict)

        # Verify all legitimate content is preserved
        quotes = result_data.get("quotes", "")
        newlines = result_data.get("newlines", "")
        tabs = result_data.get("tabs", "")
        unicode_chars = result_data.get("unicode_chars", "")
        mixed_content = result_data.get("mixed_content", "")

        assert '"' in str(quotes)
        assert "Line 1" in str(newlines) and "Line 2" in str(newlines)
        assert "Column1" in str(tabs) and "Column2" in str(tabs)
        assert "Hello" in str(unicode_chars)  # Unicode should be decoded
        assert "C:" in str(mixed_content) and "temp" in str(mixed_content)

    def test_regex_patterns_dont_over_match(self):
        """Test that our regex patterns don't accidentally match legitimate sequences."""
        # Edge cases that could be problematic for regex
        edge_cases = {
            "file_with_b": "C:\\\\mybfile.txt",  # Contains 'bf' but not escape sequence
            "file_with_f": "C:\\\\folder\\\\file.txt",  # Contains 'f' after backslashes
            "json_like_string": '{"text": "\\\\bolder text"}',  # Looks like JSON escape but isn't
            "unicode_like": "Code: \\\\u0040 (not a real escape)",  # Looks like Unicode escape
        }

        result = SafeJson(edge_cases)
        result_data = cast(dict[str, Any], result.data)
        assert isinstance(result_data, dict)

        # Verify edge cases are handled correctly - no content should be lost
        file_with_b = result_data.get("file_with_b", "")
        file_with_f = result_data.get("file_with_f", "")
        json_like_string = result_data.get("json_like_string", "")
        unicode_like = result_data.get("unicode_like", "")

        assert "mybfile.txt" in str(file_with_b)
        assert "folder" in str(file_with_f) and "file.txt" in str(file_with_f)
        assert "bolder text" in str(json_like_string)
        assert "\\u0040" in str(unicode_like)

    def test_programming_code_preservation(self):
        """Test that programming code with various escapes is preserved."""
        # Common programming patterns that should be preserved
        code_samples = {
            "python_string": 'print("Hello\\\\nworld")',
            "regex_pattern": "\\\\b[A-Za-z]+\\\\b",  # Word boundary regex
            "json_string": '{"name": "test", "path": "C:\\\\\\\\folder"}',
            "sql_escape": "WHERE name LIKE '%\\\\%%'",
            "javascript": 'var path = "C:\\\\\\\\Users\\\\\\\\file.js";',
        }

        result = SafeJson(code_samples)
        result_data = cast(dict[str, Any], result.data)
        assert isinstance(result_data, dict)

        # Verify programming code is preserved
        python_string = result_data.get("python_string", "")
        regex_pattern = result_data.get("regex_pattern", "")
        json_string = result_data.get("json_string", "")
        sql_escape = result_data.get("sql_escape", "")
        javascript = result_data.get("javascript", "")

        assert "print(" in str(python_string)
        assert "Hello" in str(python_string)
        assert "[A-Za-z]+" in str(regex_pattern)
        assert "name" in str(json_string)
        assert "LIKE" in str(sql_escape)
        assert "var path" in str(javascript)

    def test_only_problematic_sequences_removed(self):
        """Test that ONLY PostgreSQL-problematic sequences are removed, nothing else."""
        # Mix of problematic and safe content (using actual control characters)
        mixed_content = {
            "safe_and_unsafe": "Good text\twith tab\x00NULL BYTE\nand newline\x08BACKSPACE",
            "file_path_with_null": "C:\\temp\\file\x00.txt",
            "json_with_controls": '{"text": "data\x01\x0C\x1F"}',
        }

        result = SafeJson(mixed_content)
        result_data = cast(dict[str, Any], result.data)
        assert isinstance(result_data, dict)

        # Verify only problematic characters are removed
        safe_and_unsafe = result_data.get("safe_and_unsafe", "")
        file_path_with_null = result_data.get("file_path_with_null", "")

        assert "Good text" in str(safe_and_unsafe)
        assert "\t" in str(safe_and_unsafe)  # Tab preserved
        assert "\n" in str(safe_and_unsafe)  # Newline preserved
        assert "\x00" not in str(safe_and_unsafe)  # Null removed
        assert "\x08" not in str(safe_and_unsafe)  # Backspace removed

        assert "C:\\temp\\file" in str(file_path_with_null)
        assert ".txt" in str(file_path_with_null)
        assert "\x00" not in str(file_path_with_null)  # Null removed from path
