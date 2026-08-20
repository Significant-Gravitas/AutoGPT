import datetime
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, cast

import jsonschema
import pytest
from prisma import Json
from pydantic import BaseModel

from backend.util import json as json_util
from backend.util.json import SafeJson, validate_with_jsonschema


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

    def test_invalid_escape_error_prevention(self):
        """Test that SafeJson prevents 'Invalid \\escape' errors that occurred in upsert_execution_output."""
        # This reproduces the exact scenario that was causing the error:
        # POST /upsert_execution_output failed: Invalid \escape: line 1 column 36404 (char 36403)

        # Create data with various problematic escape sequences that could cause JSON parsing errors
        problematic_output_data = {
            "web_content": "Article text\x00with null\x01and control\x08chars\x0C\x1F\x7F",
            "file_path": "C:\\Users\\test\\file\x00.txt",
            "json_like_string": '{"text": "data\x00\x08\x1F"}',
            "escaped_sequences": "Text with \\u0000 and \\u0008 sequences",
            "mixed_content": "Normal text\tproperly\nformatted\rwith\x00invalid\x08chars\x1Fmixed",
            "large_text": "A" * 35000
            + "\x00\x08\x1F"
            + "B" * 5000,  # Large text like in the error
        }

        # This should not raise any JSON parsing errors
        result = SafeJson(problematic_output_data)
        assert isinstance(result, Json)

        # Verify the result is a valid Json object that can be safely stored in PostgreSQL
        result_data = cast(dict[str, Any], result.data)
        assert isinstance(result_data, dict)

        # Verify problematic characters are removed but safe content preserved
        web_content = result_data.get("web_content", "")
        file_path = result_data.get("file_path", "")
        large_text = result_data.get("large_text", "")

        # Check that control characters are removed
        assert "\x00" not in str(web_content)
        assert "\x01" not in str(web_content)
        assert "\x08" not in str(web_content)
        assert "\x0C" not in str(web_content)
        assert "\x1F" not in str(web_content)
        assert "\x7F" not in str(web_content)

        # Check that legitimate content is preserved
        assert "Article text" in str(web_content)
        assert "with null" in str(web_content)
        assert "and control" in str(web_content)
        assert "chars" in str(web_content)

        # Check file path handling
        assert "C:\\Users\\test\\file" in str(file_path)
        assert ".txt" in str(file_path)
        assert "\x00" not in str(file_path)

        # Check large text handling (the scenario from the error at char 36403)
        assert len(str(large_text)) > 35000  # Content preserved
        assert "A" * 1000 in str(large_text)  # A's preserved
        assert "B" * 1000 in str(large_text)  # B's preserved
        assert "\x00" not in str(large_text)  # Control chars removed
        assert "\x08" not in str(large_text)
        assert "\x1F" not in str(large_text)

        # Most importantly: ensure the result can be JSON-serialized without errors
        # This would have failed with the old approach
        import json

        json_string = json.dumps(result.data)  # Should not raise "Invalid \escape"
        assert len(json_string) > 0

        # And can be parsed back
        parsed_back = json.loads(json_string)
        assert isinstance(parsed_back, dict)

    def test_dict_containing_pydantic_models(self):
        """Test that dicts containing Pydantic models are properly serialized."""
        # This reproduces the bug where credential_inputs failed
        model1 = SamplePydanticModel(name="Alice", age=30)
        model2 = SamplePydanticModel(name="Bob", age=25)

        data = {
            "user1": model1,
            "user2": model2,
            "regular_data": "test",
        }

        result = SafeJson(data)
        assert isinstance(result, Json)

        # Verify it can be JSON serialized (this was the bug)
        import json

        json_string = json.dumps(result.data)
        assert "Alice" in json_string
        assert "Bob" in json_string

    def test_nested_pydantic_in_dict(self):
        """Test deeply nested Pydantic models in dicts."""
        inner_model = SamplePydanticModel(name="Inner", age=20)
        middle_model = SamplePydanticModel(
            name="Middle", age=30, metadata={"inner": inner_model}
        )

        data = {
            "level1": {
                "level2": {
                    "model": middle_model,
                    "other": "data",
                }
            }
        }

        result = SafeJson(data)
        assert isinstance(result, Json)

        import json

        json_string = json.dumps(result.data)
        assert "Middle" in json_string
        assert "Inner" in json_string

    def test_list_containing_pydantic_models_in_dict(self):
        """Test list of Pydantic models inside a dict."""
        models = [SamplePydanticModel(name=f"User{i}", age=20 + i) for i in range(5)]

        data = {
            "users": models,
            "count": len(models),
        }

        result = SafeJson(data)
        assert isinstance(result, Json)

        import json

        json_string = json.dumps(result.data)
        assert "User0" in json_string
        assert "User4" in json_string

    def test_credentials_meta_input_scenario(self):
        """Test the exact scenario from create_graph_execution that was failing."""

        # Simulate CredentialsMetaInput structure
        class MockCredentialsMetaInput(BaseModel):
            id: str
            title: Optional[str] = None
            provider: str
            type: str

        cred_input = MockCredentialsMetaInput(
            id="test-123", title="Test Credentials", provider="github", type="oauth2"
        )

        # This is how credential_inputs is structured in create_graph_execution
        credential_inputs = {"github_creds": cred_input}

        # This should work without TypeError
        result = SafeJson(credential_inputs)
        assert isinstance(result, Json)

        # Verify it can be JSON serialized
        import json

        json_string = json.dumps(result.data)
        assert "test-123" in json_string
        assert "github" in json_string
        assert "oauth2" in json_string

    def test_mixed_pydantic_and_primitives(self):
        """Test complex mix of Pydantic models and primitive types."""
        model = SamplePydanticModel(name="Test", age=25)

        data = {
            "models": [model, {"plain": "dict"}, "string", 123],
            "nested": {
                "model": model,
                "list": [1, 2, model, 4],
                "plain": "text",
            },
            "plain_list": [1, 2, 3],
        }

        result = SafeJson(data)
        assert isinstance(result, Json)

        import json

        json_string = json.dumps(result.data)
        assert "Test" in json_string
        assert "plain" in json_string

    def test_pydantic_model_with_control_chars_in_dict(self):
        """Test Pydantic model with control chars when nested in dict."""
        model = SamplePydanticModel(
            name="Test\x00User",  # Has null byte
            age=30,
            metadata={"info": "data\x08with\x0Ccontrols"},
        )

        data = {"credential": model}

        result = SafeJson(data)
        assert isinstance(result, Json)

        # Verify control characters are removed
        import json

        json_string = json.dumps(result.data)
        assert "\x00" not in json_string
        assert "\x08" not in json_string
        assert "\x0C" not in json_string
        assert "TestUser" in json_string  # Name preserved minus null byte

    def test_deeply_nested_pydantic_models_control_char_sanitization(self):
        """Test that control characters are sanitized in deeply nested Pydantic models."""

        # Create nested Pydantic models with control characters at different levels
        class InnerModel(BaseModel):
            deep_string: str
            value: int = 42
            metadata: dict = {}

        class MiddleModel(BaseModel):
            middle_string: str
            inner: InnerModel
            data: str

        class OuterModel(BaseModel):
            outer_string: str
            middle: MiddleModel

        # Create test data with control characters at every nesting level
        inner = InnerModel(
            deep_string="Deepest\x00Level\x08Control\x0CChars",  # Multiple control chars at deepest level
            metadata={
                "nested_key": "Nested\x1FValue\x7FDelete"
            },  # Control chars in nested dict
        )

        middle = MiddleModel(
            middle_string="Middle\x01StartOfHeading\x1FUnitSeparator",
            inner=inner,
            data="Some\x0BVerticalTab\x0EShiftOut",
        )

        outer = OuterModel(outer_string="Outer\x00Null\x07Bell", middle=middle)

        # Wrap in a dict with additional control characters
        data = {
            "top_level": "Top\x00Level\x08Backspace",
            "nested_model": outer,
            "list_with_strings": [
                "List\x00Item1",
                "List\x0CItem2\x1F",
                {"dict_in_list": "Dict\x08Value"},
            ],
        }

        # Process with SafeJson
        result = SafeJson(data)
        assert isinstance(result, Json)

        # Verify all control characters are removed at every level
        import json

        json_string = json.dumps(result.data)

        # Check that NO control characters remain anywhere
        control_chars = [
            "\x00",
            "\x01",
            "\x02",
            "\x03",
            "\x04",
            "\x05",
            "\x06",
            "\x07",
            "\x08",
            "\x0B",
            "\x0C",
            "\x0E",
            "\x0F",
            "\x10",
            "\x11",
            "\x12",
            "\x13",
            "\x14",
            "\x15",
            "\x16",
            "\x17",
            "\x18",
            "\x19",
            "\x1A",
            "\x1B",
            "\x1C",
            "\x1D",
            "\x1E",
            "\x1F",
            "\x7F",
        ]

        for char in control_chars:
            assert (
                char not in json_string
            ), f"Control character {repr(char)} found in result"

        # Verify specific sanitized content is present (control chars removed but text preserved)
        result_data = cast(dict[str, Any], result.data)

        # Top level
        assert "TopLevelBackspace" in json_string

        # Outer model level
        assert "OuterNullBell" in json_string

        # Middle model level
        assert "MiddleStartOfHeadingUnitSeparator" in json_string
        assert "SomeVerticalTabShiftOut" in json_string

        # Inner model level (deepest nesting)
        assert "DeepestLevelControlChars" in json_string

        # Nested dict in model
        assert "NestedValueDelete" in json_string

        # List items
        assert "ListItem1" in json_string
        assert "ListItem2" in json_string
        assert "DictValue" in json_string

        # Verify structure is preserved (not just converted to string)
        assert isinstance(result_data, dict)
        assert isinstance(result_data["nested_model"], dict)
        assert isinstance(result_data["nested_model"]["middle"], dict)
        assert isinstance(result_data["nested_model"]["middle"]["inner"], dict)
        assert isinstance(result_data["list_with_strings"], list)

        # Verify specific deep values are accessible and sanitized
        nested_model = cast(dict[str, Any], result_data["nested_model"])
        middle = cast(dict[str, Any], nested_model["middle"])
        inner = cast(dict[str, Any], middle["inner"])

        deep_string = inner["deep_string"]
        assert deep_string == "DeepestLevelControlChars"

        metadata = cast(dict[str, Any], inner["metadata"])
        nested_metadata = metadata["nested_key"]
        assert nested_metadata == "NestedValueDelete"


class TestValidateWithJsonschema:
    """Test cases for validate_with_jsonschema."""

    SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name"],
    }

    def test_valid_data_returns_none(self):
        """Valid data produces no error message."""
        assert (
            validate_with_jsonschema(self.SCHEMA, {"name": "John", "age": 30}) is None
        )

    def test_type_mismatch_returns_error_message(self):
        """A type mismatch produces the underlying jsonschema message."""
        error = validate_with_jsonschema(self.SCHEMA, {"name": 1})
        assert error is not None
        assert "is not of type 'string'" in error

    def test_missing_required_field_returns_error_message(self):
        """A missing required field produces the underlying jsonschema message."""
        error = validate_with_jsonschema(self.SCHEMA, {"age": 30})
        assert error is not None
        assert "'name' is a required property" in error

    def test_repeated_calls_with_one_schema_stay_correct(self):
        """The executor validates the same schema against different data on
        every node execution; repeated calls must not drift."""
        for i in range(5):
            assert validate_with_jsonschema(self.SCHEMA, {"name": f"n{i}"}) is None
            assert validate_with_jsonschema(self.SCHEMA, {"name": i}) is not None

    def test_distinct_schemas_are_not_conflated(self):
        """Two different schemas must keep producing their own verdicts,
        including when the first one is revisited."""
        text_schema = {"type": "object", "properties": {"f": {"type": "string"}}}
        int_schema = {"type": "object", "properties": {"f": {"type": "integer"}}}
        assert validate_with_jsonschema(text_schema, {"f": "text"}) is None
        assert validate_with_jsonschema(int_schema, {"f": "text"}) is not None
        assert validate_with_jsonschema(text_schema, {"f": "text"}) is None

    def test_equal_but_distinct_schema_objects_agree(self):
        """A graph's input schema is rebuilt per execution, so the same schema
        arrives as a fresh object each time."""
        first = {"type": "object", "properties": {"f": {"type": "integer"}}}
        second = {"type": "object", "properties": {"f": {"type": "integer"}}}
        assert validate_with_jsonschema(first, {"f": 1}) is None
        assert validate_with_jsonschema(second, {"f": 1}) is None
        assert validate_with_jsonschema(second, {"f": "x"}) is not None

    def test_malformed_schema_raises_on_every_call(self):
        """`required` with duplicate entries fails the meta-schema check, and
        agent graphs do carry such schemas. The SchemaError must keep escaping,
        not just on the first call."""
        malformed = {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["a", "a"],
        }
        json_util._VALIDATOR_CACHE.clear()
        errors: list[jsonschema.SchemaError] = []
        for _ in range(3):
            with pytest.raises(jsonschema.SchemaError) as exc_info:
                validate_with_jsonschema(malformed, {"a": "x"})
            errors.append(exc_info.value)

        def traceback_depth(error: BaseException) -> int:
            depth = 0
            traceback = error.__traceback__
            while traceback is not None:
                depth += 1
                traceback = traceback.tb_next
            return depth

        # Re-raising one cached exception grows its retained traceback on every
        # call. Keep malformed schemas uncached so each failure remains fresh.
        assert len({id(error) for error in errors}) == len(errors)
        assert len({traceback_depth(error) for error in errors}) == 1
        assert not json_util._VALIDATOR_CACHE

    def test_schema_mutated_in_place_is_not_stale(self):
        """Mutating a schema between calls must change the verdict."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"f": {"type": "string"}},
        }
        assert validate_with_jsonschema(schema, {"f": "text"}) is None
        schema["properties"]["f"] = {"type": "integer"}
        assert validate_with_jsonschema(schema, {"f": "text"}) is not None
        schema["properties"]["f"] = {"type": "string"}
        assert validate_with_jsonschema(schema, {"f": "text"}) is None

    def test_concurrent_compilation_publishes_one_validator(self, monkeypatch):
        """Concurrent misses may compile in parallel but publish one value."""
        schema = {"type": "object", "properties": {"f": {"type": "integer"}}}
        json_util._VALIDATOR_CACHE.clear()
        original_deepcopy = json_util.deepcopy
        compile_barrier = threading.Barrier(2)

        def synchronized_deepcopy(value):
            compile_barrier.wait(timeout=5)
            return original_deepcopy(value)

        monkeypatch.setattr(json_util, "deepcopy", synchronized_deepcopy)
        with ThreadPoolExecutor(max_workers=2) as pool:
            validators = list(pool.map(json_util._compiled_validator, (schema, schema)))

        assert validators[0] is validators[1]

    def test_concurrent_eviction_is_atomic(self, monkeypatch):
        """Two full-cache misses cannot evict the same entry concurrently."""
        cache: OrderedDict[bytes, Any] = OrderedDict({b"old": object()})
        monkeypatch.setattr(json_util, "_VALIDATOR_CACHE", cache)
        monkeypatch.setattr(json_util, "_VALIDATOR_CACHE_MAX_ENTRIES", 1)
        values = (object(), object())

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = (
                pool.submit(json_util._remember, b"new-a", values[0]),
                pool.submit(json_util._remember, b"new-b", values[1]),
            )
            published = tuple(future.result() for future in futures)

        assert published == values
        assert len(cache) == 1

    def test_cache_uses_lru_eviction(self, monkeypatch):
        """A recently reused schema survives the next full-cache insertion."""
        cache: OrderedDict[bytes, Any] = OrderedDict()
        monkeypatch.setattr(json_util, "_VALIDATOR_CACHE", cache)
        monkeypatch.setattr(json_util, "_VALIDATOR_CACHE_MAX_ENTRIES", 2)
        schemas = [
            {
                "type": "object",
                "properties": {"value": {"type": "integer", "minimum": minimum}},
            }
            for minimum in range(3)
        ]

        for schema in schemas[:2]:
            assert validate_with_jsonschema(schema, {"value": 2}) is None
        assert validate_with_jsonschema(schemas[0], {"value": 2}) is None
        assert validate_with_jsonschema(schemas[2], {"value": 2}) is None

        keys = [json_util._schema_cache_key(schema) for schema in schemas]
        assert list(cache) == [keys[0], keys[2]]

    @pytest.mark.parametrize(
        (
            "first_schema",
            "first_data",
            "second_schema",
            "second_data",
            "expected_entries",
        ),
        [
            (
                {
                    "type": "object",
                    "properties": {
                        "value": {"type": "number", "maximum": float("inf")}
                    },
                },
                {"value": 5},
                {
                    "type": "object",
                    "properties": {
                        "value": {"type": "number", "maximum": float("-inf")}
                    },
                },
                {"value": 5},
                0,
            ),
            (
                {"type": "object", "properties": {"value": {"enum": [(1, 2)]}}},
                {"value": [1, 2]},
                {"type": "object", "properties": {"value": {"enum": [[1, 2]]}}},
                {"value": [1, 2]},
                1,
            ),
        ],
    )
    def test_lossy_serializations_bypass_cache(
        self,
        first_schema,
        first_data,
        second_schema,
        second_data,
        expected_entries,
        monkeypatch,
    ):
        """Values that orjson conflates retain jsonschema's exact behavior."""
        cache: OrderedDict[bytes, Any] = OrderedDict()
        monkeypatch.setattr(json_util, "_VALIDATOR_CACHE", cache)

        assert validate_with_jsonschema(
            first_schema, first_data
        ) == self._baseline_outcome(first_schema, first_data)
        assert not cache
        assert validate_with_jsonschema(
            second_schema, second_data
        ) == self._baseline_outcome(second_schema, second_data)
        assert len(cache) == expected_entries

    def test_nonfinite_key_cannot_hide_malformed_schema(self, monkeypatch):
        """A non-finite key cannot collide with an invalid null constraint."""
        cache: OrderedDict[bytes, Any] = OrderedDict()
        monkeypatch.setattr(json_util, "_VALIDATOR_CACHE", cache)
        nonfinite = {
            "type": "object",
            "properties": {"value": {"type": "number", "maximum": float("inf")}},
        }
        malformed = {
            "type": "object",
            "properties": {"value": {"type": "number", "maximum": None}},
        }

        assert validate_with_jsonschema(nonfinite, {"value": 5}) is None
        with pytest.raises(jsonschema.SchemaError):
            validate_with_jsonschema(malformed, {"value": 5})
        assert not cache

    def test_oversized_schema_is_not_retained(self, monkeypatch):
        """Caller-controlled schemas above the key budget compile uncached."""
        cache: OrderedDict[bytes, Any] = OrderedDict()
        monkeypatch.setattr(json_util, "_VALIDATOR_CACHE", cache)
        schema = {
            "type": "object",
            "description": "x" * 20_000,
            "properties": {"value": {"type": "string"}},
        }

        assert validate_with_jsonschema(schema, {"value": "text"}) is None
        assert validate_with_jsonschema(schema, {"value": "text"}) is None
        assert not cache

    @pytest.mark.parametrize(
        ("schema", "data"),
        [
            (
                {
                    "type": "object",
                    "properties": {
                        "value": {"anyOf": [{"type": "string"}, {"type": "integer"}]}
                    },
                },
                {"value": []},
            ),
            (
                {
                    "type": "object",
                    "properties": {
                        "value": {"oneOf": [{"minimum": 5}, {"type": "string"}]}
                    },
                },
                {"value": 3},
            ),
            (
                {
                    "type": "object",
                    "properties": {
                        "nested": {
                            "type": "object",
                            "properties": {
                                "value": {"anyOf": [{"type": "string"}, {"minimum": 5}]}
                            },
                        }
                    },
                },
                {"nested": {"value": 3}},
            ),
        ],
    )
    def test_nested_error_message_matches_jsonschema_exactly(self, schema, data):
        """best_match keeps byte-for-byte parity for combinator errors."""
        assert validate_with_jsonschema(schema, data) == self._baseline_outcome(
            schema, data
        )

    @staticmethod
    def _baseline_outcome(schema: dict[str, Any], data: dict[str, Any]) -> str | None:
        try:
            jsonschema.validate(data, schema)
        except jsonschema.ValidationError as error:
            return str(error)
        return None
