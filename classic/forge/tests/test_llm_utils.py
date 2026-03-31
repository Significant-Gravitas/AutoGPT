"""Tests for LLM provider utility functions."""

import pytest

from forge.llm.providers.schema import (
    AssistantFunctionCall,
    AssistantToolCall,
    CompletionModelFunction,
)
from forge.llm.providers.utils import InvalidFunctionCallError, validate_tool_calls
from forge.models.json_schema import JSONSchema


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def search_function():
    return CompletionModelFunction(
        name="web_search",
        description="Search the web",
        parameters={
            "query": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="Search query",
                required=True,
            ),
        },
    )


@pytest.fixture
def write_function():
    return CompletionModelFunction(
        name="write_file",
        description="Write content to a file",
        parameters={
            "path": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="File path",
                required=True,
            ),
            "content": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="File content",
                required=True,
            ),
        },
    )


@pytest.fixture
def optional_param_function():
    return CompletionModelFunction(
        name="list_files",
        description="List files in a directory",
        parameters={
            "path": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="Directory path",
                required=True,
            ),
            "recursive": JSONSchema(
                type=JSONSchema.Type.BOOLEAN,
                description="Whether to recurse",
                required=False,
            ),
        },
    )


def _make_tool_call(name: str, arguments: dict, call_id: str = "call_1"):
    return AssistantToolCall(
        id=call_id,
        type="function",
        function=AssistantFunctionCall(name=name, arguments=arguments),
    )


# ---------------------------------------------------------------------------
# InvalidFunctionCallError
# ---------------------------------------------------------------------------
class TestInvalidFunctionCallError:
    def test_str_representation(self):
        err = InvalidFunctionCallError(
            name="web_search",
            arguments={"query": "test"},
            message="Missing required param",
        )
        assert "web_search" in str(err)
        assert "Missing required param" in str(err)

    def test_attributes(self):
        err = InvalidFunctionCallError(
            name="fn", arguments={"a": 1}, message="bad call"
        )
        assert err.name == "fn"
        assert err.arguments == {"a": 1}
        assert err.message == "bad call"

    def test_is_exception(self):
        err = InvalidFunctionCallError(name="fn", arguments={}, message="err")
        assert isinstance(err, Exception)


# ---------------------------------------------------------------------------
# validate_tool_calls
# ---------------------------------------------------------------------------
class TestValidateToolCalls:
    def test_valid_call_returns_empty_list(self, search_function):
        tc = _make_tool_call("web_search", {"query": "test"})
        errors = validate_tool_calls([tc], [search_function])
        assert errors == []

    def test_unknown_function_returns_error(self, search_function):
        tc = _make_tool_call("unknown_func", {"query": "test"})
        errors = validate_tool_calls([tc], [search_function])
        assert len(errors) == 1
        assert "Unknown function" in errors[0].message
        assert errors[0].name == "unknown_func"

    def test_multiple_valid_calls(self, search_function, write_function):
        calls = [
            _make_tool_call("web_search", {"query": "test"}, "call_1"),
            _make_tool_call(
                "write_file", {"path": "/tmp/f", "content": "data"}, "call_2"
            ),
        ]
        errors = validate_tool_calls(calls, [search_function, write_function])
        assert errors == []

    def test_empty_tool_calls_returns_empty(self, search_function):
        errors = validate_tool_calls([], [search_function])
        assert errors == []

    def test_mixed_valid_and_invalid(self, search_function, write_function):
        calls = [
            _make_tool_call("web_search", {"query": "ok"}, "call_1"),
            _make_tool_call("nonexistent", {"x": 1}, "call_2"),
        ]
        errors = validate_tool_calls(calls, [search_function, write_function])
        assert len(errors) == 1
        assert errors[0].name == "nonexistent"

    def test_valid_call_with_optional_param_omitted(self, optional_param_function):
        tc = _make_tool_call("list_files", {"path": "/tmp"})
        errors = validate_tool_calls([tc], [optional_param_function])
        assert errors == []

    def test_valid_call_with_optional_param_included(self, optional_param_function):
        tc = _make_tool_call("list_files", {"path": "/tmp", "recursive": True})
        errors = validate_tool_calls([tc], [optional_param_function])
        assert errors == []

    def test_preserves_arguments_in_error(self, search_function):
        bad_args = {"wrong_key": "value"}
        tc = _make_tool_call("unknown_func", bad_args)
        errors = validate_tool_calls([tc], [search_function])
        assert errors[0].arguments == bad_args

    def test_multiple_unknown_functions(self, search_function):
        calls = [
            _make_tool_call("fake1", {}, "call_1"),
            _make_tool_call("fake2", {}, "call_2"),
        ]
        errors = validate_tool_calls(calls, [search_function])
        assert len(errors) == 2
        names = {e.name for e in errors}
        assert names == {"fake1", "fake2"}
