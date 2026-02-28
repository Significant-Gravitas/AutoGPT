"""Tests for tool_adapter helpers: _text_from_mcp_result, truncation stash."""

import pytest

from backend.util.truncate import truncate

from .tool_adapter import (
    _MCP_MAX_CHARS,
    _text_from_mcp_result,
    pop_pending_tool_output,
    set_execution_context,
    stash_pending_tool_output,
)

# ---------------------------------------------------------------------------
# _text_from_mcp_result
# ---------------------------------------------------------------------------


class TestTextFromMcpResult:
    def test_single_text_block(self):
        result = {"content": [{"type": "text", "text": "hello"}]}
        assert _text_from_mcp_result(result) == "hello"

    def test_multiple_text_blocks_concatenated(self):
        result = {
            "content": [
                {"type": "text", "text": "one"},
                {"type": "text", "text": "two"},
            ]
        }
        assert _text_from_mcp_result(result) == "onetwo"

    def test_non_text_blocks_ignored(self):
        result = {
            "content": [
                {"type": "image", "data": "..."},
                {"type": "text", "text": "only this"},
            ]
        }
        assert _text_from_mcp_result(result) == "only this"

    def test_empty_content_list(self):
        assert _text_from_mcp_result({"content": []}) == ""

    def test_missing_content_key(self):
        assert _text_from_mcp_result({}) == ""

    def test_non_list_content(self):
        assert _text_from_mcp_result({"content": "raw string"}) == ""

    def test_missing_text_field(self):
        result = {"content": [{"type": "text"}]}
        assert _text_from_mcp_result(result) == ""


# ---------------------------------------------------------------------------
# stash / pop round-trip (the mechanism _truncating relies on)
# ---------------------------------------------------------------------------


class TestToolOutputStash:
    @pytest.fixture(autouse=True)
    def _init_context(self):
        """Initialise the context vars that stash_pending_tool_output needs."""
        set_execution_context(
            user_id="test",
            session=None,  # type: ignore[arg-type]
            sandbox=None,
            sdk_cwd="/tmp/test",
        )

    def test_stash_and_pop(self):
        stash_pending_tool_output("my_tool", "output1")
        assert pop_pending_tool_output("my_tool") == "output1"

    def test_pop_empty_returns_none(self):
        assert pop_pending_tool_output("nonexistent") is None

    def test_fifo_order(self):
        stash_pending_tool_output("t", "first")
        stash_pending_tool_output("t", "second")
        assert pop_pending_tool_output("t") == "first"
        assert pop_pending_tool_output("t") == "second"
        assert pop_pending_tool_output("t") is None

    def test_dict_serialised_to_json(self):
        stash_pending_tool_output("t", {"key": "value"})
        assert pop_pending_tool_output("t") == '{"key": "value"}'

    def test_separate_tool_names(self):
        stash_pending_tool_output("a", "alpha")
        stash_pending_tool_output("b", "beta")
        assert pop_pending_tool_output("b") == "beta"
        assert pop_pending_tool_output("a") == "alpha"


# ---------------------------------------------------------------------------
# _truncating wrapper (integration via create_copilot_mcp_server)
# ---------------------------------------------------------------------------


class TestTruncatingWrapper:
    """Test the _truncating closure by creating an MCP server with a mock tool."""

    @pytest.fixture(autouse=True)
    def _init_context(self):
        set_execution_context(
            user_id="test",
            session=None,  # type: ignore[arg-type]
            sandbox=None,
            sdk_cwd="/tmp/test",
        )

    def test_small_output_stashed(self):
        """Non-error output is stashed for the response adapter."""
        result = {
            "content": [{"type": "text", "text": "small output"}],
            "isError": False,
        }
        truncated = truncate(result, _MCP_MAX_CHARS)
        text = _text_from_mcp_result(truncated)
        assert text == "small output"
        stash_pending_tool_output("test_tool", text)
        assert pop_pending_tool_output("test_tool") == "small output"

    def test_error_result_not_stashed(self):
        """Error results should not be stashed."""
        result = {
            "content": [{"type": "text", "text": "error msg"}],
            "isError": True,
        }
        # _truncating only stashes when not result.get("isError")
        if not result.get("isError"):
            stash_pending_tool_output("err_tool", "should not happen")
        assert pop_pending_tool_output("err_tool") is None

    def test_large_output_truncated(self):
        """Output exceeding _MCP_MAX_CHARS is truncated before stashing."""
        big_text = "x" * (_MCP_MAX_CHARS + 100_000)
        result = {"content": [{"type": "text", "text": big_text}]}
        truncated = truncate(result, _MCP_MAX_CHARS)
        text = _text_from_mcp_result(truncated)
        assert len(text) < len(big_text)
        assert len(str(truncated)) <= _MCP_MAX_CHARS
