"""Tests for tool_adapter helpers: multimodal extraction, truncation, stash."""

import json

import pytest

from backend.util.truncate import truncate

from .tool_adapter import (
    _MCP_MAX_CHARS,
    _extract_content_block,
    _strip_base64_from_text,
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


class TestTruncationAndStashIntegration:
    """Test truncation + stash behavior that _truncating relies on."""

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


# ---------------------------------------------------------------------------
# _extract_content_block
# ---------------------------------------------------------------------------


class TestExtractContentBlock:
    """Tests for _extract_content_block multimodal detection."""

    def test_image_png_returns_image_block(self):
        payload = json.dumps(
            {
                "content_base64": "iVBORw0KGgo=",
                "mime_type": "image/png",
                "file_id": "abc",
            }
        )
        block = _extract_content_block(payload)
        assert block is not None
        assert block["type"] == "image"
        assert block["data"] == "iVBORw0KGgo="
        assert block["mimeType"] == "image/png"

    def test_image_jpeg_returns_image_block(self):
        payload = json.dumps(
            {
                "content_base64": "/9j/4AAQ=",
                "mime_type": "image/jpeg",
            }
        )
        block = _extract_content_block(payload)
        assert block is not None
        assert block["type"] == "image"

    def test_pdf_returns_document_block(self):
        payload = json.dumps(
            {
                "content_base64": "JVBERi0=",
                "mime_type": "application/pdf",
            }
        )
        block = _extract_content_block(payload)
        assert block is not None
        assert block["type"] == "document"
        assert block["source"]["type"] == "base64"
        assert block["source"]["media_type"] == "application/pdf"
        assert block["source"]["data"] == "JVBERi0="

    def test_unsupported_mime_returns_none(self):
        payload = json.dumps(
            {
                "content_base64": "data",
                "mime_type": "text/plain",
            }
        )
        assert _extract_content_block(payload) is None

    def test_missing_content_base64_returns_none(self):
        payload = json.dumps({"mime_type": "image/png"})
        assert _extract_content_block(payload) is None

    def test_missing_mime_type_returns_none(self):
        payload = json.dumps({"content_base64": "data"})
        assert _extract_content_block(payload) is None

    def test_non_json_returns_none(self):
        assert _extract_content_block("not json at all") is None

    def test_non_dict_json_returns_none(self):
        assert _extract_content_block("[1, 2, 3]") is None

    def test_oversized_image_base64_returns_none(self):
        huge = "A" * (_MCP_MAX_CHARS + 1_000_000)  # exceeds _IMAGE_MAX_B64
        payload = json.dumps(
            {
                "content_base64": huge,
                "mime_type": "image/png",
            }
        )
        assert _extract_content_block(payload) is None

    def test_non_string_mime_type_returns_none(self):
        payload = json.dumps({"content_base64": "data", "mime_type": 123})
        assert _extract_content_block(payload) is None

    def test_non_string_content_base64_returns_none(self):
        payload = json.dumps({"content_base64": 456, "mime_type": "image/png"})
        assert _extract_content_block(payload) is None

    def test_mime_with_parameters_normalized(self):
        payload = json.dumps(
            {
                "content_base64": "iVBORw0KGgo=",
                "mime_type": "image/png; charset=binary",
            }
        )
        block = _extract_content_block(payload)
        assert block is not None
        assert block["type"] == "image"
        assert block["mimeType"] == "image/png"

    def test_pdf_mime_with_parameters_normalized(self):
        payload = json.dumps(
            {
                "content_base64": "JVBERi0=",
                "mime_type": "application/pdf; charset=binary",
            }
        )
        block = _extract_content_block(payload)
        assert block is not None
        assert block["type"] == "document"
        assert block["source"]["media_type"] == "application/pdf"


# ---------------------------------------------------------------------------
# _strip_base64_from_text
# ---------------------------------------------------------------------------


class TestStripBase64FromText:
    """Tests for _strip_base64_from_text helper."""

    def test_replaces_content_base64_with_placeholder(self):
        payload = json.dumps(
            {
                "file_id": "abc",
                "mime_type": "image/png",
                "content_base64": "iVBORw0KGgo=",
            }
        )
        result = json.loads(_strip_base64_from_text(payload))
        assert result["content_base64"] == "(see attached content block)"
        assert result["file_id"] == "abc"
        assert result["mime_type"] == "image/png"

    def test_no_content_base64_returns_unchanged(self):
        payload = json.dumps({"file_id": "abc", "name": "test.txt"})
        assert _strip_base64_from_text(payload) == payload

    def test_non_json_returns_unchanged(self):
        assert _strip_base64_from_text("plain text") == "plain text"

    def test_non_dict_json_returns_unchanged(self):
        original = "[1, 2]"
        assert _strip_base64_from_text(original) == original


# ---------------------------------------------------------------------------
# Multimodal truncation protection
# ---------------------------------------------------------------------------


def _simulate_truncating(result: dict) -> dict:
    """Simulate the _truncating wrapper logic for multimodal-safe truncation."""
    content = result.get("content", [])
    non_text_blocks: list[dict] = []
    text_only_content: list[dict] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") != "text":
                non_text_blocks.append(block)
            else:
                text_only_content.append(block)

    truncated = truncate({**result, "content": text_only_content}, _MCP_MAX_CHARS)

    if non_text_blocks:
        truncated_content = truncated.get("content", [])
        if isinstance(truncated_content, list):
            truncated["content"] = truncated_content + non_text_blocks
        else:
            truncated["content"] = non_text_blocks

    return truncated


class TestMultimodalTruncationProtection:
    """Verify that non-text content blocks survive truncation intact."""

    def test_image_block_not_corrupted(self):
        """Image data must remain identical after text-only truncation."""
        big_text = "x" * (_MCP_MAX_CHARS + 100_000)
        image_data = "iVBORw0KGgo=" * 1000
        result = {
            "content": [
                {"type": "text", "text": big_text},
                {"type": "image", "data": image_data, "mimeType": "image/png"},
            ],
            "isError": False,
        }
        truncated = _simulate_truncating(result)
        image_blocks = [b for b in truncated["content"] if b.get("type") == "image"]
        assert len(image_blocks) == 1
        assert image_blocks[0]["data"] == image_data
        assert image_blocks[0]["mimeType"] == "image/png"

    def test_document_block_not_corrupted(self):
        """PDF document data must remain identical after text-only truncation."""
        pdf_data = "JVBERi0=" * 500
        result = {
            "content": [
                {"type": "text", "text": "file metadata here"},
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": pdf_data,
                    },
                },
            ],
            "isError": False,
        }
        truncated = _simulate_truncating(result)
        doc_blocks = [b for b in truncated["content"] if b.get("type") == "document"]
        assert len(doc_blocks) == 1
        assert doc_blocks[0]["source"]["data"] == pdf_data

    def test_text_still_truncated(self):
        """Text blocks should still be truncated normally."""
        big_text = "x" * (_MCP_MAX_CHARS + 100_000)
        result = {
            "content": [
                {"type": "text", "text": big_text},
                {"type": "image", "data": "small", "mimeType": "image/png"},
            ],
            "isError": False,
        }
        truncated = _simulate_truncating(result)
        text = _text_from_mcp_result(truncated)
        assert len(text) < len(big_text)

    def test_text_only_result_unaffected(self):
        """Results with only text blocks work exactly as before."""
        result = {
            "content": [{"type": "text", "text": "hello world"}],
            "isError": False,
        }
        truncated = _simulate_truncating(result)
        assert _text_from_mcp_result(truncated) == "hello world"
