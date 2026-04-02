"""Tests for BaseTool large-output persistence in execute()."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.tools.base import (
    _LARGE_OUTPUT_THRESHOLD,
    BaseTool,
    _persist_and_summarize,
    _summarize_binary_fields,
)
from backend.copilot.tools.models import ResponseType, ToolResponseBase


class _HugeOutputTool(BaseTool):
    """Fake tool that returns an arbitrarily large output."""

    def __init__(self, output_size: int) -> None:
        self._output_size = output_size

    @property
    def name(self) -> str:
        return "huge_output_tool"

    @property
    def description(self) -> str:
        return "Returns a huge output"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def _execute(self, user_id, session, **kwargs) -> ToolResponseBase:
        return ToolResponseBase(
            type=ResponseType.ERROR,
            message="x" * self._output_size,
        )


# ---------------------------------------------------------------------------
# _persist_and_summarize
# ---------------------------------------------------------------------------


class TestPersistAndSummarize:
    @pytest.mark.asyncio
    async def test_returns_middle_out_preview_with_retrieval_instructions(self):
        raw = "A" * 200_000

        mock_workspace = MagicMock()
        mock_workspace.id = "ws-1"
        mock_db = AsyncMock()
        mock_db.get_or_create_workspace = AsyncMock(return_value=mock_workspace)

        mock_manager = AsyncMock()

        with (
            patch("backend.copilot.tools.base.workspace_db", return_value=mock_db),
            patch(
                "backend.copilot.tools.base.WorkspaceManager",
                return_value=mock_manager,
            ),
        ):
            result = await _persist_and_summarize(raw, "user-1", "session-1", "tc-123")

        assert "<tool-output-truncated" in result
        assert "</tool-output-truncated>" in result
        assert "total_chars=200000" in result
        assert 'workspace_path="tool-outputs/tc-123.json"' in result
        assert "read_workspace_file" in result
        # Middle-out sentinel from truncate()
        assert "omitted" in result
        # Total result is much shorter than the raw output
        assert len(result) < len(raw)

        # Verify write_file was called with full content
        mock_manager.write_file.assert_awaited_once()
        call_kwargs = mock_manager.write_file.call_args
        assert call_kwargs.kwargs["content"] == raw.encode("utf-8")
        assert call_kwargs.kwargs["path"] == "tool-outputs/tc-123.json"

    @pytest.mark.asyncio
    async def test_fallback_on_workspace_error(self):
        """If workspace write fails, return raw output for normal truncation."""
        raw = "B" * 200_000
        mock_db = AsyncMock()
        mock_db.get_or_create_workspace = AsyncMock(side_effect=RuntimeError("boom"))

        with patch("backend.copilot.tools.base.workspace_db", return_value=mock_db):
            result = await _persist_and_summarize(raw, "user-1", "session-1", "tc-fail")

        assert result == raw  # unchanged — fallback to normal truncation


# ---------------------------------------------------------------------------
# BaseTool.execute — integration with persistence
# ---------------------------------------------------------------------------


class TestBaseToolExecuteLargeOutput:
    @pytest.mark.asyncio
    async def test_small_output_not_persisted(self):
        """Outputs under the threshold go through without persistence."""
        tool = _HugeOutputTool(output_size=100)
        session = MagicMock()
        session.session_id = "s-1"

        with patch(
            "backend.copilot.tools.base._persist_and_summarize",
            new_callable=AsyncMock,
        ) as persist_mock:
            result = await tool.execute("user-1", session, "tc-small")
        persist_mock.assert_not_awaited()
        assert "<tool-output-truncated" not in str(result.output)

    @pytest.mark.asyncio
    async def test_large_output_persisted(self):
        """Outputs over the threshold trigger persistence + preview."""
        tool = _HugeOutputTool(output_size=_LARGE_OUTPUT_THRESHOLD + 10_000)
        session = MagicMock()
        session.session_id = "s-1"

        mock_workspace = MagicMock()
        mock_workspace.id = "ws-1"
        mock_db = AsyncMock()
        mock_db.get_or_create_workspace = AsyncMock(return_value=mock_workspace)
        mock_manager = AsyncMock()

        with (
            patch("backend.copilot.tools.base.workspace_db", return_value=mock_db),
            patch(
                "backend.copilot.tools.base.WorkspaceManager",
                return_value=mock_manager,
            ),
        ):
            result = await tool.execute("user-1", session, "tc-big")

        assert "<tool-output-truncated" in str(result.output)
        assert "read_workspace_file" in str(result.output)
        mock_manager.write_file.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_persistence_without_user_id(self):
        """Anonymous users skip persistence (no workspace)."""
        tool = _HugeOutputTool(output_size=_LARGE_OUTPUT_THRESHOLD + 10_000)
        session = MagicMock()
        session.session_id = "s-1"

        # user_id=None → should not attempt persistence
        with patch(
            "backend.copilot.tools.base._persist_and_summarize",
            new_callable=AsyncMock,
        ) as persist_mock:
            result = await tool.execute(None, session, "tc-anon")
        persist_mock.assert_not_awaited()
        # Output is set but not wrapped in <tool-output-truncated> tags
        # (it will be middle-out truncated by model_post_init instead)
        assert "<tool-output-truncated" not in str(result.output)


# ---------------------------------------------------------------------------
# _summarize_binary_fields
# ---------------------------------------------------------------------------


class TestSummarizeBinaryFields:
    def test_replaces_large_content_base64(self):
        import json

        data = {"content_base64": "A" * 10_000, "name": "file.png"}
        result = json.loads(_summarize_binary_fields(json.dumps(data)))
        assert result["name"] == "file.png"
        assert "<binary" in result["content_base64"]
        assert "bytes>" in result["content_base64"]

    def test_preserves_small_content_base64(self):
        import json

        data = {"content_base64": "AQID", "name": "tiny.bin"}
        result_str = _summarize_binary_fields(json.dumps(data))
        result = json.loads(result_str)
        assert result["content_base64"] == "AQID"  # unchanged

    def test_non_json_passthrough(self):
        raw = "not json at all"
        assert _summarize_binary_fields(raw) == raw

    def test_no_binary_fields_unchanged(self):
        import json

        data = {"message": "hello", "type": "info"}
        raw = json.dumps(data)
        assert _summarize_binary_fields(raw) == raw
