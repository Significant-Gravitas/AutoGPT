"""Tests for BaseTool large-output persistence in execute()."""

import json
import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.tools.base import (
    _DIGEST_PREVIEW_CHARS,
    _DIGEST_THRESHOLD,
    _LARGE_OUTPUT_THRESHOLD,
    BaseTool,
    _index_json,
    _outline,
    _persist_and_summarize,
    _summarize_binary_fields,
)
from backend.copilot.tools.models import (
    BlockDetails,
    BlockDetailsResponse,
    ResponseType,
    ToolResponseBase,
)


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


# ---------------------------------------------------------------------------
# AUTOPILOT_DELEGATION digest
# ---------------------------------------------------------------------------


class _SchemaOutputTool(_HugeOutputTool):
    """Returns a block-details payload, which is what the digest exists for."""

    async def _execute(self, user_id, session, **kwargs) -> ToolResponseBase:
        return BlockDetailsResponse(
            message="Block 'Wide Block' details.",
            block=BlockDetails(
                id="b-1",
                name="Wide Block",
                description="A block with many fields",
                inputs={
                    "properties": {
                        f"field_{i}": {
                            "type": "string",
                            "description": "d" * (self._output_size // 40),
                        }
                        for i in range(40)
                    },
                    "required": ["field_0"],
                },
            ),
        )


def _workspace_patches(manager):
    workspace = MagicMock()
    workspace.id = "ws-1"
    db = AsyncMock()
    db.get_or_create_workspace = AsyncMock(return_value=workspace)
    return (
        patch("backend.copilot.tools.base.workspace_db", return_value=db),
        patch("backend.copilot.tools.base.WorkspaceManager", return_value=manager),
    )


async def _execute_with_flag(tool, flag_on: bool, manager=None):
    session = MagicMock()
    session.session_id = "s-1"
    db_patch, mgr_patch = _workspace_patches(manager or AsyncMock())
    with (
        db_patch,
        mgr_patch,
        patch(
            "backend.copilot.tools.base.is_feature_enabled",
            new_callable=AsyncMock,
            return_value=flag_on,
        ),
    ):
        return await tool.execute("user-1", session, "tc-digest")


class TestDigestThreshold:
    def test_the_budget_cannot_exceed_its_own_trigger(self):
        """Between 80K and 95K the legacy preview makes the context bigger;
        deriving the budget from the trigger makes that unrepresentable."""
        assert _DIGEST_PREVIEW_CHARS < _DIGEST_THRESHOLD

    @pytest.mark.asyncio
    async def test_flag_off_leaves_a_mid_sized_output_byte_identical(self):
        tool = _SchemaOutputTool(output_size=_DIGEST_THRESHOLD * 3)
        result = await _execute_with_flag(tool, flag_on=False)
        expected = (await tool._execute("user-1", MagicMock())).model_dump_json(
            exclude_none=True
        )
        assert result.output == expected

    @pytest.mark.asyncio
    async def test_flag_on_digests_a_mid_sized_output(self):
        manager = AsyncMock()
        tool = _SchemaOutputTool(output_size=_DIGEST_THRESHOLD * 3)
        result = await _execute_with_flag(tool, flag_on=True, manager=manager)
        output = str(result.output)

        assert 'format="outline"' in output
        assert len(output) < _DIGEST_THRESHOLD
        # Decision-grade: every property name survives, so the model can tell
        # from the digest alone whether it needs the file.
        assert all(f"field_{i}" in output for i in range(40))
        # And the way back names the tool and its arguments.
        assert 'read_workspace_file(path="tool-outputs/tc-digest.json"' in output
        assert "offset=<offset>, length=<length>" in output
        # Without this the model reads the outline as a prefix of the output.
        assert "not a literal prefix" in output
        # A flat length= wider than the file turns one "narrow" read into a
        # full one, which is how a digest ends up costing more than it saves.
        assert "length=50000" not in output
        assert "@" in output

    @pytest.mark.asyncio
    async def test_flag_on_writes_the_full_text_to_the_workspace(self):
        manager = AsyncMock()
        tool = _SchemaOutputTool(output_size=_DIGEST_THRESHOLD * 3)
        result = await _execute_with_flag(tool, flag_on=True, manager=manager)
        written = manager.write_file.await_args.kwargs["content"].decode()
        assert len(written) > _DIGEST_THRESHOLD
        assert written == (await tool._execute("user-1", MagicMock())).model_dump_json(
            exclude_none=True
        )
        assert len(str(result.output)) < len(written)

    @pytest.mark.asyncio
    async def test_a_small_output_never_consults_the_flag(self):
        tool = _SchemaOutputTool(output_size=200)
        with patch(
            "backend.copilot.tools.base.is_feature_enabled",
            new_callable=AsyncMock,
        ) as flag:
            session = MagicMock()
            session.session_id = "s-1"
            await tool.execute("user-1", session, "tc-small")
        flag.assert_not_awaited()


def _outline_of(data, budget: int) -> tuple[str, str]:
    body, parsed, offsets = _index_json(json.dumps(data))
    return body, _outline(parsed, offsets, budget)


class TestOutline:
    def test_non_json_is_not_indexed(self):
        assert _index_json("not json at all") is None
        assert _index_json(json.dumps("a bare string")) is None

    def test_names_every_key_before_spending_the_budget_on_values(self):
        data = {"a": {"deep": {"x": "v" * 4_000}}, "b": 1, "c": [1, 2, 3]}
        _, outline = _outline_of(data, 400)
        assert "b=1" in outline
        assert "a={…1 keys @" in outline
        assert "c=[…3 items @" in outline
        assert len(outline) <= 400

    def test_every_indexed_window_slices_out_exactly_that_node(self):
        """The whole point of the index: one narrow read, not the whole file."""
        data = {"block": {"inputs": {"q": {"type": "string"}}}, "n": [1, {"a": 2}]}
        body, parsed, offsets = _index_json(json.dumps(data))
        assert parsed == data
        assert json.loads(body) == data
        assert offsets["$"] == (0, len(body))
        for path, (start, length) in offsets.items():
            node = data
            for step in path[1:].replace("[", ".").replace("]", "").split(".")[1:]:
                node = node[int(step)] if isinstance(node, list) else node[step]
            assert json.loads(body[start : start + length]) == node, path

    def test_the_outline_quotes_each_node_its_real_window(self):
        """An index nothing quotes is no index: the window in the outline has
        to slice that node out of the file the model will read."""
        data = {"block": {"inputs": {"q": {"type": "string", "title": "Q"}}}}
        body, outline = _outline_of(data, 600)
        window = re.search(r"inputs=\{…1 keys @(\d+)\+(\d+)\}", outline)
        assert window, outline
        start, length = int(window[1]), int(window[2])
        assert json.loads(body[start : start + length]) == data["block"]["inputs"]

    def test_a_single_long_string_still_says_something(self):
        """Structure alone leaves the budget unspent, so the scalar cap widens."""
        _, outline = _outline_of({"content": "abcdef" * 1_000}, 800)
        assert len(outline) > 400
        assert len(outline) <= 800

    def test_says_how_much_it_left_out(self):
        data = {f"k{i}": {"v": "x" * 200} for i in range(30)}
        _, outline = _outline_of(data, 600)
        assert "more nodes not shown" in outline
        assert len(outline) <= 600
