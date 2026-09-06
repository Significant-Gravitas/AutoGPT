"""Tests for BaseTool large-output persistence in execute()."""

import base64
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
from backend.copilot.tools.workspace_files import (
    ReadWorkspaceFileTool,
    WorkspaceFileContentResponse,
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

    @pytest.mark.asyncio
    async def test_the_excerpt_envelope_is_unchanged_from_before_the_digest(self):
        """Flag off has to be byte-identical above 80K too, which is what the
        "safe to merge dark" claim rests on."""
        db_patch, mgr_patch = _workspace_patches(AsyncMock())
        with db_patch, mgr_patch:
            result = await _persist_and_summarize(
                "A" * 200_000, "user-1", "session-1", "tc-123"
            )

        assert 'workspace_path="tool-outputs/tc-123.json">' in result
        assert "format=" not in result
        assert "to read any section. To process the file" in result


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

    digest_large_output = True

    def __init__(self, output_size: int, message: str = "Block details.", **field):
        super().__init__(output_size)
        self._message = message
        self._field = field

    async def _execute(self, user_id, session, **kwargs) -> ToolResponseBase:
        return BlockDetailsResponse(
            message=self._message,
            block=BlockDetails(
                id="b-1",
                name="Wide Block",
                description="A block with many fields",
                inputs={
                    "properties": {
                        f"field_{i}": {
                            "type": "string",
                            "description": "d" * (self._output_size // 40),
                            **self._field,
                        }
                        for i in range(40)
                    },
                    "required": ["field_0"],
                },
            ),
        )


class _BinaryOutputTool(_HugeOutputTool):
    """Returns base64 payload: 1K of it tells the model nothing its size doesn't."""

    digest_large_output = True

    async def _execute(self, user_id, session, **kwargs) -> ToolResponseBase:
        return WorkspaceFileContentResponse(
            file_id="f-1",
            name="shot.png",
            path="tool-outputs/shot.png",
            mime_type="image/png",
            content_base64="A" * self._output_size,
            message="Screenshot captured.",
        )


class _RetrievalTool(ReadWorkspaceFileTool):
    """The real retrieval tool with only the workspace read stubbed out."""

    def __init__(self, text: str) -> None:
        self._text = text

    async def _execute(self, user_id, session, **kwargs) -> ToolResponseBase:
        return WorkspaceFileContentResponse(
            file_id="f-1",
            name="tc-digest.json",
            path="tool-outputs/tc-digest.json",
            mime_type="text/plain",
            content_base64=base64.b64encode(self._text.encode()).decode(),
            message=f"Read chars 0-{len(self._text)} of {len(self._text):,} total",
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
    async def test_a_tool_that_does_not_opt_in_is_never_digested(self):
        """An outline is only readable back in windows, so a tool whose bulk is
        one long text (the building guide, a docs page) must not be digested."""
        tool = _HugeOutputTool(output_size=_DIGEST_THRESHOLD * 4)
        assert tool.digest_large_output is False
        result = await _execute_with_flag(tool, flag_on=True)
        expected = (await tool._execute("user-1", MagicMock())).model_dump_json(
            exclude_none=True
        )
        assert result.output == expected

    @pytest.mark.asyncio
    async def test_a_retrieved_window_is_never_digested(self):
        """The way back must fit through the door: base64 pushes any window
        over ~5.8K past the trigger, and digesting it returns base64, not text."""
        text = "R" * (_DIGEST_THRESHOLD * 2)
        result = await _execute_with_flag(_RetrievalTool(text), flag_on=True)
        payload = json.loads(str(result.output))
        assert base64.b64decode(payload["content_base64"]).decode() == text

    @pytest.mark.asyncio
    async def test_the_outline_summarizes_binary_fields_instead_of_quoting_them(self):
        """1K of base64 is worth nothing to the model; its size is."""
        tool = _BinaryOutputTool(output_size=_DIGEST_THRESHOLD * 2)
        result = await _execute_with_flag(tool, flag_on=True)
        output = str(result.output)
        assert "<binary, ~12,000 bytes>" in output
        assert "AAAAAAAA" not in output

    @pytest.mark.asyncio
    async def test_the_outline_keeps_the_message_whole(self):
        """`message` carries the tool's own instruction to the model — on
        run_block, that credentials must not go in input_data."""
        message = "Block 'Wide Block' details. " + "Do not fabricate credentials. " * 8
        tool = _SchemaOutputTool(output_size=_DIGEST_THRESHOLD * 3, message=message)
        result = await _execute_with_flag(tool, flag_on=True)
        assert message in str(result.output)

    @pytest.mark.asyncio
    async def test_a_long_message_cannot_crowd_out_the_field_names(self):
        """`view_agent_output` interpolates raw node errors into its message,
        so an unbounded one would spend the whole outline on a stack trace."""
        tool = _SchemaOutputTool(
            output_size=_DIGEST_THRESHOLD * 3, message="E" * (_DIGEST_THRESHOLD // 2)
        )
        output = str((await _execute_with_flag(tool, flag_on=True)).output)
        # Uncapped this is 0: the message alone consumes the whole budget.
        assert sum(f"field_{i}" in output for i in range(40)) >= 10

    @pytest.mark.asyncio
    async def test_the_persisted_text_is_the_text_the_offsets_index(self):
        """pydantic and json.dumps disagree on float exponents, so persisting
        the pydantic serialisation would shift every later window by a char."""
        manager = AsyncMock()
        tool = _SchemaOutputTool(output_size=_DIGEST_THRESHOLD * 3, default=1e-7)
        result = await _execute_with_flag(tool, flag_on=True, manager=manager)
        written = manager.write_file.await_args.kwargs["content"].decode()
        window = re.search(r"required=\[…1 items @(\d+)\+(\d+)\]", str(result.output))
        assert window, result.output
        start, length = int(window[1]), int(window[2])
        assert json.loads(written[start : start + length]) == ["field_0"]

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
        assert "a={…1 keys" in outline
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

    def test_no_window_costs_more_to_read_back_than_the_whole_output(self):
        """read_workspace_file base64-encodes its slice, so a window spanning
        most of the file is a read the model must never be offered."""
        data = {"tag": "x", "record": {f"k{i}": {"v": "v" * 200} for i in range(10)}}
        body, outline = _outline_of(data, 900)
        widest = max(int(m) for m in re.findall(r"@\d+\+(\d+)", outline))
        assert widest * 4 // 3 < len(body)

    def test_an_oversized_node_gets_its_first_chunk(self):
        """A node too big to quote whole used to carry no window at all, which
        left the model nothing to read it back with."""
        data = {"tag": "x", "record": {f"k{i}": {"v": "v" * 200} for i in range(10)}}
        body, parsed, offsets = _index_json(json.dumps(data))
        outline = _outline(parsed, offsets, 900)
        node_start, node_length = offsets["$.record"]
        window = re.search(r"record=\{…10 keys @(\d+)\+(\d+)\}", outline)
        assert window, outline
        start, length = int(window[1]), int(window[2])
        assert start == node_start
        assert length < node_length, "an oversized node's window must be capped"
        assert body[start : start + length] == body[node_start:][:length]

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
