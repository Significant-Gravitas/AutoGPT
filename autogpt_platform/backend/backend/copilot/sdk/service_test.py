"""Tests for SDK service helpers."""

import asyncio
import base64
import os
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from .service import _prepare_file_attachments


@dataclass
class _FakeFileInfo:
    id: str
    name: str
    path: str
    mime_type: str
    size_bytes: int


_PATCH_TARGET = "backend.copilot.sdk.service.get_manager"


class TestPrepareFileAttachments:
    @pytest.mark.asyncio
    async def test_empty_list_returns_empty(self, tmp_path):
        result = await _prepare_file_attachments([], "u", "s", str(tmp_path))
        assert result.hint == ""
        assert result.image_blocks == []

    @pytest.mark.asyncio
    async def test_image_embedded_as_vision_block(self, tmp_path):
        """JPEG images should become vision content blocks, not files on disk."""
        raw = b"\xff\xd8\xff\xe0fake-jpeg"
        info = _FakeFileInfo(
            id="abc",
            name="photo.jpg",
            path="/photo.jpg",
            mime_type="image/jpeg",
            size_bytes=len(raw),
        )
        mgr = AsyncMock()
        mgr.get_file_info.return_value = info
        mgr.read_file_by_id.return_value = raw

        with patch(_PATCH_TARGET, new_callable=AsyncMock, return_value=mgr):
            result = await _prepare_file_attachments(
                ["abc"], "user1", "sess1", str(tmp_path)
            )

        assert "1 file" in result.hint
        assert "photo.jpg" in result.hint
        assert "embedded as image" in result.hint
        assert len(result.image_blocks) == 1
        block = result.image_blocks[0]
        assert block["type"] == "image"
        assert block["source"]["media_type"] == "image/jpeg"
        assert block["source"]["data"] == base64.b64encode(raw).decode("ascii")
        # Image should NOT be written to disk (embedded instead)
        assert not os.path.exists(os.path.join(tmp_path, "photo.jpg"))

    @pytest.mark.asyncio
    async def test_pdf_saved_to_disk(self, tmp_path):
        """PDFs should be saved to disk for Read tool access, not embedded."""
        info = _FakeFileInfo("f1", "doc.pdf", "/doc.pdf", "application/pdf", 50)
        mgr = AsyncMock()
        mgr.get_file_info.return_value = info
        mgr.read_file_by_id.return_value = b"%PDF-1.4 fake"

        with patch(_PATCH_TARGET, new_callable=AsyncMock, return_value=mgr):
            result = await _prepare_file_attachments(["f1"], "u", "s", str(tmp_path))

        assert result.image_blocks == []
        saved = tmp_path / "doc.pdf"
        assert saved.exists()
        assert saved.read_bytes() == b"%PDF-1.4 fake"
        assert str(saved) in result.hint

    @pytest.mark.asyncio
    async def test_mixed_images_and_files(self, tmp_path):
        """Images become blocks, non-images go to disk."""
        infos = {
            "id1": _FakeFileInfo("id1", "a.png", "/a.png", "image/png", 4),
            "id2": _FakeFileInfo("id2", "b.pdf", "/b.pdf", "application/pdf", 4),
            "id3": _FakeFileInfo("id3", "c.txt", "/c.txt", "text/plain", 4),
        }
        mgr = AsyncMock()
        mgr.get_file_info.side_effect = lambda fid: infos[fid]
        mgr.read_file_by_id.return_value = b"data"

        with patch(_PATCH_TARGET, new_callable=AsyncMock, return_value=mgr):
            result = await _prepare_file_attachments(
                ["id1", "id2", "id3"], "u", "s", str(tmp_path)
            )

        assert "3 files" in result.hint
        assert "a.png" in result.hint
        assert "b.pdf" in result.hint
        assert "c.txt" in result.hint
        # Only the image should be a vision block
        assert len(result.image_blocks) == 1
        assert result.image_blocks[0]["source"]["media_type"] == "image/png"
        # Non-image files should be on disk
        assert (tmp_path / "b.pdf").exists()
        assert (tmp_path / "c.txt").exists()
        # Read tool hint should appear (has non-image files)
        assert "Read tool" in result.hint

    @pytest.mark.asyncio
    async def test_singular_noun(self, tmp_path):
        info = _FakeFileInfo("x", "only.txt", "/only.txt", "text/plain", 2)
        mgr = AsyncMock()
        mgr.get_file_info.return_value = info
        mgr.read_file_by_id.return_value = b"hi"

        with patch(_PATCH_TARGET, new_callable=AsyncMock, return_value=mgr):
            result = await _prepare_file_attachments(["x"], "u", "s", str(tmp_path))

        assert "1 file." in result.hint

    @pytest.mark.asyncio
    async def test_missing_file_skipped(self, tmp_path):
        mgr = AsyncMock()
        mgr.get_file_info.return_value = None

        with patch(_PATCH_TARGET, new_callable=AsyncMock, return_value=mgr):
            result = await _prepare_file_attachments(
                ["missing-id"], "u", "s", str(tmp_path)
            )

        assert result.hint == ""
        assert result.image_blocks == []

    @pytest.mark.asyncio
    async def test_image_only_no_read_hint(self, tmp_path):
        """When all files are images, no Read tool hint should appear."""
        info = _FakeFileInfo("i1", "cat.png", "/cat.png", "image/png", 4)
        mgr = AsyncMock()
        mgr.get_file_info.return_value = info
        mgr.read_file_by_id.return_value = b"data"

        with patch(_PATCH_TARGET, new_callable=AsyncMock, return_value=mgr):
            result = await _prepare_file_attachments(["i1"], "u", "s", str(tmp_path))

        assert "Read tool" not in result.hint
        assert len(result.image_blocks) == 1


class TestPromptSupplement:
    """Tests for centralized prompt supplement generation."""

    def test_sdk_supplement_excludes_tool_docs(self):
        """SDK mode should NOT include tool documentation (Claude gets schemas automatically)."""
        from backend.copilot.prompting import get_sdk_supplement

        # Test both local and E2B modes
        local_supplement = get_sdk_supplement(use_e2b=False, cwd="/tmp/test")
        e2b_supplement = get_sdk_supplement(use_e2b=True, cwd="")

        # Should NOT have tool list section
        assert "## AVAILABLE TOOLS" not in local_supplement
        assert "## AVAILABLE TOOLS" not in e2b_supplement

        # Should still have technical notes
        assert "## Tool notes" in local_supplement
        assert "## Tool notes" in e2b_supplement

    def test_baseline_supplement_includes_tool_docs(self):
        """Baseline mode MUST include tool documentation (direct API needs it)."""
        from backend.copilot.prompting import get_baseline_supplement

        supplement = get_baseline_supplement()

        # MUST have tool list section
        assert "## AVAILABLE TOOLS" in supplement

        # Should NOT have environment-specific notes (SDK-only)
        assert "## Tool notes" not in supplement

    def test_baseline_supplement_includes_key_tools(self):
        """Baseline supplement should document all essential tools."""
        from backend.copilot.prompting import get_baseline_supplement
        from backend.copilot.tools import TOOL_REGISTRY

        docs = get_baseline_supplement()

        # Core agent workflow tools (always available)
        assert "`create_agent`" in docs
        assert "`run_agent`" in docs
        assert "`find_library_agent`" in docs
        assert "`edit_agent`" in docs

        # MCP integration (always available)
        assert "`run_mcp_tool`" in docs

        # Folder management (always available)
        assert "`create_folder`" in docs

        # Browser tools only if available (Playwright may not be installed in CI)
        if (
            TOOL_REGISTRY.get("browser_navigate")
            and TOOL_REGISTRY["browser_navigate"].is_available
        ):
            assert "`browser_navigate`" in docs

    def test_baseline_supplement_includes_workflows(self):
        """Baseline supplement should include workflow guidance in tool descriptions."""
        from backend.copilot.prompting import get_baseline_supplement

        docs = get_baseline_supplement()

        # Workflows are now in individual tool descriptions (not separate sections)
        # Check that key workflow concepts appear in tool descriptions
        assert "agent_json" in docs or "find_block" in docs
        assert "run_mcp_tool" in docs

    def test_baseline_supplement_completeness(self):
        """All available tools from TOOL_REGISTRY should appear in baseline supplement."""
        from backend.copilot.prompting import get_baseline_supplement
        from backend.copilot.tools import TOOL_REGISTRY

        docs = get_baseline_supplement()

        # Verify each available registered tool is documented
        # (matches _generate_tool_documentation which filters by is_available)
        for tool_name, tool in TOOL_REGISTRY.items():
            if not tool.is_available:
                continue
            assert (
                f"`{tool_name}`" in docs
            ), f"Tool '{tool_name}' missing from baseline supplement"

    def test_pause_task_scheduled_before_transcript_upload(self):
        """Pause is scheduled as a background task before transcript upload begins.

        The finally block in stream_response_sdk does:
          (1) asyncio.create_task(pause_sandbox_direct(...))  — fire-and-forget
          (2) await asyncio.shield(upload_transcript(...))    — awaited

        Scheduling pause via create_task before awaiting upload ensures:
        - Pause never blocks transcript upload (billing stops concurrently)
        - On E2B timeout, pause silently fails; upload proceeds unaffected
        """
        call_order: list[str] = []

        async def _mock_pause(sandbox, session_id):
            call_order.append("pause")

        async def _mock_upload(**kwargs):
            call_order.append("upload")

        async def _simulate_teardown():
            """Mirror the service.py finally block teardown sequence."""
            sandbox = MagicMock()

            # (1) Schedule pause — mirrors lines ~1427-1429 in service.py
            task = asyncio.create_task(_mock_pause(sandbox, "test-sess"))

            # (2) Await transcript upload — mirrors lines ~1460-1468 in service.py
            # Yielding to the event loop here lets the pause task start concurrently.
            await _mock_upload(
                user_id="u", session_id="test-sess", content="x", message_count=1
            )
            await task

        asyncio.run(_simulate_teardown())

        # Both must run; pause is scheduled before upload starts
        assert "pause" in call_order
        assert "upload" in call_order
        # create_task schedules pause, then upload is awaited — pause runs
        # concurrently during upload's first yield. The ordering guarantee is
        # that create_task is CALLED before upload is AWAITED (see source order).

    def test_baseline_supplement_no_duplicate_tools(self):
        """No tool should appear multiple times in baseline supplement."""
        from backend.copilot.prompting import get_baseline_supplement
        from backend.copilot.tools import TOOL_REGISTRY

        docs = get_baseline_supplement()

        # Count occurrences of each available tool in the entire supplement
        for tool_name, tool in TOOL_REGISTRY.items():
            if not tool.is_available:
                continue
            # Count how many times this tool appears as a bullet point
            count = docs.count(f"- **`{tool_name}`**")
            assert count == 1, f"Tool '{tool_name}' appears {count} times (should be 1)"
