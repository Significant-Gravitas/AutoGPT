"""Tests for SDK service helpers."""

import os
from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

import pytest

from .service import _prepare_file_attachments


@dataclass
class _FakeFileInfo:
    id: str
    name: str
    path: str
    mime_type: str
    size_bytes: int


class TestPrepareFileAttachments:
    @pytest.mark.asyncio
    async def test_empty_list_returns_empty(self, tmp_path):
        assert await _prepare_file_attachments([], "u", "s", str(tmp_path)) == ""

    @pytest.mark.asyncio
    async def test_single_file(self, tmp_path):
        info = _FakeFileInfo(
            id="abc",
            name="photo.jpg",
            path="/photo.jpg",
            mime_type="image/jpeg",
            size_bytes=1024,
        )
        mgr = AsyncMock()
        mgr.get_file_info.return_value = info
        mgr.read_file_by_id.return_value = b"\xff\xd8\xff\xe0fake-jpeg"

        with patch(
            "backend.copilot.tools.workspace_files.get_manager",
            new_callable=AsyncMock,
            return_value=mgr,
        ):
            hint = await _prepare_file_attachments(
                ["abc"], "user1", "sess1", str(tmp_path)
            )

        assert "1 file" in hint
        assert "photo.jpg" in hint
        assert "Read tool" in hint
        # File should have been written to tmp_path
        assert os.path.exists(os.path.join(tmp_path, "photo.jpg"))

    @pytest.mark.asyncio
    async def test_multiple_files(self, tmp_path):
        infos = {
            "id1": _FakeFileInfo("id1", "a.png", "/a.png", "image/png", 500),
            "id2": _FakeFileInfo("id2", "b.pdf", "/b.pdf", "application/pdf", 2000),
            "id3": _FakeFileInfo("id3", "c.txt", "/c.txt", "text/plain", 100),
        }
        mgr = AsyncMock()
        mgr.get_file_info.side_effect = lambda fid: infos[fid]
        mgr.read_file_by_id.return_value = b"data"

        with patch(
            "backend.copilot.tools.workspace_files.get_manager",
            new_callable=AsyncMock,
            return_value=mgr,
        ):
            hint = await _prepare_file_attachments(
                ["id1", "id2", "id3"], "u", "s", str(tmp_path)
            )

        assert "3 files" in hint
        assert "a.png" in hint
        assert "b.pdf" in hint
        assert "c.txt" in hint

    @pytest.mark.asyncio
    async def test_singular_noun(self, tmp_path):
        info = _FakeFileInfo("x", "only.txt", "/only.txt", "text/plain", 10)
        mgr = AsyncMock()
        mgr.get_file_info.return_value = info
        mgr.read_file_by_id.return_value = b"hi"

        with patch(
            "backend.copilot.tools.workspace_files.get_manager",
            new_callable=AsyncMock,
            return_value=mgr,
        ):
            hint = await _prepare_file_attachments(["x"], "u", "s", str(tmp_path))

        assert "1 file " in hint
        assert "files" not in hint

    @pytest.mark.asyncio
    async def test_missing_file_skipped(self, tmp_path):
        mgr = AsyncMock()
        mgr.get_file_info.return_value = None
        mgr.read_file_by_id.side_effect = Exception("should not be called")

        with patch(
            "backend.copilot.tools.workspace_files.get_manager",
            new_callable=AsyncMock,
            return_value=mgr,
        ):
            hint = await _prepare_file_attachments(
                ["missing-id"], "u", "s", str(tmp_path)
            )

        assert hint == ""

    @pytest.mark.asyncio
    async def test_file_written_to_disk(self, tmp_path):
        info = _FakeFileInfo("f1", "doc.pdf", "/doc.pdf", "application/pdf", 50)
        mgr = AsyncMock()
        mgr.get_file_info.return_value = info
        mgr.read_file_by_id.return_value = b"%PDF-1.4 fake"

        with patch(
            "backend.copilot.tools.workspace_files.get_manager",
            new_callable=AsyncMock,
            return_value=mgr,
        ):
            hint = await _prepare_file_attachments(["f1"], "u", "s", str(tmp_path))

        saved = tmp_path / "doc.pdf"
        assert saved.exists()
        assert saved.read_bytes() == b"%PDF-1.4 fake"
        assert str(saved) in hint
