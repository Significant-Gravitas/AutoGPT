"""Tests for SDK service helpers."""

import base64
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
