"""Tests for SDK service helpers."""

from .service import _build_file_attachment_hint


class TestBuildFileAttachmentHint:
    def test_empty_list_returns_empty(self):
        assert _build_file_attachment_hint([]) == ""

    def test_single_file(self):
        hint = _build_file_attachment_hint(["file-abc-123"])
        assert "1 file" in hint
        assert "`file-abc-123`" in hint
        assert "read_workspace_file" in hint

    def test_multiple_files(self):
        hint = _build_file_attachment_hint(["id1", "id2", "id3"])
        assert "3 files" in hint
        assert "`id1`" in hint
        assert "`id2`" in hint
        assert "`id3`" in hint

    def test_singular_noun(self):
        hint = _build_file_attachment_hint(["only-one"])
        assert "1 file " in hint  # "file" not "files"
        assert "files" not in hint
