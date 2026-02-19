"""Tests for workspace file tool helpers and path validation."""

import base64
import os

import pytest

from backend.copilot.tools.workspace_files import (
    _resolve_write_content,
    _validate_ephemeral_path,
)

# We need to mock make_session_path to return a known temp dir for tests.
# The real one uses WORKSPACE_PREFIX = "/tmp/copilot-"


@pytest.fixture
def ephemeral_dir(tmp_path, monkeypatch):
    """Create a temp dir that acts as the ephemeral session directory."""
    session_dir = tmp_path / "copilot-test-session"
    session_dir.mkdir()

    monkeypatch.setattr(
        "backend.copilot.tools.workspace_files.make_session_path",
        lambda session_id: str(session_dir),
    )
    return session_dir


# ---------------------------------------------------------------------------
# _validate_ephemeral_path
# ---------------------------------------------------------------------------


class TestValidateEphemeralPath:
    def test_valid_path(self, ephemeral_dir):
        target = ephemeral_dir / "file.txt"
        target.touch()
        result = _validate_ephemeral_path(
            str(target), param_name="test", session_id="s1"
        )
        assert isinstance(result, str)
        assert result == os.path.realpath(str(target))

    def test_path_traversal_rejected(self, ephemeral_dir):
        evil_path = str(ephemeral_dir / ".." / "etc" / "passwd")
        result = _validate_ephemeral_path(evil_path, param_name="test", session_id="s1")
        # Should return ErrorResponse
        from backend.copilot.tools.models import ErrorResponse

        assert isinstance(result, ErrorResponse)

    def test_different_session_rejected(self, ephemeral_dir, tmp_path):
        other_dir = tmp_path / "copilot-evil-session"
        other_dir.mkdir()
        target = other_dir / "steal.txt"
        target.touch()
        result = _validate_ephemeral_path(
            str(target), param_name="test", session_id="s1"
        )
        from backend.copilot.tools.models import ErrorResponse

        assert isinstance(result, ErrorResponse)

    def test_symlink_escape_rejected(self, ephemeral_dir, tmp_path):
        """Symlink inside session dir pointing outside should be rejected."""
        outside_file = tmp_path / "secret.txt"
        outside_file.write_text("secret")
        symlink = ephemeral_dir / "link.txt"
        symlink.symlink_to(outside_file)
        result = _validate_ephemeral_path(
            str(symlink), param_name="test", session_id="s1"
        )
        from backend.copilot.tools.models import ErrorResponse

        assert isinstance(result, ErrorResponse)

    def test_nested_path_valid(self, ephemeral_dir):
        nested = ephemeral_dir / "subdir" / "deep"
        nested.mkdir(parents=True)
        target = nested / "data.csv"
        target.touch()
        result = _validate_ephemeral_path(
            str(target), param_name="test", session_id="s1"
        )
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _resolve_write_content
# ---------------------------------------------------------------------------


class TestResolveWriteContent:
    def test_no_sources_returns_error(self):
        from backend.copilot.tools.models import ErrorResponse

        result = _resolve_write_content(None, None, None, "s1")
        assert isinstance(result, ErrorResponse)

    def test_multiple_sources_returns_error(self):
        from backend.copilot.tools.models import ErrorResponse

        result = _resolve_write_content("text", "b64data", None, "s1")
        assert isinstance(result, ErrorResponse)

    def test_plain_text_content(self):
        result = _resolve_write_content("hello world", None, None, "s1")
        assert result == b"hello world"

    def test_base64_content(self):
        raw = b"binary data"
        b64 = base64.b64encode(raw).decode()
        result = _resolve_write_content(None, b64, None, "s1")
        assert result == raw

    def test_invalid_base64_falls_back_to_text(self):
        result = _resolve_write_content(None, "not-valid-b64!!!", None, "s1")
        assert result == b"not-valid-b64!!!"

    def test_source_path(self, ephemeral_dir):
        target = ephemeral_dir / "input.txt"
        target.write_bytes(b"file content")
        result = _resolve_write_content(None, None, str(target), "s1")
        assert result == b"file content"

    def test_source_path_not_found(self, ephemeral_dir):
        from backend.copilot.tools.models import ErrorResponse

        missing = str(ephemeral_dir / "nope.txt")
        result = _resolve_write_content(None, None, missing, "s1")
        assert isinstance(result, ErrorResponse)

    def test_source_path_outside_ephemeral(self, ephemeral_dir, tmp_path):
        from backend.copilot.tools.models import ErrorResponse

        outside = tmp_path / "outside.txt"
        outside.write_text("nope")
        result = _resolve_write_content(None, None, str(outside), "s1")
        assert isinstance(result, ErrorResponse)

    def test_empty_string_sources_treated_as_none(self):
        from backend.copilot.tools.models import ErrorResponse

        # All empty strings → same as no sources
        result = _resolve_write_content("", "", "", "s1")
        assert isinstance(result, ErrorResponse)

    def test_empty_string_source_path_with_text(self):
        # source_path="" should be normalised to None, so only content counts
        result = _resolve_write_content("hello", "", "", "s1")
        assert result == b"hello"
