"""Tests for E2B file-tool path validation and local read safety.

Pure unit tests with no external dependencies (no E2B, no sandbox).
"""

import os

import pytest

from .e2b_file_tools import (
    _MAX_RESPONSE_CHARS,
    _maybe_truncate,
    _read_local,
    _resolve_remote,
)
from .tool_adapter import _current_project_dir

_SDK_PROJECTS_DIR = os.path.realpath(os.path.expanduser("~/.claude/projects"))


# ---------------------------------------------------------------------------
# _resolve_remote — sandbox path normalisation & boundary enforcement
# ---------------------------------------------------------------------------


class TestResolveRemote:
    def test_relative_path_resolved(self):
        assert _resolve_remote("src/main.py") == "/home/user/src/main.py"

    def test_absolute_within_sandbox(self):
        assert _resolve_remote("/home/user/file.txt") == "/home/user/file.txt"

    def test_workdir_itself(self):
        assert _resolve_remote("/home/user") == "/home/user"

    def test_relative_dotslash(self):
        assert _resolve_remote("./README.md") == "/home/user/README.md"

    def test_traversal_blocked(self):
        with pytest.raises(ValueError, match="must be within /home/user"):
            _resolve_remote("../../etc/passwd")

    def test_absolute_traversal_blocked(self):
        with pytest.raises(ValueError, match="must be within /home/user"):
            _resolve_remote("/home/user/../../etc/passwd")

    def test_absolute_outside_sandbox_blocked(self):
        with pytest.raises(ValueError, match="must be within /home/user"):
            _resolve_remote("/etc/passwd")

    def test_root_blocked(self):
        with pytest.raises(ValueError, match="must be within /home/user"):
            _resolve_remote("/")

    def test_home_other_user_blocked(self):
        with pytest.raises(ValueError, match="must be within /home/user"):
            _resolve_remote("/home/other/file.txt")

    def test_deep_nested_allowed(self):
        assert _resolve_remote("a/b/c/d/e.txt") == "/home/user/a/b/c/d/e.txt"

    def test_trailing_slash_normalised(self):
        assert _resolve_remote("src/") == "/home/user/src"

    def test_double_dots_within_sandbox_ok(self):
        """Path that resolves back within /home/user is allowed."""
        assert _resolve_remote("a/b/../c.txt") == "/home/user/a/c.txt"


# ---------------------------------------------------------------------------
# _read_local — host filesystem reads with allowlist enforcement
#
# In E2B mode, _read_local only allows tool-results paths (via
# is_allowed_local_path without sdk_cwd).  Regular files live on the
# sandbox, not the host.
# ---------------------------------------------------------------------------


class TestReadLocal:
    def _make_tool_results_file(self, encoded: str, filename: str, content: str) -> str:
        """Create a tool-results file and return its path."""
        tool_results_dir = os.path.join(_SDK_PROJECTS_DIR, encoded, "tool-results")
        os.makedirs(tool_results_dir, exist_ok=True)
        filepath = os.path.join(tool_results_dir, filename)
        with open(filepath, "w") as f:
            f.write(content)
        return filepath

    def test_read_tool_results_file(self):
        """Reading a tool-results file should succeed."""
        encoded = "-tmp-copilot-e2b-test-read"
        filepath = self._make_tool_results_file(
            encoded, "result.txt", "line 1\nline 2\nline 3\n"
        )
        token = _current_project_dir.set(encoded)
        try:
            result = _read_local(filepath, offset=0, limit=2000)
            assert result["isError"] is False
            assert "line 1" in result["content"][0]["text"]
            assert "line 2" in result["content"][0]["text"]
        finally:
            _current_project_dir.reset(token)
            os.unlink(filepath)

    def test_read_disallowed_path_blocked(self):
        """Reading /etc/passwd should be blocked by the allowlist."""
        result = _read_local("/etc/passwd", offset=0, limit=10)
        assert result["isError"] is True
        assert "not allowed" in result["content"][0]["text"].lower()

    def test_read_nonexistent_tool_results(self):
        """A tool-results path that doesn't exist returns FileNotFoundError."""
        encoded = "-tmp-copilot-e2b-test-nofile"
        tool_results_dir = os.path.join(_SDK_PROJECTS_DIR, encoded, "tool-results")
        os.makedirs(tool_results_dir, exist_ok=True)
        filepath = os.path.join(tool_results_dir, "nonexistent.txt")
        token = _current_project_dir.set(encoded)
        try:
            result = _read_local(filepath, offset=0, limit=10)
            assert result["isError"] is True
            assert "not found" in result["content"][0]["text"].lower()
        finally:
            _current_project_dir.reset(token)
            os.rmdir(tool_results_dir)

    def test_read_traversal_path_blocked(self):
        """A traversal attempt that escapes allowed directories is blocked."""
        result = _read_local("/tmp/copilot-abc/../../etc/shadow", offset=0, limit=10)
        assert result["isError"] is True
        assert "not allowed" in result["content"][0]["text"].lower()

    def test_read_arbitrary_host_path_blocked(self):
        """Arbitrary host paths are blocked even if they exist."""
        result = _read_local("/proc/self/environ", offset=0, limit=10)
        assert result["isError"] is True

    def test_read_with_offset_and_limit(self):
        """Offset and limit should control which lines are returned."""
        encoded = "-tmp-copilot-e2b-test-offset"
        content = "".join(f"line {i}\n" for i in range(10))
        filepath = self._make_tool_results_file(encoded, "lines.txt", content)
        token = _current_project_dir.set(encoded)
        try:
            result = _read_local(filepath, offset=3, limit=2)
            assert result["isError"] is False
            text = result["content"][0]["text"]
            assert "line 3" in text
            assert "line 4" in text
            assert "line 2" not in text
            assert "line 5" not in text
        finally:
            _current_project_dir.reset(token)
            os.unlink(filepath)

    def test_read_without_project_dir_blocks_all(self):
        """Without _current_project_dir set, all paths are blocked."""
        result = _read_local("/tmp/anything.txt", offset=0, limit=10)
        assert result["isError"] is True


# ---------------------------------------------------------------------------
# _maybe_truncate — middle-out truncation for large responses
# ---------------------------------------------------------------------------


class TestMaybeTruncate:
    def test_short_text_unchanged(self):
        text = "hello world"
        assert _maybe_truncate(text) == text

    def test_exact_limit_unchanged(self):
        text = "x" * _MAX_RESPONSE_CHARS
        assert _maybe_truncate(text) == text

    def test_over_limit_truncated(self):
        text = "x" * (_MAX_RESPONSE_CHARS + 1000)
        result = _maybe_truncate(text)
        assert len(result) < len(text)
        assert "omitted" in result

    def test_head_and_tail_preserved(self):
        head = "HEAD_MARKER_" + "a" * 50_000
        tail = "b" * 50_000 + "_TAIL_MARKER"
        middle = "m" * (_MAX_RESPONSE_CHARS + 50_000)
        text = head + middle + tail
        result = _maybe_truncate(text)
        assert result.startswith("HEAD_MARKER_")
        assert result.endswith("_TAIL_MARKER")
        assert "omitted" in result

    def test_empty_string(self):
        assert _maybe_truncate("") == ""
