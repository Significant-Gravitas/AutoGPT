"""Tests for E2B file-tool path validation and local read safety.

Pure unit tests with no external dependencies (no E2B, no sandbox).
"""

import os
import shutil
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.copilot.context import E2B_WORKDIR, SDK_PROJECTS_DIR, _current_project_dir

from .e2b_file_tools import (
    _check_sandbox_symlink_escape,
    _read_local,
    _sandbox_write,
    resolve_sandbox_path,
)

# ---------------------------------------------------------------------------
# resolve_sandbox_path — sandbox path normalisation & boundary enforcement
# ---------------------------------------------------------------------------


class TestResolveSandboxPath:
    def test_relative_path_resolved(self):
        assert resolve_sandbox_path("src/main.py") == f"{E2B_WORKDIR}/src/main.py"

    def test_absolute_within_sandbox(self):
        assert (
            resolve_sandbox_path(f"{E2B_WORKDIR}/file.txt") == f"{E2B_WORKDIR}/file.txt"
        )

    def test_workdir_itself(self):
        assert resolve_sandbox_path(E2B_WORKDIR) == E2B_WORKDIR

    def test_relative_dotslash(self):
        assert resolve_sandbox_path("./README.md") == f"{E2B_WORKDIR}/README.md"

    def test_traversal_blocked(self):
        with pytest.raises(ValueError, match="must be within"):
            resolve_sandbox_path("../../etc/passwd")

    def test_absolute_traversal_blocked(self):
        with pytest.raises(ValueError, match="must be within"):
            resolve_sandbox_path(f"{E2B_WORKDIR}/../../etc/passwd")

    def test_absolute_outside_sandbox_blocked(self):
        with pytest.raises(ValueError, match="must be within"):
            resolve_sandbox_path("/etc/passwd")

    def test_root_blocked(self):
        with pytest.raises(ValueError, match="must be within"):
            resolve_sandbox_path("/")

    def test_home_other_user_blocked(self):
        with pytest.raises(ValueError, match="must be within"):
            resolve_sandbox_path("/home/other/file.txt")

    def test_deep_nested_allowed(self):
        assert resolve_sandbox_path("a/b/c/d/e.txt") == f"{E2B_WORKDIR}/a/b/c/d/e.txt"

    def test_trailing_slash_normalised(self):
        assert resolve_sandbox_path("src/") == f"{E2B_WORKDIR}/src"

    def test_double_dots_within_sandbox_ok(self):
        """Path that resolves back within E2B_WORKDIR is allowed."""
        assert resolve_sandbox_path("a/b/../c.txt") == f"{E2B_WORKDIR}/a/c.txt"

    def test_tmp_absolute_allowed(self):
        assert resolve_sandbox_path("/tmp/data.txt") == "/tmp/data.txt"

    def test_tmp_nested_allowed(self):
        assert resolve_sandbox_path("/tmp/a/b/c.txt") == "/tmp/a/b/c.txt"

    def test_tmp_itself_allowed(self):
        assert resolve_sandbox_path("/tmp") == "/tmp"

    def test_tmp_escape_blocked(self):
        with pytest.raises(ValueError, match="must be within"):
            resolve_sandbox_path("/tmp/../etc/passwd")

    def test_tmp_prefix_collision_blocked(self):
        """A path like /tmp_evil should be blocked (not a prefix match)."""
        with pytest.raises(ValueError, match="must be within"):
            resolve_sandbox_path("/tmp_evil/malicious.txt")


# ---------------------------------------------------------------------------
# _read_local — host filesystem reads with allowlist enforcement
#
# In E2B mode, _read_local only allows tool-results paths (via
# is_allowed_local_path without sdk_cwd).  Regular files live on the
# sandbox, not the host.
# ---------------------------------------------------------------------------


class TestReadLocal:
    _CONV_UUID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

    def _make_tool_results_file(self, encoded: str, filename: str, content: str) -> str:
        """Create a tool-results file under <encoded>/<uuid>/tool-results/."""
        tool_results_dir = os.path.join(
            SDK_PROJECTS_DIR, encoded, self._CONV_UUID, "tool-results"
        )
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
        tool_results_dir = os.path.join(
            SDK_PROJECTS_DIR, encoded, self._CONV_UUID, "tool-results"
        )
        os.makedirs(tool_results_dir, exist_ok=True)
        filepath = os.path.join(tool_results_dir, "nonexistent.txt")
        token = _current_project_dir.set(encoded)
        try:
            result = _read_local(filepath, offset=0, limit=10)
            assert result["isError"] is True
            assert "not found" in result["content"][0]["text"].lower()
        finally:
            _current_project_dir.reset(token)
            shutil.rmtree(os.path.join(SDK_PROJECTS_DIR, encoded), ignore_errors=True)

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
# _check_sandbox_symlink_escape — symlink escape detection
# ---------------------------------------------------------------------------


def _make_sandbox(stdout: str, exit_code: int = 0) -> SimpleNamespace:
    """Build a minimal sandbox mock whose commands.run returns a fixed result."""
    run_result = SimpleNamespace(stdout=stdout, exit_code=exit_code)
    commands = SimpleNamespace(run=AsyncMock(return_value=run_result))
    return SimpleNamespace(commands=commands)


class TestCheckSandboxSymlinkEscape:
    @pytest.mark.asyncio
    async def test_canonical_path_within_workdir_returns_path(self):
        """When readlink -f resolves to a path inside E2B_WORKDIR, returns it."""
        sandbox = _make_sandbox(stdout=f"{E2B_WORKDIR}/src\n", exit_code=0)
        result = await _check_sandbox_symlink_escape(sandbox, f"{E2B_WORKDIR}/src")
        assert result == f"{E2B_WORKDIR}/src"

    @pytest.mark.asyncio
    async def test_workdir_itself_returns_workdir(self):
        """When readlink -f resolves to E2B_WORKDIR exactly, returns E2B_WORKDIR."""
        sandbox = _make_sandbox(stdout=f"{E2B_WORKDIR}\n", exit_code=0)
        result = await _check_sandbox_symlink_escape(sandbox, E2B_WORKDIR)
        assert result == E2B_WORKDIR

    @pytest.mark.asyncio
    async def test_symlink_escape_returns_none(self):
        """When readlink -f resolves outside E2B_WORKDIR (symlink escape), returns None."""
        sandbox = _make_sandbox(stdout="/etc\n", exit_code=0)
        result = await _check_sandbox_symlink_escape(sandbox, f"{E2B_WORKDIR}/evil")
        assert result is None

    @pytest.mark.asyncio
    async def test_nonzero_exit_code_returns_none(self):
        """A non-zero exit code from readlink -f returns None."""
        sandbox = _make_sandbox(stdout="", exit_code=1)
        result = await _check_sandbox_symlink_escape(sandbox, f"{E2B_WORKDIR}/src")
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_stdout_returns_none(self):
        """Empty stdout from readlink (e.g. path doesn't exist yet) returns None."""
        sandbox = _make_sandbox(stdout="", exit_code=0)
        result = await _check_sandbox_symlink_escape(sandbox, f"{E2B_WORKDIR}/src")
        assert result is None

    @pytest.mark.asyncio
    async def test_prefix_collision_returns_none(self):
        """A path prefixed with E2B_WORKDIR but not within it is rejected."""
        sandbox = _make_sandbox(stdout=f"{E2B_WORKDIR}-evil\n", exit_code=0)
        result = await _check_sandbox_symlink_escape(sandbox, f"{E2B_WORKDIR}-evil")
        assert result is None

    @pytest.mark.asyncio
    async def test_deeply_nested_path_within_workdir(self):
        """Deep nested paths inside E2B_WORKDIR are allowed."""
        sandbox = _make_sandbox(stdout=f"{E2B_WORKDIR}/a/b/c/d\n", exit_code=0)
        result = await _check_sandbox_symlink_escape(sandbox, f"{E2B_WORKDIR}/a/b/c/d")
        assert result == f"{E2B_WORKDIR}/a/b/c/d"

    @pytest.mark.asyncio
    async def test_tmp_path_allowed(self):
        """Paths resolving to /tmp are allowed."""
        sandbox = _make_sandbox(stdout="/tmp/workdir\n", exit_code=0)
        result = await _check_sandbox_symlink_escape(sandbox, "/tmp/workdir")
        assert result == "/tmp/workdir"

    @pytest.mark.asyncio
    async def test_tmp_itself_allowed(self):
        """The /tmp directory itself is allowed."""
        sandbox = _make_sandbox(stdout="/tmp\n", exit_code=0)
        result = await _check_sandbox_symlink_escape(sandbox, "/tmp")
        assert result == "/tmp"


# ---------------------------------------------------------------------------
# _sandbox_write — routing writes through shell for /tmp paths
# ---------------------------------------------------------------------------


class TestSandboxWrite:
    @pytest.mark.asyncio
    async def test_tmp_path_uses_shell_command(self):
        """Writes to /tmp should use commands.run (shell) instead of files.write."""
        run_result = SimpleNamespace(stdout="", stderr="", exit_code=0)
        commands = SimpleNamespace(run=AsyncMock(return_value=run_result))
        files = SimpleNamespace(write=AsyncMock())
        sandbox = SimpleNamespace(commands=commands, files=files)

        await _sandbox_write(sandbox, "/tmp/test.py", "print('hello')")

        commands.run.assert_called_once()
        files.write.assert_not_called()

    @pytest.mark.asyncio
    async def test_home_user_path_uses_files_api(self):
        """Writes to /home/user should use sandbox.files.write."""
        run_result = SimpleNamespace(stdout="", stderr="", exit_code=0)
        commands = SimpleNamespace(run=AsyncMock(return_value=run_result))
        files = SimpleNamespace(write=AsyncMock())
        sandbox = SimpleNamespace(commands=commands, files=files)

        await _sandbox_write(sandbox, "/home/user/test.py", "print('hello')")

        files.write.assert_called_once_with("/home/user/test.py", "print('hello')")
        commands.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_tmp_nested_path_uses_shell_command(self):
        """Writes to nested /tmp paths should use commands.run."""
        run_result = SimpleNamespace(stdout="", stderr="", exit_code=0)
        commands = SimpleNamespace(run=AsyncMock(return_value=run_result))
        files = SimpleNamespace(write=AsyncMock())
        sandbox = SimpleNamespace(commands=commands, files=files)

        await _sandbox_write(sandbox, "/tmp/subdir/file.txt", "content")

        commands.run.assert_called_once()
        files.write.assert_not_called()

    @pytest.mark.asyncio
    async def test_tmp_write_shell_failure_raises(self):
        """Shell write failure should raise RuntimeError."""
        run_result = SimpleNamespace(stdout="", stderr="No space left", exit_code=1)
        commands = SimpleNamespace(run=AsyncMock(return_value=run_result))
        sandbox = SimpleNamespace(commands=commands)

        with pytest.raises(RuntimeError, match="shell write failed"):
            await _sandbox_write(sandbox, "/tmp/test.txt", "content")

    @pytest.mark.asyncio
    async def test_tmp_write_preserves_content_with_special_chars(self):
        """Content with special shell characters should be preserved via base64."""
        import base64

        run_result = SimpleNamespace(stdout="", stderr="", exit_code=0)
        commands = SimpleNamespace(run=AsyncMock(return_value=run_result))
        sandbox = SimpleNamespace(commands=commands)

        content = "print(\"Hello $USER\")\n# a `backtick` and 'quotes'\n"
        await _sandbox_write(sandbox, "/tmp/special.py", content)

        # Verify the command contains base64-encoded content
        call_args = commands.run.call_args[0][0]
        # Extract the base64 string from the command
        encoded_in_cmd = call_args.split("echo ")[1].split(" |")[0].strip("'")
        decoded = base64.b64decode(encoded_in_cmd).decode()
        assert decoded == content
