"""Tests for unified file-tool handlers (E2B + non-E2B), path validation,
local read safety, truncation detection, and per-path edit locking.

Pure unit tests with no external dependencies (no E2B, no sandbox).
"""

import hashlib
import os
import shutil
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.copilot.context import E2B_WORKDIR, SDK_PROJECTS_DIR, _current_project_dir
from backend.copilot.sdk.tool_adapter import SDK_DISALLOWED_TOOLS

from .e2b_file_tools import (
    _BRIDGE_SHELL_MAX_BYTES,
    _BRIDGE_SKIP_BYTES,
    _DEFAULT_READ_LIMIT,
    _LARGE_CONTENT_WARN_CHARS,
    EDIT_TOOL_NAME,
    EDIT_TOOL_SCHEMA,
    READ_TOOL_NAME,
    READ_TOOL_SCHEMA,
    WRITE_TOOL_NAME,
    WRITE_TOOL_SCHEMA,
    _check_sandbox_symlink_escape,
    _edit_locks,
    _handle_edit_file,
    _handle_read_file,
    _handle_write_file,
    _read_local,
    _sandbox_write,
    bridge_and_annotate,
    bridge_to_sandbox,
    resolve_sandbox_path,
)


@pytest.fixture(autouse=True)
def _clear_edit_locks():
    """Clear the module-level _edit_locks dict between tests to prevent bleed."""
    _edit_locks.clear()
    yield
    _edit_locks.clear()


def _expected_bridge_path(file_path: str, prefix: str = "/tmp") -> str:
    """Compute the expected sandbox path for a bridged file."""
    expanded = os.path.realpath(os.path.expanduser(file_path))
    basename = os.path.basename(expanded)
    source_id = hashlib.sha256(expanded.encode()).hexdigest()[:12]
    return f"{prefix}/{source_id}-{basename}"


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
# In E2B mode, _read_local only allows tool-results/tool-outputs paths
# (via is_allowed_local_path without sdk_cwd).  Regular files live on
# the sandbox, not the host.
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
            result = _read_local(filepath, offset=0, limit=_DEFAULT_READ_LIMIT)
            assert result["isError"] is False
            assert "line 1" in result["content"][0]["text"]
            assert "line 2" in result["content"][0]["text"]
        finally:
            _current_project_dir.reset(token)
            os.unlink(filepath)

    def test_read_tool_outputs_file(self):
        """Reading a tool-outputs file should also succeed."""
        encoded = "-tmp-copilot-e2b-test-read-outputs"
        tool_outputs_dir = os.path.join(
            SDK_PROJECTS_DIR, encoded, self._CONV_UUID, "tool-outputs"
        )
        os.makedirs(tool_outputs_dir, exist_ok=True)
        filepath = os.path.join(tool_outputs_dir, "sdk-abc123.json")
        with open(filepath, "w") as f:
            f.write('{"data": "test"}\n')
        token = _current_project_dir.set(encoded)
        try:
            result = _read_local(filepath, offset=0, limit=_DEFAULT_READ_LIMIT)
            assert result["isError"] is False
            assert "test" in result["content"][0]["text"]
        finally:
            _current_project_dir.reset(token)
            shutil.rmtree(os.path.join(SDK_PROJECTS_DIR, encoded), ignore_errors=True)

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


# ---------------------------------------------------------------------------
# bridge_to_sandbox — copy SDK-internal files into E2B sandbox
# ---------------------------------------------------------------------------


def _make_bridge_sandbox() -> SimpleNamespace:
    """Build a sandbox mock suitable for bridge_to_sandbox tests."""
    run_result = SimpleNamespace(stdout="", stderr="", exit_code=0)
    commands = SimpleNamespace(run=AsyncMock(return_value=run_result))
    files = SimpleNamespace(write=AsyncMock())
    return SimpleNamespace(commands=commands, files=files)


class TestBridgeToSandbox:
    @pytest.mark.asyncio
    async def test_happy_path_small_file(self, tmp_path):
        """A small file is bridged to /tmp/<hash>-<basename> via _sandbox_write."""
        f = tmp_path / "result.json"
        f.write_text('{"ok": true}')
        sandbox = _make_bridge_sandbox()

        result = await bridge_to_sandbox(
            sandbox, str(f), offset=0, limit=_DEFAULT_READ_LIMIT
        )

        expected = _expected_bridge_path(str(f))
        assert result == expected
        sandbox.commands.run.assert_called_once()
        cmd = sandbox.commands.run.call_args[0][0]
        assert "result.json" in cmd
        sandbox.files.write.assert_not_called()

    @pytest.mark.asyncio
    async def test_skip_when_offset_nonzero(self, tmp_path):
        """Bridging is skipped when offset != 0 (partial read)."""
        f = tmp_path / "data.txt"
        f.write_text("content")
        sandbox = _make_bridge_sandbox()

        result = await bridge_to_sandbox(
            sandbox, str(f), offset=10, limit=_DEFAULT_READ_LIMIT
        )

        assert result is None
        sandbox.commands.run.assert_not_called()
        sandbox.files.write.assert_not_called()

    @pytest.mark.asyncio
    async def test_skip_when_limit_too_small(self, tmp_path):
        """Bridging is skipped when limit < _DEFAULT_READ_LIMIT (partial read)."""
        f = tmp_path / "data.txt"
        f.write_text("content")
        sandbox = _make_bridge_sandbox()

        await bridge_to_sandbox(sandbox, str(f), offset=0, limit=100)

        sandbox.commands.run.assert_not_called()
        sandbox.files.write.assert_not_called()

    @pytest.mark.asyncio
    async def test_nonexistent_file_does_not_raise(self, tmp_path):
        """Bridging a non-existent file logs but does not propagate errors."""
        sandbox = _make_bridge_sandbox()

        await bridge_to_sandbox(
            sandbox, str(tmp_path / "ghost.txt"), offset=0, limit=_DEFAULT_READ_LIMIT
        )

        sandbox.commands.run.assert_not_called()
        sandbox.files.write.assert_not_called()

    @pytest.mark.asyncio
    async def test_sandbox_write_failure_returns_none(self, tmp_path):
        """If sandbox write fails, returns None (best-effort)."""
        f = tmp_path / "data.txt"
        f.write_text("content")
        sandbox = _make_bridge_sandbox()
        sandbox.commands.run.side_effect = RuntimeError("E2B timeout")

        result = await bridge_to_sandbox(
            sandbox, str(f), offset=0, limit=_DEFAULT_READ_LIMIT
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_large_file_uses_files_api(self, tmp_path):
        """Files > 32 KB but <= 50 MB are written to /home/user/ via files.write."""
        f = tmp_path / "big.json"
        f.write_bytes(b"x" * (_BRIDGE_SHELL_MAX_BYTES + 1))
        sandbox = _make_bridge_sandbox()

        result = await bridge_to_sandbox(
            sandbox, str(f), offset=0, limit=_DEFAULT_READ_LIMIT
        )

        expected = _expected_bridge_path(str(f), prefix="/home/user")
        assert result == expected
        sandbox.files.write.assert_called_once()
        call_args = sandbox.files.write.call_args[0]
        assert call_args[0] == expected
        sandbox.commands.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_small_binary_file_preserves_bytes(self, tmp_path):
        """A small binary file is bridged to /tmp via base64 without corruption."""
        binary_data = bytes(range(256))
        f = tmp_path / "image.png"
        f.write_bytes(binary_data)
        sandbox = _make_bridge_sandbox()

        result = await bridge_to_sandbox(
            sandbox, str(f), offset=0, limit=_DEFAULT_READ_LIMIT
        )

        expected = _expected_bridge_path(str(f))
        assert result == expected
        sandbox.commands.run.assert_called_once()
        cmd = sandbox.commands.run.call_args[0][0]
        assert "base64" in cmd
        sandbox.files.write.assert_not_called()

    @pytest.mark.asyncio
    async def test_large_binary_file_writes_raw_bytes(self, tmp_path):
        """A large binary file is bridged to /home/user/ as raw bytes."""
        binary_data = bytes(range(256)) * 200
        f = tmp_path / "photo.jpg"
        f.write_bytes(binary_data)
        sandbox = _make_bridge_sandbox()

        result = await bridge_to_sandbox(
            sandbox, str(f), offset=0, limit=_DEFAULT_READ_LIMIT
        )

        expected = _expected_bridge_path(str(f), prefix="/home/user")
        assert result == expected
        sandbox.files.write.assert_called_once()
        call_args = sandbox.files.write.call_args[0]
        assert call_args[0] == expected
        assert call_args[1] == binary_data
        sandbox.commands.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_very_large_file_skipped(self, tmp_path):
        """Files > 50 MB are skipped entirely."""
        f = tmp_path / "huge.bin"
        # Create a sparse file to avoid actually writing 50 MB
        with open(f, "wb") as fh:
            fh.seek(_BRIDGE_SKIP_BYTES + 1)
            fh.write(b"\0")
        sandbox = _make_bridge_sandbox()

        result = await bridge_to_sandbox(
            sandbox, str(f), offset=0, limit=_DEFAULT_READ_LIMIT
        )

        assert result is None

        sandbox.commands.run.assert_not_called()
        sandbox.files.write.assert_not_called()


# ---------------------------------------------------------------------------
# bridge_and_annotate — shared helper wrapping bridge_to_sandbox + annotation
# ---------------------------------------------------------------------------


class TestBridgeAndAnnotate:
    @pytest.mark.asyncio
    async def test_returns_annotation_on_success(self, tmp_path):
        """On success, returns a newline-prefixed annotation with the sandbox path."""
        f = tmp_path / "data.json"
        f.write_text('{"ok": true}')
        sandbox = _make_bridge_sandbox()

        annotation = await bridge_and_annotate(
            sandbox, str(f), offset=0, limit=_DEFAULT_READ_LIMIT
        )

        expected_path = _expected_bridge_path(str(f))
        assert annotation == f"\n[Sandbox copy available at {expected_path}]"

    @pytest.mark.asyncio
    async def test_returns_none_when_skipped(self, tmp_path):
        """When bridging is skipped (e.g. offset != 0), returns None."""
        f = tmp_path / "data.json"
        f.write_text("content")
        sandbox = _make_bridge_sandbox()

        annotation = await bridge_and_annotate(
            sandbox, str(f), offset=10, limit=_DEFAULT_READ_LIMIT
        )

        assert annotation is None


# ===========================================================================
# Non-E2B (local SDK working dir) tests — ported from file_tools_test.py
# ===========================================================================


@pytest.fixture
def sdk_cwd(tmp_path, monkeypatch):
    """Provide a temporary SDK working directory with no sandbox."""
    cwd = str(tmp_path / "copilot-test-session")
    os.makedirs(cwd, exist_ok=True)
    monkeypatch.setattr("backend.copilot.sdk.e2b_file_tools.get_sdk_cwd", lambda: cwd)
    # Ensure no sandbox is returned (non-E2B mode)
    monkeypatch.setattr(
        "backend.copilot.sdk.e2b_file_tools.get_current_sandbox", lambda: None
    )
    monkeypatch.setattr("backend.copilot.sdk.e2b_file_tools._get_sandbox", lambda: None)

    def _patched_is_allowed(path: str, cwd_arg: str | None = None) -> bool:
        resolved = os.path.realpath(path)
        norm_cwd = os.path.realpath(cwd)
        return resolved == norm_cwd or resolved.startswith(norm_cwd + os.sep)

    monkeypatch.setattr(
        "backend.copilot.sdk.e2b_file_tools.is_allowed_local_path",
        _patched_is_allowed,
    )
    return cwd


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestWriteToolSchema:
    def test_file_path_is_first_property(self):
        """file_path should be listed first in schema so truncation preserves it."""
        props = list(WRITE_TOOL_SCHEMA["properties"].keys())
        assert props[0] == "file_path"

    def test_no_required_in_schema(self):
        """required is omitted so MCP SDK does not reject truncated calls."""
        assert "required" not in WRITE_TOOL_SCHEMA


# ---------------------------------------------------------------------------
# Normal write (non-E2B)
# ---------------------------------------------------------------------------


class TestNormalWrite:
    @pytest.mark.asyncio
    async def test_write_creates_file(self, sdk_cwd):
        result = await _handle_write_file(
            {"file_path": "hello.txt", "content": "Hello, world!"}
        )
        assert not result["isError"]
        written = open(os.path.join(sdk_cwd, "hello.txt")).read()
        assert written == "Hello, world!"

    @pytest.mark.asyncio
    async def test_write_creates_parent_dirs(self, sdk_cwd):
        result = await _handle_write_file(
            {"file_path": "sub/dir/file.py", "content": "print('hi')"}
        )
        assert not result["isError"]
        assert os.path.isfile(os.path.join(sdk_cwd, "sub", "dir", "file.py"))

    @pytest.mark.asyncio
    async def test_write_absolute_path_within_cwd(self, sdk_cwd):
        abs_path = os.path.join(sdk_cwd, "abs.txt")
        result = await _handle_write_file(
            {"file_path": abs_path, "content": "absolute"}
        )
        assert not result["isError"]
        assert open(abs_path).read() == "absolute"

    @pytest.mark.asyncio
    async def test_success_message_contains_path(self, sdk_cwd):
        result = await _handle_write_file({"file_path": "msg.txt", "content": "ok"})
        text = result["content"][0]["text"]
        assert "Successfully wrote" in text
        assert "msg.txt" in text


# ---------------------------------------------------------------------------
# Large content warning
# ---------------------------------------------------------------------------


class TestLargeContentWarning:
    @pytest.mark.asyncio
    async def test_large_content_warns(self, sdk_cwd):
        big_content = "x" * (_LARGE_CONTENT_WARN_CHARS + 1)
        result = await _handle_write_file(
            {"file_path": "big.txt", "content": big_content}
        )
        assert not result["isError"]
        text = result["content"][0]["text"]
        assert "WARNING" in text
        assert "large" in text.lower()

    @pytest.mark.asyncio
    async def test_normal_content_no_warning(self, sdk_cwd):
        result = await _handle_write_file(
            {"file_path": "small.txt", "content": "small"}
        )
        text = result["content"][0]["text"]
        assert "WARNING" not in text


# ---------------------------------------------------------------------------
# Truncation detection
# ---------------------------------------------------------------------------


class TestWriteTruncationDetection:
    @pytest.mark.asyncio
    async def test_partial_truncation_content_no_path(self, sdk_cwd):
        """Simulates API truncating file_path but preserving content."""
        result = await _handle_write_file({"content": "some content here"})
        assert result["isError"]
        text = result["content"][0]["text"]
        assert "truncated" in text.lower()
        assert "file_path" in text.lower()

    @pytest.mark.asyncio
    async def test_complete_truncation_empty_args(self, sdk_cwd):
        """Simulates API truncating to empty args {}."""
        result = await _handle_write_file({})
        assert result["isError"]
        text = result["content"][0]["text"]
        assert "truncated" in text.lower()
        assert "smaller steps" in text.lower()

    @pytest.mark.asyncio
    async def test_empty_file_path_string(self, sdk_cwd):
        """Empty string file_path should trigger truncation error."""
        result = await _handle_write_file({"file_path": "", "content": "data"})
        assert result["isError"]


# ---------------------------------------------------------------------------
# Path validation (write)
# ---------------------------------------------------------------------------


class TestWritePathValidation:
    @pytest.mark.asyncio
    async def test_path_traversal_blocked(self, sdk_cwd):
        result = await _handle_write_file(
            {"file_path": "../../etc/passwd", "content": "evil"}
        )
        assert result["isError"]
        text = result["content"][0]["text"]
        assert "must be within" in text.lower()

    @pytest.mark.asyncio
    async def test_absolute_outside_cwd_blocked(self, sdk_cwd):
        result = await _handle_write_file(
            {"file_path": "/etc/passwd", "content": "evil"}
        )
        assert result["isError"]

    @pytest.mark.asyncio
    async def test_no_sdk_cwd_returns_error(self, monkeypatch):
        monkeypatch.setattr(
            "backend.copilot.sdk.e2b_file_tools.get_sdk_cwd", lambda: ""
        )
        monkeypatch.setattr(
            "backend.copilot.sdk.e2b_file_tools._get_sandbox", lambda: None
        )
        result = await _handle_write_file({"file_path": "test.txt", "content": "hi"})
        assert result["isError"]
        text = result["content"][0]["text"]
        assert "working directory" in text.lower()


# ---------------------------------------------------------------------------
# CLI built-in disallowed
# ---------------------------------------------------------------------------


class TestCliBuiltinDisallowed:
    def test_write_in_disallowed_tools(self):
        assert "Write" in SDK_DISALLOWED_TOOLS

    def test_tool_name_is_write(self):
        assert WRITE_TOOL_NAME == "Write"

    def test_edit_in_disallowed_tools(self):
        assert "Edit" in SDK_DISALLOWED_TOOLS


# ===========================================================================
# Read tool tests (non-E2B)
# ===========================================================================


class TestReadToolSchema:
    def test_file_path_is_first_property(self):
        props = list(READ_TOOL_SCHEMA["properties"].keys())
        assert props[0] == "file_path"

    def test_no_required_in_schema(self):
        """required is omitted so MCP SDK does not reject truncated calls."""
        assert "required" not in READ_TOOL_SCHEMA

    def test_tool_name_is_read_file(self):
        assert READ_TOOL_NAME == "read_file"


class TestNormalRead:
    @pytest.mark.asyncio
    async def test_read_file(self, sdk_cwd):
        path = os.path.join(sdk_cwd, "hello.txt")
        with open(path, "w") as f:
            f.write("line1\nline2\nline3\n")
        result = await _handle_read_file({"file_path": "hello.txt"})
        assert not result["isError"]
        text = result["content"][0]["text"]
        assert "line1" in text
        assert "line2" in text
        assert "line3" in text

    @pytest.mark.asyncio
    async def test_read_with_line_numbers(self, sdk_cwd):
        path = os.path.join(sdk_cwd, "numbered.txt")
        with open(path, "w") as f:
            f.write("alpha\nbeta\ngamma\n")
        result = await _handle_read_file({"file_path": "numbered.txt"})
        text = result["content"][0]["text"]
        assert "1\t" in text
        assert "2\t" in text
        assert "3\t" in text

    @pytest.mark.asyncio
    async def test_read_absolute_path_within_cwd(self, sdk_cwd):
        path = os.path.join(sdk_cwd, "abs.txt")
        with open(path, "w") as f:
            f.write("absolute content")
        result = await _handle_read_file({"file_path": path})
        assert not result["isError"]
        assert "absolute content" in result["content"][0]["text"]


class TestReadOffsetLimit:
    @pytest.mark.asyncio
    async def test_read_with_offset(self, sdk_cwd):
        path = os.path.join(sdk_cwd, "lines.txt")
        with open(path, "w") as f:
            for i in range(10):
                f.write(f"line{i}\n")
        result = await _handle_read_file(
            {"file_path": "lines.txt", "offset": 5, "limit": 3}
        )
        text = result["content"][0]["text"]
        assert "line5" in text
        assert "line6" in text
        assert "line7" in text
        assert "line4" not in text
        assert "line8" not in text

    @pytest.mark.asyncio
    async def test_read_with_limit(self, sdk_cwd):
        path = os.path.join(sdk_cwd, "many.txt")
        with open(path, "w") as f:
            for i in range(100):
                f.write(f"line{i}\n")
        result = await _handle_read_file({"file_path": "many.txt", "limit": 2})
        text = result["content"][0]["text"]
        assert "line0" in text
        assert "line1" in text
        assert "line2" not in text

    @pytest.mark.asyncio
    async def test_offset_line_numbers_are_correct(self, sdk_cwd):
        path = os.path.join(sdk_cwd, "offset_nums.txt")
        with open(path, "w") as f:
            for i in range(10):
                f.write(f"line{i}\n")
        result = await _handle_read_file(
            {"file_path": "offset_nums.txt", "offset": 3, "limit": 2}
        )
        text = result["content"][0]["text"]
        assert "4\t" in text
        assert "5\t" in text


class TestReadInvalidOffsetLimit:
    @pytest.mark.asyncio
    async def test_non_integer_offset(self, sdk_cwd):
        path = os.path.join(sdk_cwd, "valid.txt")
        with open(path, "w") as f:
            f.write("content\n")
        result = await _handle_read_file({"file_path": "valid.txt", "offset": "abc"})
        assert result["isError"]
        text = result["content"][0]["text"]
        assert "invalid" in text.lower()

    @pytest.mark.asyncio
    async def test_non_integer_limit(self, sdk_cwd):
        path = os.path.join(sdk_cwd, "valid.txt")
        with open(path, "w") as f:
            f.write("content\n")
        result = await _handle_read_file({"file_path": "valid.txt", "limit": "xyz"})
        assert result["isError"]
        text = result["content"][0]["text"]
        assert "invalid" in text.lower()


class TestReadFileNotFound:
    @pytest.mark.asyncio
    async def test_file_not_found(self, sdk_cwd):
        result = await _handle_read_file({"file_path": "nonexistent.txt"})
        assert result["isError"]
        text = result["content"][0]["text"]
        assert "not found" in text.lower()


class TestReadPathTraversal:
    @pytest.mark.asyncio
    async def test_path_traversal_blocked(self, sdk_cwd):
        result = await _handle_read_file({"file_path": "../../etc/passwd"})
        assert result["isError"]
        text = result["content"][0]["text"]
        assert "must be within" in text.lower()

    @pytest.mark.asyncio
    async def test_absolute_outside_cwd_blocked(self, sdk_cwd):
        result = await _handle_read_file({"file_path": "/etc/passwd"})
        assert result["isError"]


class TestReadBinaryFile:
    @pytest.mark.asyncio
    async def test_binary_file_rejected(self, sdk_cwd):
        path = os.path.join(sdk_cwd, "image.png")
        with open(path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n")
        result = await _handle_read_file({"file_path": "image.png"})
        assert result["isError"]
        text = result["content"][0]["text"]
        assert "binary" in text.lower()

    @pytest.mark.asyncio
    async def test_text_file_not_rejected_as_binary(self, sdk_cwd):
        path = os.path.join(sdk_cwd, "code.py")
        with open(path, "w") as f:
            f.write("print('hello')\n")
        result = await _handle_read_file({"file_path": "code.py"})
        assert not result["isError"]


class TestReadTruncationDetection:
    @pytest.mark.asyncio
    async def test_truncation_offset_without_file_path(self, sdk_cwd):
        """offset present but file_path missing — truncated call."""
        result = await _handle_read_file({"offset": 5})
        assert result["isError"]
        text = result["content"][0]["text"]
        assert "truncated" in text.lower()

    @pytest.mark.asyncio
    async def test_truncation_limit_without_file_path(self, sdk_cwd):
        """limit present but file_path missing — truncated call."""
        result = await _handle_read_file({"limit": 100})
        assert result["isError"]
        text = result["content"][0]["text"]
        assert "truncated" in text.lower()

    @pytest.mark.asyncio
    async def test_no_truncation_plain_empty(self, sdk_cwd):
        """Empty args — treated as complete truncation."""
        result = await _handle_read_file({})
        assert result["isError"]
        text = result["content"][0]["text"]
        assert "truncated" in text.lower() or "empty arguments" in text.lower()


class TestReadEmptyFilePath:
    @pytest.mark.asyncio
    async def test_empty_file_path(self, sdk_cwd):
        result = await _handle_read_file({"file_path": ""})
        assert result["isError"]

    @pytest.mark.asyncio
    async def test_no_sdk_cwd(self, monkeypatch):
        monkeypatch.setattr(
            "backend.copilot.sdk.e2b_file_tools.get_sdk_cwd", lambda: ""
        )
        monkeypatch.setattr(
            "backend.copilot.sdk.e2b_file_tools._get_sandbox", lambda: None
        )
        monkeypatch.setattr(
            "backend.copilot.sdk.e2b_file_tools._is_allowed_local",
            lambda p: False,
        )
        result = await _handle_read_file({"file_path": "test.txt"})
        assert result["isError"]
        assert "working directory" in result["content"][0]["text"].lower()


# ===========================================================================
# Edit tool tests (non-E2B)
# ===========================================================================


class TestEditToolSchema:
    def test_file_path_is_first_property(self):
        props = list(EDIT_TOOL_SCHEMA["properties"].keys())
        assert props[0] == "file_path"

    def test_no_required_in_schema(self):
        """required is omitted so MCP SDK does not reject truncated calls."""
        assert "required" not in EDIT_TOOL_SCHEMA

    def test_tool_name_is_edit(self):
        assert EDIT_TOOL_NAME == "Edit"


class TestNormalEdit:
    @pytest.mark.asyncio
    async def test_simple_replacement(self, sdk_cwd):
        path = os.path.join(sdk_cwd, "edit_me.txt")
        with open(path, "w") as f:
            f.write("Hello World\n")
        result = await _handle_edit_file(
            {"file_path": "edit_me.txt", "old_string": "World", "new_string": "Earth"}
        )
        assert not result["isError"]
        content = open(path).read()
        assert content == "Hello Earth\n"

    @pytest.mark.asyncio
    async def test_edit_reports_replacement_count(self, sdk_cwd):
        path = os.path.join(sdk_cwd, "count.txt")
        with open(path, "w") as f:
            f.write("one two three\n")
        result = await _handle_edit_file(
            {"file_path": "count.txt", "old_string": "two", "new_string": "2"}
        )
        text = result["content"][0]["text"]
        assert "1 replacement" in text

    @pytest.mark.asyncio
    async def test_edit_absolute_path(self, sdk_cwd):
        path = os.path.join(sdk_cwd, "abs_edit.txt")
        with open(path, "w") as f:
            f.write("before\n")
        result = await _handle_edit_file(
            {"file_path": path, "old_string": "before", "new_string": "after"}
        )
        assert not result["isError"]
        assert open(path).read() == "after\n"


class TestEditOldStringNotFound:
    @pytest.mark.asyncio
    async def test_old_string_not_found(self, sdk_cwd):
        path = os.path.join(sdk_cwd, "nope.txt")
        with open(path, "w") as f:
            f.write("Hello World\n")
        result = await _handle_edit_file(
            {"file_path": "nope.txt", "old_string": "MISSING", "new_string": "x"}
        )
        assert result["isError"]
        text = result["content"][0]["text"]
        assert "not found" in text.lower()


class TestEditOldStringNotUnique:
    @pytest.mark.asyncio
    async def test_not_unique_without_replace_all(self, sdk_cwd):
        path = os.path.join(sdk_cwd, "dup.txt")
        with open(path, "w") as f:
            f.write("foo bar foo baz\n")
        result = await _handle_edit_file(
            {"file_path": "dup.txt", "old_string": "foo", "new_string": "qux"}
        )
        assert result["isError"]
        text = result["content"][0]["text"]
        assert "2 times" in text
        assert open(path).read() == "foo bar foo baz\n"


class TestEditReplaceAll:
    @pytest.mark.asyncio
    async def test_replace_all(self, sdk_cwd):
        path = os.path.join(sdk_cwd, "all.txt")
        with open(path, "w") as f:
            f.write("foo bar foo baz foo\n")
        result = await _handle_edit_file(
            {
                "file_path": "all.txt",
                "old_string": "foo",
                "new_string": "qux",
                "replace_all": True,
            }
        )
        assert not result["isError"]
        content = open(path).read()
        assert content == "qux bar qux baz qux\n"
        text = result["content"][0]["text"]
        assert "3 replacement" in text


class TestEditPartialTruncation:
    @pytest.mark.asyncio
    async def test_partial_truncation(self, sdk_cwd):
        """file_path missing but old_string/new_string present."""
        result = await _handle_edit_file(
            {"old_string": "something", "new_string": "else"}
        )
        assert result["isError"]
        text = result["content"][0]["text"]
        assert "truncated" in text.lower()

    @pytest.mark.asyncio
    async def test_complete_truncation(self, sdk_cwd):
        result = await _handle_edit_file({})
        assert result["isError"]
        text = result["content"][0]["text"]
        assert "truncated" in text.lower()

    @pytest.mark.asyncio
    async def test_empty_file_path_with_content(self, sdk_cwd):
        result = await _handle_edit_file(
            {"file_path": "", "old_string": "x", "new_string": "y"}
        )
        assert result["isError"]


class TestEditPathTraversal:
    @pytest.mark.asyncio
    async def test_path_traversal_blocked(self, sdk_cwd):
        result = await _handle_edit_file(
            {
                "file_path": "../../etc/passwd",
                "old_string": "root",
                "new_string": "evil",
            }
        )
        assert result["isError"]
        text = result["content"][0]["text"]
        assert "must be within" in text.lower()

    @pytest.mark.asyncio
    async def test_absolute_outside_cwd_blocked(self, sdk_cwd):
        result = await _handle_edit_file(
            {
                "file_path": "/etc/passwd",
                "old_string": "root",
                "new_string": "evil",
            }
        )
        assert result["isError"]


class TestEditFileNotFound:
    @pytest.mark.asyncio
    async def test_file_not_found(self, sdk_cwd):
        result = await _handle_edit_file(
            {
                "file_path": "nonexistent.txt",
                "old_string": "x",
                "new_string": "y",
            }
        )
        assert result["isError"]
        text = result["content"][0]["text"]
        assert "not found" in text.lower()

    @pytest.mark.asyncio
    async def test_no_sdk_cwd(self, monkeypatch):
        monkeypatch.setattr(
            "backend.copilot.sdk.e2b_file_tools.get_sdk_cwd", lambda: ""
        )
        monkeypatch.setattr(
            "backend.copilot.sdk.e2b_file_tools._get_sandbox", lambda: None
        )
        result = await _handle_edit_file(
            {"file_path": "test.txt", "old_string": "x", "new_string": "y"}
        )
        assert result["isError"]
        assert "working directory" in result["content"][0]["text"].lower()


# ---------------------------------------------------------------------------
# Concurrent edit locking
# ---------------------------------------------------------------------------


class TestConcurrentEditLocking:
    @pytest.mark.asyncio
    async def test_concurrent_edits_are_serialised(self, sdk_cwd):
        """Two parallel Edit calls on the same file must not race.

        Each edit appends a unique line by replacing a sentinel. Without the
        per-path lock one update would silently overwrite the other; with the
        lock both replacements must be present in the final file.

        The handler yields via ``asyncio.sleep(0)`` between the read and write
        phases, allowing the event loop to schedule the second coroutine.  The
        per-path lock ensures the second edit cannot proceed until the first
        completes — without it, the test would fail because edit_b would read
        a stale file and overwrite edit_a's change.
        """
        import asyncio as _asyncio

        path = os.path.join(sdk_cwd, "concurrent.txt")
        with open(path, "w") as f:
            f.write("line1\nline2\n")

        # Two coroutines both replace a *different* substring — they must not
        # race through the read-modify-write cycle.
        async def edit_a():
            return await _handle_edit_file(
                {
                    "file_path": "concurrent.txt",
                    "old_string": "line1",
                    "new_string": "EDITED_A",
                }
            )

        async def edit_b():
            return await _handle_edit_file(
                {
                    "file_path": "concurrent.txt",
                    "old_string": "line2",
                    "new_string": "EDITED_B",
                }
            )

        results = await _asyncio.gather(edit_a(), edit_b())
        for r in results:
            assert not r["isError"], r["content"][0]["text"]

        final = open(path).read()
        assert "EDITED_A" in final
        assert "EDITED_B" in final


# ---------------------------------------------------------------------------
# E2B mode: relative paths are routed to the sandbox, not the host
# ---------------------------------------------------------------------------


class TestReadFileE2BRouting:
    """Verify that _handle_read_file routes correctly in E2B mode.

    When E2B is active, relative paths (e.g. "output.txt") resolve against
    sdk_cwd on the host via _is_allowed_local — but those files were written to
    the sandbox, not to sdk_cwd.  The fix: when E2B is active, only SDK-internal
    tool-results/tool-outputs paths are read from the host; everything else is
    routed to the sandbox.
    """

    @pytest.mark.asyncio
    async def test_relative_path_in_e2b_mode_goes_to_sandbox(
        self, monkeypatch, tmp_path
    ):
        """A plain relative path in E2B mode must be read from the sandbox, not the host."""
        cwd = str(tmp_path / "copilot-session")
        os.makedirs(cwd)

        # Set up sdk_cwd so _is_allowed_local would return True for "output.txt"
        monkeypatch.setattr(
            "backend.copilot.sdk.e2b_file_tools.get_sdk_cwd", lambda: cwd
        )
        monkeypatch.setattr(
            "backend.copilot.sdk.e2b_file_tools.is_allowed_local_path",
            lambda path, cwd_arg=None: os.path.realpath(
                os.path.join(cwd, path) if not os.path.isabs(path) else path
            ).startswith(os.path.realpath(cwd)),
        )

        # Create a sandbox mock that returns "sandbox content"
        sandbox = SimpleNamespace(
            files=SimpleNamespace(
                read=AsyncMock(return_value=b"sandbox content\n"),
                make_dir=AsyncMock(),
            ),
            commands=SimpleNamespace(run=AsyncMock()),
        )
        monkeypatch.setattr(
            "backend.copilot.sdk.e2b_file_tools._get_sandbox", lambda: sandbox
        )

        result = await _handle_read_file({"file_path": "output.txt"})

        # Should NOT be an error (file was read from sandbox)
        assert not result.get("isError"), result["content"][0]["text"]
        assert "sandbox content" in result["content"][0]["text"]
        # The sandbox files.read must have been called
        sandbox.files.read.assert_called_once()

    @pytest.mark.asyncio
    async def test_absolute_tmp_path_in_e2b_goes_to_sandbox(self, monkeypatch):
        """An absolute /tmp path (sdk_cwd-relative) in E2B mode is routed to the sandbox.

        sdk_cwd is always under /tmp in production (e.g. /tmp/copilot-<session>/).
        An absolute path like /tmp/copilot-xxx/result.txt must be read from the
        sandbox rather than the host even though _is_allowed_local would return True
        for it.
        """
        cwd = "/tmp/copilot-test-session-xyz"
        absolute_path = "/tmp/copilot-test-session-xyz/result.txt"

        monkeypatch.setattr(
            "backend.copilot.sdk.e2b_file_tools.get_sdk_cwd", lambda: cwd
        )
        # Simulate _is_allowed_local returning True for the path (as it would in prod)
        monkeypatch.setattr(
            "backend.copilot.sdk.e2b_file_tools.is_allowed_local_path",
            lambda path, cwd_arg=None: path.startswith(cwd),
        )

        sandbox = SimpleNamespace(
            files=SimpleNamespace(
                read=AsyncMock(return_value=b"sandbox result\n"),
                make_dir=AsyncMock(),
            ),
            commands=SimpleNamespace(run=AsyncMock()),
        )
        monkeypatch.setattr(
            "backend.copilot.sdk.e2b_file_tools._get_sandbox", lambda: sandbox
        )

        result = await _handle_read_file({"file_path": absolute_path})

        assert not result.get("isError"), result["content"][0]["text"]
        assert "sandbox result" in result["content"][0]["text"]
        sandbox.files.read.assert_called_once()
