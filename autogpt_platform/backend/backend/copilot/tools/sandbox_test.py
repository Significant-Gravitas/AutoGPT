from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from backend.copilot.tools import sandbox
from backend.copilot.tools.sandbox import (
    _build_bwrap_command,
    _build_vetto_command,
    get_sandbox_backend,
    has_full_sandbox,
    make_session_path,
    run_sandboxed,
)


def test_make_session_path_valid():
    path = make_session_path("session-123_abc")
    assert path.startswith(sandbox.WORKSPACE_PREFIX)
    assert "session-123abc" in path


def test_make_session_path_traversal_prevention():
    path = make_session_path("../../etc/passwd")
    assert path.startswith(sandbox.WORKSPACE_PREFIX)
    assert "etcpasswd" in path


def test_get_sandbox_backend_vetto():
    with patch(
        "shutil.which",
        side_effect=lambda bin: "/usr/bin/vetto" if bin == "vetto" else None,
    ):
        sandbox._SANDBOX_BACKEND = None
        backend = get_sandbox_backend()
        assert backend == "vetto"
        assert has_full_sandbox() is True


def test_get_sandbox_backend_bwrap_fallback():
    def mock_which(bin):
        if bin == "bwrap":
            return "/usr/bin/bwrap"
        return None

    with patch("shutil.which", side_effect=mock_which), patch(
        "platform.system", return_value="Linux"
    ):
        sandbox._SANDBOX_BACKEND = None
        backend = get_sandbox_backend()
        assert backend == "bwrap"
        assert has_full_sandbox() is True


def test_get_sandbox_backend_none():
    with patch("shutil.which", return_value=None):
        sandbox._SANDBOX_BACKEND = None
        backend = get_sandbox_backend()
        assert backend is None
        assert has_full_sandbox() is False


def test_build_vetto_command():
    command = ["python", "-c", "print('hello')"]
    cwd = "/tmp/copilot-test"
    env = {"SAFE_VAR": "val", "PATH": "/bin"}
    timeout = 30

    cmd = _build_vetto_command(command, cwd, env, timeout)
    assert cmd[0] == "vetto"
    assert cmd[1] == "run"
    assert "--workspace" in cmd
    assert cwd in cmd
    assert "--timeout=30s" in cmd
    assert "--net=off" in cmd
    assert "--env" in cmd
    assert "SAFE_VAR=val" in cmd
    assert "--" in cmd
    assert cmd[-3:] == ["python", "-c", "print('hello')"]


def test_build_bwrap_command():
    command = ["echo", "test"]
    cwd = "/tmp/copilot-test"
    env = {"KEY": "VALUE"}

    cmd = _build_bwrap_command(command, cwd, env)
    assert cmd[0] == "bwrap"
    assert "--unshare-user" in cmd
    assert "--clearenv" in cmd
    assert "--setenv" in cmd
    assert "KEY" in cmd
    assert "VALUE" in cmd
    assert "--unshare-net" in cmd


@pytest.mark.asyncio
async def test_run_sandboxed_raises_when_no_backend():
    with patch("backend.copilot.tools.sandbox.get_sandbox_backend", return_value=None):
        with pytest.raises(RuntimeError) as exc_info:
            await run_sandboxed(["echo", "1"], cwd="/tmp")
        assert "requires a sandbox runtime" in str(exc_info.value)


@pytest.mark.asyncio
async def test_run_sandboxed_invokes_vetto():
    mock_proc = MagicMock()
    mock_proc.communicate = AsyncMock(return_value=(b"vetto stdout\n", b""))
    mock_proc.returncode = 0
    mock_proc.pid = 9999

    with patch(
        "backend.copilot.tools.sandbox.get_sandbox_backend", return_value="vetto"
    ), patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
        stdout, stderr, exit_code, timed_out = await run_sandboxed(
            ["echo", "vetto-test"],
            cwd="/tmp/copilot-test",
            timeout=20,
        )

        assert mock_exec.called
        call_args = mock_exec.call_args[0]
        assert call_args[0] == "vetto"
        assert call_args[1] == "run"
        assert stdout == "vetto stdout\n"
        assert exit_code == 0
        assert timed_out is False
