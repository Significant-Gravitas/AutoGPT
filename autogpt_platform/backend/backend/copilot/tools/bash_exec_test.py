"""Tests for BashExecTool — E2B path with token injection."""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from e2b import CommandExitException

from ._test_data import make_session
from .bash_exec import MAX_BASH_TIMEOUT_SECONDS, BashExecTool
from .models import BashExecResponse, ErrorResponse

_USER = "user-bash-exec-test"


def _make_tool() -> BashExecTool:
    return BashExecTool()


def _make_sandbox(exit_code: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    result = MagicMock()
    result.exit_code = exit_code
    result.stdout = stdout
    result.stderr = stderr

    sandbox = MagicMock()
    handle = MagicMock()
    handle.wait = AsyncMock(return_value=result)
    handle.kill = AsyncMock(return_value=True)
    sandbox.command_handle = handle
    sandbox.commands.run = AsyncMock(return_value=handle)
    sandbox.kill = AsyncMock(return_value=True)
    return sandbox


def _lease_env(env: dict[str, str]):
    @asynccontextmanager
    async def lease():
        yield env

    return lease()


class TestBashExecE2BTokenInjection:
    @pytest.mark.asyncio(loop_scope="session")
    async def test_token_injected_when_user_id_set(self):
        """When user_id is provided, integration env vars are merged into sandbox envs."""
        tool = _make_tool()
        session = make_session(user_id=_USER)
        sandbox = _make_sandbox(stdout="ok")
        env_vars = {"GH_TOKEN": "gh-secret", "GITHUB_TOKEN": "gh-secret"}

        with (
            patch(
                "backend.copilot.tools.bash_exec.leased_integration_env_vars",
                side_effect=lambda _user_id, _lease_sink=None: _lease_env(env_vars),
            ) as mock_get_env,
            patch(
                "backend.copilot.tools.bash_exec.get_github_user_git_identity",
                new=AsyncMock(return_value=None),
            ),
        ):
            result = await tool._execute_on_e2b(
                sandbox=sandbox,
                command="echo hi",
                timeout=10,
                session_id=session.session_id,
                user_id=_USER,
            )

        mock_get_env.assert_called_once()
        assert mock_get_env.call_args.args[0] == _USER
        call_kwargs = sandbox.commands.run.call_args[1]
        assert call_kwargs["background"] is True
        assert call_kwargs["envs"]["GH_TOKEN"] == "gh-secret"
        assert call_kwargs["envs"]["GITHUB_TOKEN"] == "gh-secret"
        assert isinstance(result, BashExecResponse)

    @pytest.mark.asyncio(loop_scope="session")
    async def test_active_command_is_cancelled_when_credential_lease_is_lost(self):
        tool = _make_tool()
        session = make_session(user_id=_USER)
        started = asyncio.Event()
        cancelled = asyncio.Event()
        killed = asyncio.Event()
        released = asyncio.Event()
        failed = asyncio.Event()
        lease = MagicMock()

        async def wait_for_failure() -> None:
            await failed.wait()
            raise RuntimeError("credential lease heartbeat failed")

        lease.wait_for_failure = AsyncMock(side_effect=wait_for_failure)
        lease.validate = AsyncMock()

        @asynccontextmanager
        async def leased(_user_id: str, lease_sink: list):
            lease_sink.append(lease)
            try:
                yield {"GH_TOKEN": "gh-secret"}
            finally:
                assert killed.is_set()
                released.set()

        sandbox = MagicMock()

        async def paused_wait():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        handle = MagicMock()
        handle.wait = AsyncMock(side_effect=paused_wait)

        async def kill_command():
            killed.set()
            return True

        handle.kill = AsyncMock(side_effect=kill_command)
        sandbox.commands.run = AsyncMock(return_value=handle)
        sandbox.kill = AsyncMock(return_value=True)
        with (
            patch(
                "backend.copilot.tools.bash_exec.leased_integration_env_vars",
                side_effect=leased,
            ),
            patch(
                "backend.copilot.tools.bash_exec.get_github_user_git_identity",
                new=AsyncMock(return_value=None),
            ),
        ):
            execution = asyncio.create_task(
                tool._execute_on_e2b(
                    sandbox=sandbox,
                    command="sleep 120",
                    timeout=120,
                    session_id=session.session_id,
                    user_id=_USER,
                )
            )
            await started.wait()
            failed.set()
            result = await asyncio.wait_for(execution, timeout=1)

        assert isinstance(result, ErrorResponse)
        assert cancelled.is_set()
        assert killed.is_set()
        assert released.is_set()
        handle.kill.assert_awaited_once()
        sandbox.kill.assert_not_awaited()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_git_identity_set_from_github_profile(self):
        """When user has a connected GitHub account, git env vars are set from their profile."""
        tool = _make_tool()
        session = make_session(user_id=_USER)
        sandbox = _make_sandbox(stdout="ok")
        identity = {
            "GIT_AUTHOR_NAME": "Test User",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "Test User",
            "GIT_COMMITTER_EMAIL": "test@example.com",
        }

        with (
            patch(
                "backend.copilot.tools.bash_exec.leased_integration_env_vars",
                side_effect=lambda _user_id, _lease_sink=None: _lease_env({}),
            ),
            patch(
                "backend.copilot.tools.bash_exec.get_github_user_git_identity",
                new=AsyncMock(return_value=identity),
            ),
        ):
            await tool._execute_on_e2b(
                sandbox=sandbox,
                command="git commit -m test",
                timeout=10,
                session_id=session.session_id,
                user_id=_USER,
            )

        call_kwargs = sandbox.commands.run.call_args[1]
        assert call_kwargs["envs"]["GIT_AUTHOR_NAME"] == "Test User"
        assert call_kwargs["envs"]["GIT_AUTHOR_EMAIL"] == "test@example.com"
        assert call_kwargs["envs"]["GIT_COMMITTER_NAME"] == "Test User"
        assert call_kwargs["envs"]["GIT_COMMITTER_EMAIL"] == "test@example.com"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_no_git_identity_when_github_not_connected(self):
        """When user has no GitHub account, git identity env vars are absent."""
        tool = _make_tool()
        session = make_session(user_id=_USER)
        sandbox = _make_sandbox(stdout="ok")

        with (
            patch(
                "backend.copilot.tools.bash_exec.leased_integration_env_vars",
                side_effect=lambda _user_id, _lease_sink=None: _lease_env({}),
            ),
            patch(
                "backend.copilot.tools.bash_exec.get_github_user_git_identity",
                new=AsyncMock(return_value=None),
            ),
        ):
            await tool._execute_on_e2b(
                sandbox=sandbox,
                command="echo hi",
                timeout=10,
                session_id=session.session_id,
                user_id=_USER,
            )

        call_kwargs = sandbox.commands.run.call_args[1]
        assert "GIT_AUTHOR_NAME" not in call_kwargs["envs"]
        assert "GIT_COMMITTER_EMAIL" not in call_kwargs["envs"]

    @pytest.mark.asyncio(loop_scope="session")
    async def test_nonzero_exit_returned_as_bash_exec_response(self):
        """CommandExitException (non-zero exit) must become a BashExecResponse with scrubbed output."""
        tool = _make_tool()
        session = make_session(user_id=_USER)

        sandbox = _make_sandbox()
        sandbox.command_handle.wait = AsyncMock(
            side_effect=CommandExitException(
                stdout="not logged in gh-secret",
                stderr="oops gh-secret",
                exit_code=1,
                error=None,
            )
        )

        with (
            patch(
                "backend.copilot.tools.bash_exec.leased_integration_env_vars",
                side_effect=lambda _user_id, _lease_sink=None: _lease_env(
                    {"GH_TOKEN": "gh-secret"}
                ),
            ),
            patch(
                "backend.copilot.tools.bash_exec.get_github_user_git_identity",
                new=AsyncMock(return_value=None),
            ),
        ):
            result = await tool._execute_on_e2b(
                sandbox=sandbox,
                command="gh auth status 2>&1",
                timeout=10,
                session_id=session.session_id,
                user_id=_USER,
            )

        assert isinstance(result, BashExecResponse)
        assert result.exit_code == 1
        assert result.timed_out is False
        assert result.stdout == "not logged in [REDACTED]"
        assert result.stderr == "oops [REDACTED]"
        assert result.message == "Command executed with status code 1"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_no_token_injection_when_user_id_is_none(self):
        """When user_id is None, integration credentials must not be acquired."""
        tool = _make_tool()
        session = make_session(user_id=_USER)
        sandbox = _make_sandbox(stdout="ok")

        with (
            patch(
                "backend.copilot.tools.bash_exec.leased_integration_env_vars",
            ) as mock_get_env,
            patch(
                "backend.copilot.tools.bash_exec.get_github_user_git_identity",
                new=AsyncMock(return_value=None),
            ) as mock_get_identity,
        ):
            result = await tool._execute_on_e2b(
                sandbox=sandbox,
                command="echo hi",
                timeout=10,
                session_id=session.session_id,
                user_id=None,
            )

        mock_get_env.assert_not_called()
        mock_get_identity.assert_not_called()
        call_kwargs = sandbox.commands.run.call_args[1]
        assert "GH_TOKEN" not in call_kwargs["envs"]
        assert "GIT_AUTHOR_NAME" not in call_kwargs["envs"]
        assert isinstance(result, BashExecResponse)


class TestBashExecSdkToolResultRedirect:
    """A command that references an SDK tool-result path (e.g. the model
    tries to ``cat /root/.claude/projects/.../tool-results/foo.json``)
    must be short-circuited with a redirect to ``read_tool_result`` /
    ``@@agptfile`` before the sandbox returns the generic
    ``Permission denied`` that the model can't act on."""

    @pytest.mark.asyncio(loop_scope="session")
    async def test_redirect_on_absolute_sdk_path(self):
        tool = _make_tool()
        session = make_session(user_id=_USER)
        sandbox = _make_sandbox()
        cmd = (
            "cat /root/.claude/projects/-tmp-copilot-abc/"
            "abc/tool-results/toolu_x.json | jq ."
        )
        with patch(
            "backend.copilot.tools.bash_exec.get_current_sandbox",
            return_value=sandbox,
        ):
            result = await tool._execute(
                user_id=_USER,
                session=session,
                command=cmd,
                timeout=10,
            )
        assert isinstance(result, ErrorResponse)
        assert "read_tool_result" in result.message
        assert "@@agptfile" in result.message
        # Offending fragment must be the SDK path, not the executable name
        # — the model needs to know which fragment tripped the redirect.
        assert "tool-results/toolu_x.json" in result.message
        assert "Offending fragment: 'cat'" not in result.message
        # Sandbox must not have been invoked at all.
        sandbox.commands.run.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_no_redirect_on_user_path_containing_tool_outputs(self):
        """Regression: a user repo path that happens to contain a
        ``tool-outputs`` directory must NOT trigger the redirect, since
        the user's data isn't an SDK tool-result file."""
        tool = _make_tool()
        session = make_session(user_id=_USER)
        sandbox = _make_sandbox(stdout="ok")
        with (
            patch(
                "backend.copilot.tools.bash_exec.get_current_sandbox",
                return_value=sandbox,
            ),
            patch(
                "backend.copilot.tools.bash_exec.leased_integration_env_vars",
                side_effect=lambda _user_id, _lease_sink=None: _lease_env({}),
            ),
            patch(
                "backend.copilot.tools.bash_exec.get_github_user_git_identity",
                new=AsyncMock(return_value=None),
            ),
        ):
            result = await tool._execute(
                user_id=_USER,
                session=session,
                command="ls my-pipeline/tool-outputs/data.json",
                timeout=10,
            )
        assert isinstance(result, BashExecResponse)
        sandbox.commands.run.assert_called_once()


@pytest.mark.asyncio(loop_scope="session")
async def test_timeout_above_lease_window_is_rejected_before_dispatch():
    tool = _make_tool()
    session = make_session(user_id=_USER)
    sandbox = _make_sandbox(stdout="never")
    with patch(
        "backend.copilot.tools.bash_exec.get_current_sandbox",
        return_value=sandbox,
    ):
        result = await tool._execute(
            user_id=_USER,
            session=session,
            command="sleep 9999",
            timeout=MAX_BASH_TIMEOUT_SECONDS + 1,
        )

    assert isinstance(result, ErrorResponse)
    assert result.error == "invalid_timeout"
    sandbox.commands.run.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_redirect_on_relative_tool_outputs_path(self):
        tool = _make_tool()
        session = make_session(user_id=_USER)
        sandbox = _make_sandbox()
        with patch(
            "backend.copilot.tools.bash_exec.get_current_sandbox",
            return_value=sandbox,
        ):
            result = await tool._execute(
                user_id=_USER,
                session=session,
                command="cat tool-outputs/toolu_x.json | head -50",
                timeout=10,
            )
        assert isinstance(result, ErrorResponse)
        assert "read_tool_result" in result.message
        sandbox.commands.run.assert_not_called()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_normal_command_still_runs(self):
        tool = _make_tool()
        session = make_session(user_id=_USER)
        sandbox = _make_sandbox(stdout="hello")
        with (
            patch(
                "backend.copilot.tools.bash_exec.get_current_sandbox",
                return_value=sandbox,
            ),
            patch(
                "backend.copilot.tools.bash_exec.leased_integration_env_vars",
                side_effect=lambda _user_id, _lease_sink=None: _lease_env({}),
            ),
            patch(
                "backend.copilot.tools.bash_exec.get_github_user_git_identity",
                new=AsyncMock(return_value=None),
            ),
        ):
            result = await tool._execute(
                user_id=_USER,
                session=session,
                command="echo hello",
                timeout=10,
            )
        assert isinstance(result, BashExecResponse)
        sandbox.commands.run.assert_called_once()
