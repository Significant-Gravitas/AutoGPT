"""Tests for BashExecTool — E2B path with token injection."""

import asyncio
import itertools
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from e2b import CommandExitException, CommandResult
from e2b.envd.rpc import handle_rpc_exception
from e2b.exceptions import NotFoundException, TimeoutException
from pydantic import SecretStr

from backend.copilot import integration_creds
from backend.data.model import APIKeyCredentials

from ._test_data import make_session
from .bash_exec import _KILL_TREE_SCRIPT, BashExecTool
from .models import BashExecResponse, ErrorResponse

_USER = "user-bash-exec-test"
_PICKED = {"github": "cred-picked"}


@pytest.fixture(autouse=True)
def picked_credentials():
    """The chat's credential picks, which live in Redis outside these tests."""
    with patch(
        "backend.copilot.tools.bash_exec.selected_credentials",
        new=AsyncMock(return_value=_PICKED),
    ):
        yield


@pytest.fixture(autouse=True)
def proxy_off():
    """No swap proxy unless a test says so: the behaviour before it existed."""
    with patch("backend.copilot.tools.bash_exec.proxy_address", return_value=None):
        yield


class _FakeRedisLock:
    """redis-py's ``Lock``, held in-process: one holder per name at a time."""

    _held: dict[str, asyncio.Lock] = {}

    def __init__(self, name: str, **kwargs):
        self._lock = self._held.setdefault(name, asyncio.Lock())

    async def acquire(self) -> bool:
        await self._lock.acquire()
        return True

    async def release(self) -> None:
        self._lock.release()


@pytest.fixture(autouse=True)
def stream_password_store():
    """Where the screen's stream password expiry lives, Redis in production."""
    _FakeRedisLock._held.clear()
    redis = MagicMock()
    redis.expire = AsyncMock()
    redis.lock = MagicMock(side_effect=_FakeRedisLock)
    with patch(
        "backend.copilot.tools.e2b_sandbox.get_redis_async",
        new=AsyncMock(return_value=redis),
    ):
        yield redis


def _make_tool() -> BashExecTool:
    return BashExecTool()


def _make_sandbox(exit_code: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    """A box whose command prints *stdout*/*stderr* and exits *exit_code*."""

    async def run(*args, on_stdout=None, on_stderr=None, **kwargs):
        if stdout and on_stdout:
            on_stdout(stdout)
        if stderr and on_stderr:
            on_stderr(stderr)
        handle = MagicMock(pid=1)
        if exit_code:
            handle.wait = AsyncMock(
                side_effect=CommandExitException(
                    stdout=stdout, stderr=stderr, exit_code=exit_code, error=None
                )
            )
        else:
            handle.wait = AsyncMock(
                return_value=CommandResult(
                    stdout=stdout, stderr=stderr, exit_code=0, error=None
                )
            )
        return handle

    sandbox = MagicMock()
    sandbox.set_timeout = AsyncMock()
    sandbox.commands.run = AsyncMock(side_effect=run)
    return sandbox


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
                "backend.copilot.tools.bash_exec.get_integration_env_vars",
                new=AsyncMock(return_value=env_vars),
            ) as mock_get_env,
            patch(
                "backend.copilot.tools.bash_exec.get_github_user_git_identity",
                new=AsyncMock(return_value=None),
            ) as mock_identity,
        ):
            result = await tool._execute_on_e2b(
                sandbox=sandbox,
                command="echo hi",
                timeout=10,
                session_id=session.session_id,
                user_id=_USER,
            )

        # The session's picks reach the token lookup, not just its scopes.
        mock_get_env.assert_awaited_once_with(_USER, None, _PICKED, None)
        # And the commit identity comes from that same GitHub account, so a
        # commit made with one account's token is not signed as another's.
        mock_identity.assert_awaited_once_with(_USER, "cred-picked")
        call_kwargs = sandbox.commands.run.call_args[1]
        assert call_kwargs["envs"]["GH_TOKEN"] == "gh-secret"
        assert call_kwargs["envs"]["GITHUB_TOKEN"] == "gh-secret"
        assert isinstance(result, BashExecResponse)

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
                "backend.copilot.tools.bash_exec.get_integration_env_vars",
                new=AsyncMock(return_value={}),
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
                "backend.copilot.tools.bash_exec.get_integration_env_vars",
                new=AsyncMock(return_value={}),
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

        sandbox = _make_sandbox(
            exit_code=1, stdout="not logged in gh-secret", stderr="oops gh-secret"
        )

        with (
            patch(
                "backend.copilot.tools.bash_exec.get_integration_env_vars",
                new=AsyncMock(return_value={"GH_TOKEN": "gh-secret"}),
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
        """When user_id is None, get_integration_env_vars must NOT be called."""
        tool = _make_tool()
        session = make_session(user_id=_USER)
        sandbox = _make_sandbox(stdout="ok")

        with (
            patch(
                "backend.copilot.tools.bash_exec.get_integration_env_vars",
                new=AsyncMock(return_value={"GH_TOKEN": "should-not-appear"}),
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
                "backend.copilot.tools.bash_exec.get_integration_env_vars",
                new=AsyncMock(return_value={}),
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
                "backend.copilot.tools.bash_exec.get_integration_env_vars",
                new=AsyncMock(return_value={}),
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


class TestBashExecBehindTheSwapProxy:
    """With the box's egress through the swap proxy, the command gets
    placeholders: the stored value never reaches the box."""

    _REAL = "ghp_the_real_stored_value"

    @pytest.fixture(autouse=True)
    def proxy_on(self):
        with patch(
            "backend.copilot.tools.bash_exec.proxy_address", return_value="proxy:1080"
        ):
            yield

    @pytest.fixture(autouse=True)
    def stored_github_credential(self):
        manager = MagicMock()
        manager.store.get_creds_by_provider = AsyncMock(
            return_value=[
                APIKeyCredentials(
                    id="cred-picked",
                    provider="github",
                    api_key=SecretStr(self._REAL),
                    title="GitHub",
                )
            ]
        )
        with (
            patch.object(integration_creds, "_manager", manager),
            patch.object(integration_creds, "_ensure_cache_invalidation_listener"),
        ):
            integration_creds._token_cache.clear()
            integration_creds._null_cache.clear()
            integration_creds._credential_id_cache.clear()
            yield
            integration_creds._token_cache.clear()
            integration_creds._null_cache.clear()
            integration_creds._credential_id_cache.clear()

    @pytest.fixture(autouse=True)
    def grants(self):
        with patch(
            "backend.copilot.tools.bash_exec.grant_to_box", new=AsyncMock()
        ) as grant:
            self.grant = grant
            yield grant

    async def _run(self, sandbox: MagicMock) -> BashExecResponse:
        session = make_session(user_id=_USER)
        sandbox.sandbox_id = "sb-1"
        with patch(
            "backend.copilot.tools.bash_exec.get_github_user_git_identity",
            new=AsyncMock(return_value={"GIT_AUTHOR_NAME": "Ada"}),
        ) as identity:
            result = await _make_tool()._execute_on_e2b(
                sandbox=sandbox,
                command="gh repo list",
                timeout=10,
                session_id=session.session_id,
                user_id=_USER,
            )
        # Commits stay attributed to the picked account.
        identity.assert_awaited_once_with(_USER, "cred-picked")
        assert isinstance(result, BashExecResponse)
        return result

    @pytest.mark.asyncio(loop_scope="session")
    async def test_the_command_gets_placeholders_and_no_token(self):
        sandbox = _make_sandbox(stdout="ok")
        with patch(
            "backend.copilot.tools.bash_exec.get_integration_env_vars"
        ) as real_tokens:
            await self._run(sandbox)
        real_tokens.assert_not_called()
        envs = sandbox.commands.run.call_args[1]["envs"]
        assert envs["GH_TOKEN"] == envs["GITHUB_TOKEN"] == "hsurr:github:cred-picked"
        assert envs["GIT_CONFIG_KEY_0"] == "credential.https://github.com.helper"
        assert envs["GIT_AUTHOR_NAME"] == "Ada"
        assert all(self._REAL not in value for value in envs.values())
        # Nothing in the box may drive E2B itself (read its rules, reconnect it).
        assert not any("E2B" in name for name in envs)
        # Granted to this box before the command ran, or it would not resolve.
        self.grant.assert_awaited_once_with("sb-1", {"github": "cred-picked"})

    @pytest.mark.asyncio(loop_scope="session")
    async def test_a_deleted_pick_blanks_the_variables_instead_of_falling_back(
        self,
    ):
        """The box's own environment may hold the default account's
        placeholder; left unset, the command would quietly act as it."""
        sandbox = _make_sandbox(stdout="ok")
        with patch(
            "backend.copilot.tools.bash_exec.selected_credentials",
            new=AsyncMock(return_value={"github": "cred-deleted"}),
        ):
            session = make_session(user_id=_USER)
            sandbox.sandbox_id = "sb-1"
            with patch(
                "backend.copilot.tools.bash_exec.get_github_user_git_identity",
                new=AsyncMock(return_value=None),
            ):
                await _make_tool()._execute_on_e2b(
                    sandbox=sandbox,
                    command="git push",
                    timeout=10,
                    session_id=session.session_id,
                    user_id=_USER,
                )
        envs = sandbox.commands.run.call_args[1]["envs"]
        assert envs["GH_TOKEN"] == envs["GITHUB_TOKEN"] == ""
        assert envs["GIT_CONFIG_COUNT"] == "0"
        self.grant.assert_awaited_once_with("sb-1", {})

    @pytest.mark.asyncio(loop_scope="session")
    async def test_the_grant_lands_before_the_command_starts(self):
        """The proxy resolves a placeholder only once it is granted, so the
        command's first request would be refused if the grant came after."""
        order: list[str] = []
        sandbox = _make_sandbox(stdout="ok")
        run = sandbox.commands.run.side_effect

        async def started(*args, **kwargs):
            order.append("run")
            return await run(*args, **kwargs)

        self.grant.side_effect = lambda *_: order.append("grant")
        sandbox.commands.run = AsyncMock(side_effect=started)
        await self._run(sandbox)
        assert order[:2] == ["grant", "run"]

    @pytest.mark.asyncio(loop_scope="session")
    async def test_a_grant_that_fails_does_not_stop_the_command(self):
        sandbox = _make_sandbox(stdout="ok")
        self.grant.side_effect = ConnectionError("redis down")
        result = await self._run(sandbox)
        assert result.exit_code == 0
        sandbox.commands.run.assert_awaited_once()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_a_placeholder_in_the_output_is_not_redacted(self):
        sandbox = _make_sandbox(stdout="GH_TOKEN=hsurr:github:cred-picked")
        result = await self._run(sandbox)
        assert result.stdout == "GH_TOKEN=hsurr:github:cred-picked"


class TestBashExecWithoutTheSwapProxy:
    @pytest.mark.asyncio(loop_scope="session")
    async def test_the_real_token_is_injected_as_before(self):
        sandbox = _make_sandbox(stdout="ok")
        session = make_session(user_id=_USER)
        with (
            patch(
                "backend.copilot.tools.bash_exec.get_integration_env_vars",
                new=AsyncMock(return_value={"GH_TOKEN": "gh-secret"}),
            ),
            patch("backend.copilot.tools.bash_exec.placeholder_grants") as placeholders,
            patch(
                "backend.copilot.tools.bash_exec.get_github_user_git_identity",
                new=AsyncMock(return_value=None),
            ),
        ):
            await _make_tool()._execute_on_e2b(
                sandbox=sandbox,
                command="gh repo list",
                timeout=10,
                session_id=session.session_id,
                user_id=_USER,
            )
        placeholders.assert_not_called()
        envs = sandbox.commands.run.call_args[1]["envs"]
        assert envs["GH_TOKEN"] == "gh-secret"
        assert "GIT_CONFIG_COUNT" not in envs


class TestBashExecUnderAProviderCeiling:
    """A run whose permissions leave GitHub out gets no GitHub variable,
    with or without the proxy."""

    @pytest.fixture(autouse=True)
    def no_github(self):
        from backend.copilot.permissions import CopilotPermissions

        with patch(
            "backend.copilot.tools.bash_exec.get_current_permissions",
            return_value=CopilotPermissions(
                providers=["github"], providers_exclude=True
            ),
        ):
            yield

    @pytest.mark.asyncio(loop_scope="session")
    @pytest.mark.parametrize("proxy", [None, "proxy:1080"])
    async def test_github_is_left_out(self, proxy):
        sandbox = _make_sandbox(stdout="ok")
        session = make_session(user_id=_USER)
        with (
            patch("backend.copilot.tools.bash_exec.proxy_address", return_value=proxy),
            patch(
                "backend.copilot.tools.bash_exec.get_integration_env_vars",
                new=AsyncMock(return_value={}),
            ) as real,
            patch(
                "backend.copilot.tools.bash_exec.placeholder_grants",
                new=AsyncMock(return_value={}),
            ) as placeholders,
            patch("backend.copilot.tools.bash_exec.grant_to_box", new=AsyncMock()),
            patch(
                "backend.copilot.tools.bash_exec.get_github_user_git_identity",
                new=AsyncMock(return_value={"GIT_AUTHOR_NAME": "Ada"}),
            ) as identity,
        ):
            await _make_tool()._execute_on_e2b(
                sandbox=sandbox,
                command="gh repo list",
                timeout=10,
                session_id=session.session_id,
                user_id=_USER,
            )
        called = placeholders if proxy else real
        assert called.await_args.args[3] == ()
        # Nor is the GitHub account read for the commit identity.
        identity.assert_not_awaited()
        assert "GIT_AUTHOR_NAME" not in sandbox.commands.run.call_args[1]["envs"]


# ---------------------------------------------------------------------------
# Timeouts against a box that behaves like E2B's
# ---------------------------------------------------------------------------

_TICK = 0.02


def _sdk_error(code: Code, message: str) -> Exception:
    """The exception the E2B SDK raises for an envd stream ending with *code*."""
    return handle_rpc_exception(ConnectError(code, message))


class _FakeProcess:
    def __init__(self, output: list[tuple[float, str, str]], runtime: float, code: int):
        self.output = output  # (seconds of run time, "stdout"/"stderr", text)
        self.runtime = runtime
        self.exit_code = code
        self.ran = 0.0
        self.printed = 0
        self.ended = False
        self.killed = False


class _FakeHandle:
    """``AsyncCommandHandle``: streams one process until it ends, the stream's
    own deadline (the command ``timeout``) passes, or the box stops answering."""

    def __init__(self, box: "_FakeBox", pid: int, timeout: float, on_stdout, on_stderr):
        self.pid = pid
        self._box = box
        self._proc = box.processes[pid]
        self._timeout = timeout
        self._on = {"stdout": on_stdout, "stderr": on_stderr}
        self._seen: dict[str, list[str]] = {"stdout": [], "stderr": []}
        self._task = asyncio.create_task(self._stream())

    async def _stream(self) -> CommandResult:
        loop = asyncio.get_running_loop()
        opened = last = loop.time()
        proc = self._proc
        while True:
            await asyncio.sleep(_TICK)
            if self._box.reached_limit():
                raise self._box.limit_error()
            if proc.killed:
                raise _sdk_error(Code.UNKNOWN, "process killed")
            # Runs on the same clock as the deadlines, however late the sleep.
            now = loop.time()
            proc.ran += now - last
            last = now
            while (
                proc.printed < len(proc.output)
                and proc.output[proc.printed][0] <= proc.ran
            ):
                _, stream, text = proc.output[proc.printed]
                proc.printed += 1
                self._seen[stream].append(text)
                if self._on[stream]:
                    self._on[stream](text)
            if proc.ran >= proc.runtime:
                proc.ended = True
                return CommandResult(
                    stdout="".join(self._seen["stdout"]),
                    stderr="".join(self._seen["stderr"]),
                    exit_code=proc.exit_code,
                    error=None,
                )
            if self._timeout and loop.time() - opened >= self._timeout:
                # A connection deadline only: E2B leaves the process running.
                raise _sdk_error(Code.DEADLINE_EXCEEDED, "deadline exceeded")

    async def wait(self) -> CommandResult:
        result = await self._task
        if result.exit_code:
            raise CommandExitException(
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.exit_code,
                error=None,
            )
        return result


class _FakeCommands:
    def __init__(self, box: "_FakeBox"):
        self._box = box
        self._pids = itertools.count(100)

    async def run(
        self,
        cmd,
        background=None,
        envs=None,
        cwd=None,
        on_stdout=None,
        on_stderr=None,
        stdin=None,
        timeout=60,
        request_timeout=None,
    ):
        self._box.answer()
        pid = next(self._pids)
        self._box.processes[pid] = self._box.next_process
        handle = _FakeHandle(self._box, pid, timeout, on_stdout, on_stderr)
        return handle if background else await handle.wait()

    async def connect(
        self, pid, timeout=60, request_timeout=None, on_stdout=None, on_stderr=None
    ):
        self._box.answer()
        proc = self._box.processes.get(pid)
        if proc is None or proc.ended or proc.killed:
            raise NotFoundException(f"process with pid {pid} not found")
        return _FakeHandle(self._box, pid, timeout, on_stdout, on_stderr)

    async def kill(self, pid, request_timeout=None) -> bool:
        self._box.answer()
        proc = self._box.processes.get(pid)
        if proc is None or proc.ended:
            return False
        proc.killed = True
        return True


class _FakeBox:
    """``AsyncSandbox`` with E2B's running-time limit: when it passes, the box
    is paused (``on_timeout="pause"``) with its processes frozen, and any open
    command stream breaks with the SDK's ``TimeoutException``.  Like
    ``auto_resume``, the next request resumes it for another *limit*
    seconds; with *gone_at_limit* it was killed instead and never answers."""

    def __init__(
        self, limit: float, process: _FakeProcess, gone_at_limit: bool = False
    ):
        self._loop = asyncio.get_running_loop()
        self._limit = limit
        self.end_at = self._loop.time() + limit
        self.gone_at_limit = gone_at_limit
        self.paused = False
        self.next_process = process
        self.processes: dict[int, _FakeProcess] = {}
        self.sandbox_id = "box-under-test"
        self.set_timeout = AsyncMock(side_effect=self._set_timeout)
        self.get_info = AsyncMock(side_effect=self._get_info)
        self.commands = _FakeCommands(self)

    async def _set_timeout(self, seconds: int) -> None:
        self.end_at = self._loop.time() + seconds

    async def _get_info(self) -> SimpleNamespace:
        left = self.end_at - self._loop.time()
        return SimpleNamespace(
            end_at=datetime.now(timezone.utc) + timedelta(seconds=left)
        )

    def seconds_left(self) -> float:
        return self.end_at - self._loop.time()

    def reached_limit(self) -> bool:
        if self._loop.time() >= self.end_at:
            self.paused = True
        return self.paused

    def limit_error(self) -> Exception:
        if self.gone_at_limit:
            return _sdk_error(Code.UNAVAILABLE, "sandbox not found")
        return _sdk_error(Code.CANCELED, "stream cancelled")

    def answer(self) -> None:
        if not self.reached_limit():
            return
        if self.gone_at_limit:
            raise self.limit_error()
        self.paused = False
        self.end_at = self._loop.time() + self._limit


async def _run(box: _FakeBox, timeout: int) -> BashExecResponse | ErrorResponse:
    result = await _make_tool()._execute_on_e2b(
        sandbox=box,  # type: ignore[arg-type]
        command="./long-job.sh",
        timeout=timeout,
        session_id="session-under-test",
        user_id=None,
    )
    assert isinstance(result, (BashExecResponse, ErrorResponse))
    return result


def _counting(runtime: float, every: float = 0.1) -> _FakeProcess:
    """A process that prints one numbered line every *every* seconds."""
    steps = int(runtime / every)
    output = [
        (every * (i + 1) - _TICK / 2, "stdout", f"{i + 1}\n") for i in range(steps)
    ]
    return _FakeProcess(output, runtime, 0)


class TestBashExecE2BTimeouts:
    """The SDK raises ``TimeoutException`` for the command's own deadline and
    also for the box pausing or dying under the command, at whatever moment
    that happens.  Only the first may be reported as the command timing out."""

    @pytest.fixture(autouse=True)
    def kill_tree(self):
        """The script that kills a timed-out command's children, run on the box."""
        with patch(
            "backend.copilot.tools.bash_exec.run_internal",
            new=AsyncMock(return_value=SimpleNamespace(exit_code=0)),
        ) as run:
            yield run

    @pytest.mark.asyncio(loop_scope="session")
    async def test_command_outlives_the_box_running_time_limit(self):
        # The turn's connect armed the limit long ago: 0.3s of it is left,
        # and the command needs 0.8s of its 5s timeout.
        box = _FakeBox(limit=0.3, process=_counting(0.8))

        result = await _run(box, timeout=5)

        assert isinstance(result, BashExecResponse)
        assert result.timed_out is False
        assert result.exit_code == 0
        assert result.stdout == "".join(f"{i}\n" for i in range(1, 9))
        assert "Timed out" not in result.stderr
        # Re-armed to the configured safety net (420s), which covers the
        # whole 5s timeout.
        box.set_timeout.assert_awaited_once_with(420)

    @pytest.mark.asyncio(loop_scope="session")
    async def test_long_timeout_extends_the_limit_past_it(self):
        box = _FakeBox(limit=30, process=_counting(0.1))

        await _run(box, timeout=3600)

        box.set_timeout.assert_awaited_once_with(3660)

    @pytest.mark.asyncio(loop_scope="session")
    async def test_timeout_past_the_hung_tool_cap_is_capped_below_it(self):
        # The turn gives up on a pending tool after two hours and pauses the
        # box, so a longer timeout would only bill a box nobody is following.
        # The cap sits below the watchdog so the tool's own timeout lands first.
        from backend.copilot.constants import HUNG_TOOL_CAP_SECONDS

        from .bash_exec import _HUNG_CAP_MARGIN_SECONDS

        box = _FakeBox(limit=30, process=_counting(0.1))

        await _run(box, timeout=7 * 24 * 60 * 60)

        box.set_timeout.assert_awaited_once_with(
            HUNG_TOOL_CAP_SECONDS - _HUNG_CAP_MARGIN_SECONDS + 60
        )

    @pytest.mark.asyncio(loop_scope="session")
    async def test_a_short_command_leaves_a_longer_limit_alone(self):
        # Another command on the box (a parallel call, or another session of
        # the same expert) already pushed the limit out to an hour.
        box = _FakeBox(limit=3660, process=_counting(0.1))

        result = await _run(box, timeout=30)

        assert isinstance(result, BashExecResponse)
        box.set_timeout.assert_not_awaited()
        assert box.seconds_left() > 3600

    @pytest.mark.asyncio(loop_scope="session")
    async def test_commands_started_together_cannot_shorten_the_limit(self):
        from .e2b_sandbox import keep_sandbox_running

        box = _FakeBox(limit=30, process=_counting(0.1))
        set_limit = box._set_timeout

        async def slow_read():
            await asyncio.sleep(0.01)
            return await box._get_info()

        async def short_lands_last(seconds: int) -> None:
            # Both read the old limit; the short call's set reaches E2B last.
            await asyncio.sleep(0.05 if seconds < 1000 else 0)
            await set_limit(seconds)

        box.get_info = AsyncMock(side_effect=slow_read)
        box.set_timeout = AsyncMock(side_effect=short_lands_last)

        assert await asyncio.gather(
            keep_sandbox_running(box, 3660),  # type: ignore[arg-type]
            keep_sandbox_running(box, 420),  # type: ignore[arg-type]
        ) == [True, True]

        assert box.seconds_left() > 3600

    @pytest.mark.asyncio(loop_scope="session")
    async def test_an_unreadable_limit_is_still_extended(self):
        from .e2b_sandbox import keep_sandbox_running

        box = _FakeBox(limit=30, process=_counting(0.1))
        box.get_info = AsyncMock(side_effect=RuntimeError("E2B API down"))

        assert await keep_sandbox_running(box, 900) is True  # type: ignore[arg-type]

        box.set_timeout.assert_awaited_once_with(900)

    @pytest.mark.asyncio(loop_scope="session")
    async def test_box_paused_mid_command_is_reattached(self):
        # E2B refuses the new limit, so the box still pauses under the
        # command; the command is frozen, not gone.
        box = _FakeBox(limit=0.3, process=_counting(0.8))
        box.set_timeout = AsyncMock(side_effect=RuntimeError("E2B API down"))

        result = await _run(box, timeout=5)

        assert isinstance(result, BashExecResponse)
        assert result.timed_out is False
        assert result.exit_code == 0
        assert result.stdout == "".join(f"{i}\n" for i in range(1, 9))
        assert "and was resumed" in result.message

    @pytest.mark.asyncio(loop_scope="session")
    async def test_box_gone_mid_command_is_not_reported_as_a_timeout(self):
        box = _FakeBox(limit=0.3, process=_counting(0.8), gone_at_limit=True)
        box.set_timeout = AsyncMock(side_effect=RuntimeError("E2B API down"))

        result = await _run(box, timeout=120)

        assert isinstance(result, BashExecResponse)
        assert result.timed_out is False
        assert result.exit_code == -1
        assert "Timed out after 120s" not in result.stderr
        assert "Timed out after 120s" not in result.message
        # What it printed before the box went away is kept.
        assert result.stdout.startswith("1\n2\n")
        assert "8\n" not in result.stdout
        assert re.match(
            r"Lost the connection to the sandbox after 0\.\d+s \(timeout 120s\)",
            result.stderr,
        )
        assert "may still be running" in result.message

    @pytest.mark.asyncio(loop_scope="session")
    async def test_box_pausing_past_the_reconnect_cap_gives_up_following(self):
        # Every reattach works, but the box pauses again 0.2s later, so the
        # 2s command is still running when the reattaches run out.
        from .bash_exec import _MAX_RECONNECTS

        proc = _counting(2.0)
        box = _FakeBox(limit=0.2, process=proc)
        box.set_timeout = AsyncMock(side_effect=RuntimeError("E2B API down"))
        box.commands.connect = AsyncMock(wraps=box.commands.connect)

        result = await _run(box, timeout=30)

        assert isinstance(result, BashExecResponse)
        assert box.commands.connect.await_count == _MAX_RECONNECTS
        assert result.timed_out is False
        assert result.exit_code == -1
        assert result.stdout.startswith("1\n2\n")
        assert "Timed out" not in result.stderr
        assert "may still be running" in result.message
        assert proc.killed is False

    @pytest.mark.asyncio(loop_scope="session")
    async def test_box_pausing_in_the_last_second_is_not_the_timeout(self):
        # The box pauses 1.2s into a 2s timeout; the command needs 1.6s.
        box = _FakeBox(limit=1.2, process=_counting(1.6))
        box.set_timeout = AsyncMock(side_effect=RuntimeError("E2B API down"))

        result = await _run(box, timeout=2)

        assert isinstance(result, BashExecResponse)
        assert result.timed_out is False
        assert result.exit_code == 0
        assert result.stdout == "".join(f"{i}\n" for i in range(1, 17))

    @pytest.mark.asyncio(loop_scope="session")
    async def test_reattach_failing_past_the_deadline_is_the_timeout(self):
        # The box pauses just before the deadline and the reattach only
        # fails once the deadline has passed.
        proc = _counting(60)
        box = _FakeBox(limit=1.8, process=proc)
        box.set_timeout = AsyncMock(side_effect=RuntimeError("E2B API down"))

        async def slow_failing_connect(*args, **kwargs):
            await asyncio.sleep(0.4)
            raise TimeoutException("deadline exceeded while reconnecting")

        box.commands.connect = AsyncMock(side_effect=slow_failing_connect)

        result = await _run(box, timeout=2)

        assert isinstance(result, BashExecResponse)
        assert result.timed_out is True
        assert (
            result.message
            == "Timed out after 2s; it was killed, with every process it started."
        )
        assert proc.killed is True

    @pytest.mark.asyncio(loop_scope="session")
    async def test_real_timeout_keeps_partial_output_and_kills_the_command(self):
        proc = _FakeProcess(
            [(0.1, "stdout", "partial\n"), (0.2, "stderr", "warning: slow\n")],
            runtime=60,
            code=0,
        )
        box = _FakeBox(limit=30, process=proc)

        result = await _run(box, timeout=1)

        assert isinstance(result, BashExecResponse)
        assert result.timed_out is True
        assert result.exit_code == -1
        assert result.stdout == "partial\n"
        assert result.stderr == "Timed out after 1s\nwarning: slow"
        assert (
            result.message
            == "Timed out after 1s; it was killed, with every process it started."
        )
        # E2B's deadline only closes the stream; the tool kills the command.
        assert proc.killed is True

    @pytest.mark.asyncio(loop_scope="session")
    async def test_timeout_kills_the_children_before_the_shell(self, kill_tree):
        # envd's kill reaches only the shell, so its children are killed first,
        # while the shell is still there to find them under.
        proc = _FakeProcess([], runtime=60, code=0)
        box = _FakeBox(limit=30, process=proc)

        def children_first(sandbox, script, **kwargs):
            assert proc.killed is False
            return SimpleNamespace(exit_code=0)

        kill_tree.side_effect = children_first

        await _run(box, timeout=1)

        kill_tree.assert_awaited_once()
        script = kill_tree.await_args.args[1]
        assert "p=100\n" in script
        assert proc.killed is True

    @pytest.mark.asyncio(loop_scope="session")
    async def test_children_that_could_not_be_killed_are_reported(self, kill_tree):
        proc = _FakeProcess([], runtime=60, code=0)
        box = _FakeBox(limit=30, process=proc)
        kill_tree.side_effect = RuntimeError("login files changed")

        result = await _run(box, timeout=1)

        assert isinstance(result, BashExecResponse)
        assert result.timed_out is True
        assert result.message == (
            "Timed out after 1s; the shell was killed, but processes it "
            "started may still be running."
        )
        assert proc.killed is True

    @pytest.mark.asyncio(loop_scope="session")
    async def test_command_ending_while_detached_is_reported_unseen(self):
        # Paused under the command, and the box's own process table no
        # longer has it when we come back.
        box = _FakeBox(limit=0.3, process=_counting(0.8))
        box.set_timeout = AsyncMock(side_effect=RuntimeError("E2B API down"))
        box.commands.connect = AsyncMock(
            side_effect=NotFoundException("process with pid 100 not found")
        )

        result = await _run(box, timeout=120)

        assert isinstance(result, BashExecResponse)
        assert result.timed_out is False
        assert result.exit_code == -1
        assert "exit code unknown" in result.stderr
        assert "Timed out" not in result.stderr

    @pytest.mark.asyncio(loop_scope="session")
    async def test_stream_not_opening_is_not_the_command_timeout(self):
        box = _FakeBox(limit=30, process=_counting(0.1))
        box.commands.run = AsyncMock(
            side_effect=TimeoutException(
                "Request timed out: the stream didn't open within "
                "'request_timeout' (60 seconds)."
            )
        )

        result = await _run(box, timeout=120)

        assert isinstance(result, ErrorResponse)
        assert result.error == "e2b_start_timeout"
        assert "Timed out after 120s" not in result.message
        assert "may or may not have run" in result.message

    @pytest.mark.asyncio(loop_scope="session")
    async def test_limit_extension_pushes_out_the_stream_password(
        self, stream_password_store
    ):
        from .e2b_sandbox import SandboxOwner, keep_sandbox_running

        box = _FakeBox(limit=30, process=_counting(0.1))
        owner = SandboxOwner(kind="session", id="session-under-test")

        assert await keep_sandbox_running(box, 900, owner) is True  # type: ignore[arg-type]

        box.set_timeout.assert_awaited_once_with(900)
        stream_password_store.expire.assert_awaited_once_with(
            owner.stream_key(), 900, gt=True
        )


def _children(pid: int) -> list[int]:
    found = []
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = stat.read_text().rsplit(") ", 1)[1].split()
        except (OSError, IndexError):
            continue
        if int(fields[1]) == pid:
            found.append(int(stat.parent.name))
    return found


def _alive(pid: int) -> bool:
    """Running or stopped; a killed child its stopped parent hasn't reaped is not."""
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[0]
    except (OSError, IndexError):
        return False
    return state not in ("Z", "X")


@pytest.mark.skipif(sys.platform != "linux", reason="reads /proc")
def test_kill_tree_script_kills_a_compound_commands_children():
    # The reviewer's repro: killing the shell of `sleep 30 && echo done` left
    # sleep running under a new parent.
    _kill_tree_of_sleep_then_echo()


@pytest.mark.skipif(sys.platform != "linux", reason="reads /proc")
def test_kill_tree_script_holds_while_other_processes_exit():
    # A process exiting between the /proc glob and awk reading its stat file
    # aborted awk, so the listing came back empty and the child was left
    # running, or stopped and orphaned once the shell was killed.
    churn = subprocess.Popen(["bash", "-c", "while :; do /bin/true; done"])
    try:
        for _ in range(10):
            _kill_tree_of_sleep_then_echo()
    finally:
        churn.kill()
        churn.wait(timeout=10)


@pytest.mark.skipif(sys.platform != "linux", reason="reads /proc")
def test_kill_tree_script_kills_children_forked_while_it_stops_them():
    # A child forked between a listing and its STOP outlived a single pass; the
    # outer shell's `; :` keeps the forking loop a child rather than p itself.
    shell = subprocess.Popen(
        ["bash", "-c", 'bash -c "while :; do sleep 30 & sleep 0.005; done"; :'],
        start_new_session=True,
    )
    try:
        time.sleep(0.3)
        subprocess.run(
            ["bash", "-c", _KILL_TREE_SCRIPT.replace("__PID__", str(shell.pid))],
            check=True,
            timeout=10,
        )
        assert [p for p in _session(shell.pid) if p != shell.pid and _alive(p)] == []
    finally:
        os.killpg(shell.pid, signal.SIGKILL)
        shell.wait(timeout=10)


def _session(sid: int) -> list[int]:
    found = []
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = stat.read_text().rsplit(") ", 1)[1].split()
        except (OSError, IndexError):
            continue
        if int(fields[3]) == sid:
            found.append(int(stat.parent.name))
    return found


def _kill_tree_of_sleep_then_echo() -> None:
    shell = subprocess.Popen(["bash", "-c", "sleep 30 && echo done"])
    try:
        for _ in range(50):
            if _children(shell.pid):
                break
            time.sleep(0.05)
        (sleeper,) = _children(shell.pid)

        subprocess.run(
            ["bash", "-c", _KILL_TREE_SCRIPT.replace("__PID__", str(shell.pid))],
            check=True,
            timeout=10,
        )

        assert not _alive(sleeper)
        # The shell is stopped, not killed: envd's kill does that, and the
        # shell must not start `echo done` in between.
        assert _alive(shell.pid)
    finally:
        shell.kill()
        shell.wait(timeout=10)
