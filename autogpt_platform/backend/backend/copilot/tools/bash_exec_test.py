"""Tests for BashExecTool — E2B path with token injection."""

import asyncio
import itertools
import re
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from e2b import CommandExitException, CommandResult
from e2b.envd.rpc import handle_rpc_exception
from e2b.exceptions import NotFoundException, TimeoutException

from ._test_data import make_session
from .bash_exec import BashExecTool
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
        mock_get_env.assert_awaited_once_with(_USER, None, _PICKED)
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
        opened = loop.time()
        proc = self._proc
        while True:
            await asyncio.sleep(_TICK)
            if self._box.reached_limit():
                raise self._box.limit_error()
            if proc.killed:
                raise _sdk_error(Code.UNKNOWN, "process killed")
            proc.ran += _TICK
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
        assert result.message == "Timed out after 1s; it was killed."
        # E2B's deadline only closes the stream; the tool kills the command.
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
