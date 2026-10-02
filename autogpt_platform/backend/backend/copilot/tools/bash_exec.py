"""Bash execution tool — run shell commands on E2B or in a bubblewrap sandbox.

When an E2B sandbox is available in the current execution context the command
runs directly on the remote E2B cloud environment.  This means:

- **Persistent filesystem**: files survive across turns via HTTP-based sync
  with the sandbox's ``/home/user`` directory (E2B files API), shared with
  SDK Read/Write/Edit tools.
- **Full internet access**: E2B sandboxes have unrestricted outbound network.
- **Execution isolation**: E2B provides a fresh, containerised Linux environment.

When E2B is *not* configured the tool falls back to **bubblewrap** (bwrap):
OS-level isolation with a whitelist-only filesystem, no network, and resource
limits.  Requires bubblewrap to be installed (Linux only).
"""

import asyncio
import logging
import shlex
from typing import Any

from e2b import AsyncSandbox, CommandExitException
from e2b.exceptions import NotFoundException, TimeoutException
from pydantic import BaseModel

from backend.copilot.constants import HUNG_TOOL_CAP_SECONDS
from backend.copilot.context import (
    E2B_WORKDIR,
    get_current_sandbox,
    looks_like_sdk_tool_result_path,
    sdk_tool_result_redirect_hint,
)
from backend.copilot.credential_selection import selected_credentials
from backend.copilot.integration_creds import (
    get_github_user_git_identity,
    get_integration_env_vars,
)
from backend.copilot.model import ChatSession
from backend.copilot.sdk.env import config as chat_config
from backend.util.e2b_network import reattach_command

from .base import BaseTool
from .connect_integration import requested_scopes
from .e2b_sandbox import SandboxOwner, keep_sandbox_running
from .models import BashExecResponse, ErrorResponse, ToolResponseBase
from .sandbox import get_workspace_dir, has_full_sandbox, run_sandboxed

logger = logging.getLogger(__name__)

# Headroom on the box's running-time limit past the command's own timeout, so
# the box is still up to report the result (or the timeout) and to be killed.
_SANDBOX_LIMIT_MARGIN_SECONDS = 60
# A stream that breaks this close to the command's deadline is its timeout.
# The SDK's own deadline starts after ours and is rounded to the millisecond,
# so it never fires earlier than this; any earlier break is the box's doing.
_DEADLINE_SLACK_SECONDS = 0.05
# Reattaches to a running command after its stream to the box drops.
_MAX_RECONNECTS = 3
_KILL_TIMEOUT_SECONDS = 10


class _E2BRun(BaseModel):
    """How a command on the box ended, with all the output we received."""

    stdout: str
    stderr: str
    elapsed: float
    exit_code: int | None = None
    timed_out: bool = False
    # Kill sent after a timeout: True done, False not found, None unknown.
    killed: bool | None = None
    # Why we stopped following a command that had not ended: the stream
    # error's type name.
    lost: str | None = None
    # The command ended while we were not attached: no exit code.
    ended_unseen: bool = False
    reconnects: int = 0


def _build_completion_response(
    stdout: str | None,
    stderr: str | None,
    exit_code: int,
    secret_values: list[str],
    session_id: str | None,
) -> BashExecResponse:
    out = stdout or ""
    err = stderr or ""
    for secret in secret_values:
        out = out.replace(secret, "[REDACTED]")
        err = err.replace(secret, "[REDACTED]")
    return BashExecResponse(
        message=f"Command executed with status code {exit_code}",
        stdout=out,
        stderr=err,
        exit_code=exit_code,
        timed_out=False,
        session_id=session_id,
    )


class BashExecTool(BaseTool):
    """Execute Bash commands on E2B or in a bubblewrap sandbox."""

    @property
    def name(self) -> str:
        return "bash_exec"

    @property
    def description(self) -> str:
        return (
            "Execute a Bash command or script. Shares filesystem with SDK file tools. "
            "Useful for scripts, data processing, and package installation. "
            "Killed after `timeout` seconds. Anything that should persist belongs "
            "in ~/workspace (a durable volume); other paths are scratch. The "
            "desktop (start_desktop) is this same machine, all paths included. "
            "Expert sessions: ~/workspace is the expert's own machine, ~/shared "
            "is the user's workspace."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Bash command or script.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds; raise for long-running commands.",
                    "default": 120,
                },
            },
            "required": ["command"],
        }

    @property
    def requires_auth(self) -> bool:
        # True because _execute_on_e2b injects user tokens (GH_TOKEN etc.)
        # when user_id is present.  Defense-in-depth: ensures only authenticated
        # users reach the token injection path.
        return True

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        command: str = "",
        timeout: int = 120,
        **kwargs: Any,
    ) -> ToolResponseBase:
        """Run a bash command on E2B (if available) or in a bubblewrap sandbox.

        Dispatches to :meth:`_execute_on_e2b` when a sandbox is present in the
        current execution context, otherwise falls back to the local bubblewrap
        sandbox.  Returns a :class:`BashExecResponse` on success or an
        :class:`ErrorResponse` when the sandbox is unavailable or the command
        is empty.
        """
        session_id = session.session_id if session else None

        command = command.strip()
        timeout = int(timeout)

        if not command:
            return ErrorResponse(
                message="No command provided.",
                error="empty_command",
                session_id=session_id,
            )

        # Pre-flight redirect: bash sandbox can't reach host-side SDK
        # tool-result paths. Without this the model burns turns retrying
        # `cat /root/.claude/projects/...` after `Permission denied`.
        if looks_like_sdk_tool_result_path(command):
            return ErrorResponse(
                message=sdk_tool_result_redirect_hint(command),
                error="sdk_tool_result_path_in_bash_command",
                session_id=session_id,
            )

        sandbox = get_current_sandbox()
        if sandbox is not None:
            return await self._execute_on_e2b(
                sandbox,
                command,
                timeout,
                session_id,
                user_id,
                required_scopes=requested_scopes(session),
                owner=(
                    SandboxOwner.for_session(session.session_id, session.expert_id)
                    if session
                    else None
                ),
            )

        # Bubblewrap fallback: local isolated execution.
        if not has_full_sandbox():
            return ErrorResponse(
                message="bash_exec requires bubblewrap sandbox (Linux only).",
                error="sandbox_unavailable",
                session_id=session_id,
            )

        workspace = get_workspace_dir(session_id or "default")

        stdout, stderr, exit_code, timed_out = await run_sandboxed(
            command=["bash", "-c", command],
            cwd=workspace,
            timeout=timeout,
        )

        return BashExecResponse(
            message=(
                "Execution timed out"
                if timed_out
                else f"Command executed with status code {exit_code}"
            ),
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            timed_out=timed_out,
            session_id=session_id,
        )

    async def _execute_on_e2b(
        self,
        sandbox: AsyncSandbox,
        command: str,
        timeout: int,
        session_id: str | None,
        user_id: str | None = None,
        required_scopes: dict[str, frozenset[str]] | None = None,
        owner: SandboxOwner | None = None,
    ) -> ToolResponseBase:
        """Execute *command* on the E2B sandbox via commands.run().

        Integration tokens (e.g. GH_TOKEN) are injected into the sandbox env
        for any user with connected accounts. E2B has full internet access, so
        CLI tools like ``gh`` work without manual authentication.
        """
        # The turn stops following a tool after the hung-tool cap and pauses
        # the box, so a longer timeout only keeps an unwatched box billing.
        timeout = min(max(timeout, 1), HUNG_TOOL_CAP_SECONDS)
        envs: dict[str, str] = {
            "PATH": "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        }
        # Collect injected secret values so we can scrub them from output.
        secret_values: list[str] = []
        if user_id is not None:
            selected = await selected_credentials(session_id)
            integration_env = await get_integration_env_vars(
                user_id, required_scopes, selected
            )
            secret_values = [v for v in integration_env.values() if v]
            envs.update(integration_env)

            # Set git author/committer identity from the user's GitHub profile
            # so commits made in the sandbox are attributed correctly.
            git_identity = await get_github_user_git_identity(
                user_id, selected.get("github")
            )
            if git_identity:
                envs.update(git_identity)

        # The box's running-time limit was armed when this turn connected;
        # E2B pauses the box at it, breaking the command's stream mid-run.
        await keep_sandbox_running(
            sandbox,
            max(
                chat_config.e2b_sandbox_timeout,
                timeout + _SANDBOX_LIMIT_MARGIN_SECONDS,
            ),
            owner,
        )
        try:
            run = await _follow_command(sandbox, command, timeout, envs)
        except TimeoutException as exc:
            return ErrorResponse(
                message=(
                    "The sandbox did not confirm the command started within "
                    "the request timeout, so it may or may not have run "
                    f"({exc}). This is not the command's {timeout}s timeout."
                ),
                error="e2b_start_timeout",
                session_id=session_id,
            )
        except Exception as exc:
            logger.error("[E2B] bash_exec failed: %s", exc, exc_info=True)
            return ErrorResponse(
                message=f"E2B execution failed: {exc}",
                error="e2b_execution_error",
                session_id=session_id,
            )
        return _run_response(run, timeout, secret_values, session_id)


async def _follow_command(
    sandbox: AsyncSandbox, command: str, timeout: int, envs: dict[str, str]
) -> _E2BRun:
    """Run *command* on the box and follow it to its end or its *timeout*.

    The SDK raises ``TimeoutException`` for more than the command's own
    deadline: the box pausing or dying, or a proxy cancelling the stream,
    surface the same way at any elapsed time.  Only a break at the deadline
    is the timeout (the command is then killed); before it, the command is
    still running on the box, so the stream is reattached instead of the
    command reported as failed.  Output printed while detached is not
    replayed by E2B.  Raises only when the command could not be started.
    """
    loop = asyncio.get_running_loop()
    stdout: list[str] = []
    stderr: list[str] = []
    started = loop.time()
    deadline = started + timeout
    handle = await sandbox.commands.run(
        f"bash -c {shlex.quote(command)}",
        background=True,
        cwd=E2B_WORKDIR,
        timeout=timeout,
        envs=envs,
        on_stdout=stdout.append,
        on_stderr=stderr.append,
    )
    pid = handle.pid
    reconnects = 0

    def outcome(**kwargs: Any) -> _E2BRun:
        return _E2BRun(
            stdout="".join(stdout),
            stderr="".join(stderr),
            elapsed=loop.time() - started,
            reconnects=reconnects,
            **kwargs,
        )

    while True:
        try:
            result = await handle.wait()
            return outcome(exit_code=result.exit_code)
        except CommandExitException as exc:
            return outcome(exit_code=exc.exit_code)
        except Exception as exc:
            remaining = deadline - loop.time()
            if remaining <= _DEADLINE_SLACK_SECONDS:
                return outcome(timed_out=True, killed=await _kill(sandbox, pid))
            logger.warning(
                "[E2B] bash_exec lost its stream to pid %s after %.1fs of %ds "
                "(%s: %s)",
                pid,
                loop.time() - started,
                timeout,
                type(exc).__name__,
                exc,
            )
            if reconnects >= _MAX_RECONNECTS:
                return outcome(lost=type(exc).__name__)
            reconnects += 1
            try:
                handle = await reattach_command(
                    sandbox,
                    pid,
                    timeout=remaining,
                    on_stdout=stdout.append,
                    on_stderr=stderr.append,
                )
            except NotFoundException:
                return outcome(ended_unseen=True)
            except Exception as reconnect_exc:
                if deadline - loop.time() <= _DEADLINE_SLACK_SECONDS:
                    return outcome(timed_out=True, killed=await _kill(sandbox, pid))
                logger.warning(
                    "[E2B] bash_exec could not reattach to pid %s: %s",
                    pid,
                    reconnect_exc,
                )
                return outcome(lost=type(exc).__name__)


async def _kill(sandbox: AsyncSandbox, pid: int) -> bool | None:
    try:
        return await asyncio.wait_for(
            sandbox.commands.kill(pid), timeout=_KILL_TIMEOUT_SECONDS
        )
    except Exception as exc:
        logger.warning("[E2B] Could not kill timed-out pid %s: %s", pid, exc)
        return None


def _run_response(
    run: _E2BRun, timeout: int, secret_values: list[str], session_id: str | None
) -> BashExecResponse:
    if run.exit_code is not None:
        response = _build_completion_response(
            run.stdout, run.stderr, run.exit_code, secret_values, session_id
        )
        if run.reconnects:
            response.message += (
                f" (the connection to the sandbox dropped {run.reconnects} "
                "time(s) and was resumed; output printed meanwhile may be "
                "missing)"
            )
        return response

    elapsed = f"{run.elapsed:.1f}s"
    if run.timed_out:
        if run.killed is True:
            fate = "it was killed."
        elif run.killed is False:
            fate = "it had already ended."
        else:
            fate = "it could not be killed and may still be running."
        note = f"Timed out after {timeout}s"
        message = f"{note}; {fate}"
    elif run.ended_unseen:
        note = (
            f"Lost the connection to the sandbox after {elapsed} (timeout "
            f"{timeout}s); the command ended meanwhile, exit code unknown"
        )
        message = (
            f"{note}. Any output it printed after the drop is lost; it was "
            "not killed and did not time out."
        )
    else:
        note = (
            f"Lost the connection to the sandbox after {elapsed} (timeout "
            f"{timeout}s); the command may still be running"
        )
        message = (
            f"{note}. It did not time out and was not killed (the stream "
            f"to it broke: {run.lost}). "
            "Check whether it is still running (e.g. with ps) before running "
            "it again."
        )
    response = _build_completion_response(
        run.stdout, run.stderr, -1, secret_values, session_id
    )
    # First line of stderr, so it is what the chat shows for this call.
    response.stderr = f"{note}\n{response.stderr}".rstrip("\n")
    response.message = message
    response.timed_out = run.timed_out
    return response
