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

import logging
from typing import TYPE_CHECKING, Any

from backend.copilot.model import ChatSession

from .base import BaseTool
from .models import BashExecResponse, ErrorResponse, ToolResponseBase
from .sandbox import get_workspace_dir, has_full_sandbox, run_sandboxed

if TYPE_CHECKING:
    from e2b import AsyncSandbox

logger = logging.getLogger(__name__)

# Working directory inside E2B sandboxes (must match _E2B_WORKDIR in e2b_sandbox.py).
_E2B_WORKDIR = "/home/user"


class BashExecTool(BaseTool):
    """Execute Bash commands on E2B or in a bubblewrap sandbox."""

    @property
    def name(self) -> str:
        return "bash_exec"

    @property
    def description(self) -> str:
        return (
            "Execute a Bash command or script. "
            "Full Bash scripting is supported (loops, conditionals, pipes, "
            "functions, etc.). "
            "The working directory is shared with the SDK Read/Write/Edit/Glob/Grep "
            "tools — files created by either are immediately visible to both. "
            "Execution is killed after the timeout (default 30s, max 120s). "
            "Returns stdout and stderr. "
            "Useful for file manipulation, data processing, running scripts, "
            "and installing packages."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Bash command or script to execute.",
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        "Max execution time in seconds (default 30, max 120)."
                    ),
                    "default": 30,
                },
            },
            "required": ["command"],
        }

    @property
    def requires_auth(self) -> bool:
        return False

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs: Any,
    ) -> ToolResponseBase:
        session_id = session.session_id if session else None

        command: str = (kwargs.get("command") or "").strip()
        timeout: int = int(kwargs.get("timeout", 30))

        if not command:
            return ErrorResponse(
                message="No command provided.",
                error="empty_command",
                session_id=session_id,
            )

        # E2B path: run on remote cloud sandbox when available.
        from backend.copilot.sdk.tool_adapter import get_current_sandbox

        sandbox = get_current_sandbox()
        if sandbox is not None:
            return await self._execute_on_e2b(sandbox, command, timeout, session_id)

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
                else f"Command executed (exit {exit_code})"
            ),
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            timed_out=timed_out,
            session_id=session_id,
        )

    async def _execute_on_e2b(
        self,
        sandbox: "AsyncSandbox",
        command: str,
        timeout: int,
        session_id: str | None,
    ) -> ToolResponseBase:
        """Execute *command* on the E2B sandbox via commands.run()."""
        try:
            result = await sandbox.commands.run(
                f"bash -c {_shell_quote(command)}",
                cwd=_E2B_WORKDIR,
                timeout=timeout,
                envs={"PATH": "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"},
            )
            return BashExecResponse(
                message=f"Command executed on E2B (exit {result.exit_code})",
                stdout=result.stdout or "",
                stderr=result.stderr or "",
                exit_code=result.exit_code,
                timed_out=False,
                session_id=session_id,
            )
        except Exception as exc:
            # Distinguish timeout from other errors using E2B's typed exception
            from e2b.exceptions import TimeoutException

            if isinstance(exc, TimeoutException):
                return BashExecResponse(
                    message="Execution timed out",
                    stdout="",
                    stderr=f"Timed out after {timeout}s",
                    exit_code=-1,
                    timed_out=True,
                    session_id=session_id,
                )
            logger.error("[E2B] bash_exec failed: %s", exc, exc_info=True)
            return ErrorResponse(
                message=f"E2B execution failed: {exc}",
                error="e2b_execution_error",
                session_id=session_id,
            )


def _shell_quote(s: str) -> str:
    """Single-quote a string for safe shell embedding."""
    return "'" + s.replace("'", "'\\''") + "'"
