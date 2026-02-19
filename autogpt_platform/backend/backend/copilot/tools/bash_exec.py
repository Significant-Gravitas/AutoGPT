"""Bash execution tool — run shell commands in a sandbox.

Supports two backends:
- **e2b** (preferred): VM-level isolation with network access, enabled via
  the COPILOT_E2B feature flag.
- **bubblewrap** (fallback): kernel-level isolation, no network, Linux-only.

Full Bash scripting is allowed (loops, conditionals, pipes, functions, etc.).
"""

import logging
import shlex
from typing import Any

from backend.copilot.model import ChatSession

from .base import BaseTool
from .models import BashExecResponse, ErrorResponse, ToolResponseBase
from .sandbox import get_workspace_dir, has_full_sandbox, run_sandboxed

logger = logging.getLogger(__name__)

_SANDBOX_HOME = "/home/user"


class BashExecTool(BaseTool):
    """Execute Bash commands in a bubblewrap sandbox."""

    @property
    def name(self) -> str:
        return "bash_exec"

    @property
    def description(self) -> str:
        if _is_e2b_available():
            return (
                "Execute a Bash command or script in an e2b sandbox (microVM). "
                "Full Bash scripting is supported (loops, conditionals, pipes, "
                "functions, etc.). "
                "The sandbox shares the same filesystem as the read_file/write_file "
                "tools — files created by any tool are accessible to all others. "
                "Network access IS available (pip install, curl, etc.). "
                "Working directory is /home/user/. "
                "Execution is killed after the timeout (default 30s, max 120s). "
                "Returns stdout and stderr."
            )
        if not has_full_sandbox():
            return (
                "Bash execution is DISABLED — bubblewrap sandbox is not "
                "available on this platform. Do not call this tool."
            )
        return (
            "Execute a Bash command or script in a bubblewrap sandbox. "
            "Full Bash scripting is supported (loops, conditionals, pipes, "
            "functions, etc.). "
            "The sandbox shares the same working directory as the SDK Read/Write "
            "tools — files created by either are accessible to both. "
            "SECURITY: Only system directories (/usr, /bin, /lib, /etc) are "
            "visible read-only, the per-session workspace is the only writable "
            "path, environment variables are wiped (no secrets), all network "
            "access is blocked at the kernel level, and resource limits are "
            "enforced (max 64 processes, 512MB memory, 50MB file size). "
            "Application code, configs, and other directories are NOT accessible. "
            "To fetch web content, use the web_fetch tool instead. "
            "Execution is killed after the timeout (default 30s, max 120s). "
            "Returns stdout and stderr. "
            "Useful for file manipulation, data processing with Unix tools "
            "(grep, awk, sed, jq, etc.), and running shell scripts."
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
        timeout: int = kwargs.get("timeout", 30)

        if not command:
            return ErrorResponse(
                message="No command provided.",
                error="empty_command",
                session_id=session_id,
            )

        # --- E2B path ---
        if _is_e2b_available():
            return await self._execute_e2b(
                command, timeout, session, user_id, session_id
            )

        # --- Bubblewrap fallback ---
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

    async def _execute_e2b(
        self,
        command: str,
        timeout: int,
        session: ChatSession,
        user_id: str | None,
        session_id: str | None,
    ) -> ToolResponseBase:
        """Execute command in e2b sandbox."""
        try:
            from backend.copilot.sdk.tool_adapter import get_sandbox_manager

            manager = get_sandbox_manager()
            if manager is None:
                return ErrorResponse(
                    message="E2B sandbox manager not available.",
                    error="sandbox_unavailable",
                    session_id=session_id,
                )

            sandbox = await manager.get_or_create(
                session_id or "default", user_id or "anonymous"
            )
            result = await sandbox.commands.run(
                f"bash -c {shlex.quote(command)}",
                cwd=_SANDBOX_HOME,
                timeout=timeout,
            )

            return BashExecResponse(
                message=f"Command executed (exit {result.exit_code})",
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.exit_code,
                timed_out=False,
                session_id=session_id,
            )
        except Exception as e:
            error_str = str(e)
            if "timeout" in error_str.lower():
                return BashExecResponse(
                    message="Execution timed out",
                    stdout="",
                    stderr=f"Execution timed out after {timeout}s",
                    exit_code=-1,
                    timed_out=True,
                    session_id=session_id,
                )
            return ErrorResponse(
                message=f"E2B execution failed: {e}",
                error=error_str,
                session_id=session_id,
            )


# ------------------------------------------------------------------
# Module-level helpers (placed after classes that call them)
# ------------------------------------------------------------------


def _is_e2b_available() -> bool:
    """Check if e2b sandbox is available via execution context."""
    try:
        from backend.copilot.sdk.tool_adapter import get_sandbox_manager

        return get_sandbox_manager() is not None
    except Exception:
        return False
