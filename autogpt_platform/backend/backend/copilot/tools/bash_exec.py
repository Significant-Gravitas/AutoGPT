"""Bash execution tool — run shell commands in a bubblewrap sandbox.

Full Bash scripting is allowed (loops, conditionals, pipes, functions, etc.).
Safety comes from OS-level isolation (bubblewrap): only system dirs visible
read-only, writable workspace only, clean env, no network.

Requires bubblewrap (``bwrap``) — the tool is disabled when bwrap is not
available (e.g. macOS development).
"""

import logging
from typing import Any

from backend.copilot.model import ChatSession

from .base import BaseTool
from .models import BashExecResponse, ErrorResponse, ToolResponseBase
from .sandbox import get_workspace_dir, has_full_sandbox, run_sandboxed

logger = logging.getLogger(__name__)


class BashExecTool(BaseTool):
    """Execute Bash commands in a bubblewrap sandbox."""

    @property
    def name(self) -> str:
        return "bash_exec"

    @property
    def description(self) -> str:
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

        if not has_full_sandbox():
            return ErrorResponse(
                message="bash_exec requires bubblewrap sandbox (Linux only).",
                error="sandbox_unavailable",
                session_id=session_id,
            )

        command: str = (kwargs.get("command") or "").strip()
        timeout: int = kwargs.get("timeout", 30)

        if not command:
            return ErrorResponse(
                message="No command provided.",
                error="empty_command",
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
