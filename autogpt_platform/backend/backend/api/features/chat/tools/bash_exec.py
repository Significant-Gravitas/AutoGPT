"""Bash execution tool — run shell commands in a network-isolated sandbox.

Full Bash scripting is allowed (loops, conditionals, pipes, functions, etc.).
Safety comes from kernel-level network isolation and workspace confinement,
not from restricting language features.
"""

import logging
import re
from typing import Any

from backend.api.features.chat.model import ChatSession
from backend.api.features.chat.tools.base import BaseTool
from backend.api.features.chat.tools.models import (
    BashExecResponse,
    ErrorResponse,
    ToolResponseBase,
)
from backend.api.features.chat.tools.sandbox import (
    get_workspace_dir,
    has_network_sandbox,
    run_sandboxed,
)

logger = logging.getLogger(__name__)

# Destructive patterns blocked regardless of network sandbox
_BLOCKED_PATTERNS: list[tuple[str, str]] = [
    (r"rm\s+-[a-zA-Z]*r[a-zA-Z]*\s+/(?!\w)", "Recursive removal of root paths"),
    (r"dd\s+.*of=/dev/", "Direct disk writes"),
    (r"mkfs\b", "Filesystem formatting"),
    (r":\(\)\s*\{", "Fork bomb"),
    (r"\bshutdown\b|\breboot\b|\bhalt\b|\bpoweroff\b", "System power commands"),
    (r"/dev/sd[a-z]|/dev/nvme|/dev/hd[a-z]", "Raw disk device access"),
]

# Commands blocked when kernel network sandbox is NOT available (fallback)
_NETWORK_COMMANDS = {
    "curl",
    "wget",
    "ssh",
    "scp",
    "sftp",
    "rsync",
    "nc",
    "ncat",
    "netcat",
    "telnet",
    "ftp",
    "ping",
    "traceroute",
    "nslookup",
    "dig",
    "host",
    "nmap",
}


class BashExecTool(BaseTool):
    """Execute Bash commands in a sandboxed environment."""

    @property
    def name(self) -> str:
        return "bash_exec"

    @property
    def description(self) -> str:
        return (
            "Execute a Bash command or script in a sandboxed environment. "
            "Full Bash scripting is supported (loops, conditionals, pipes, functions, etc.). "
            "SECURITY: All internet/network access is blocked at the kernel level "
            "(no curl, wget, nc, or any outbound connections). "
            "To fetch web content, use the web_fetch tool instead. "
            "Commands run in an isolated per-session workspace directory — "
            "they cannot access files outside that directory. "
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
        command: str = (kwargs.get("command") or "").strip()
        timeout: int = kwargs.get("timeout", 30)
        session_id = session.session_id if session else None

        if not command:
            return ErrorResponse(
                message="No command provided.",
                error="empty_command",
                session_id=session_id,
            )

        # Block destructive patterns
        for pattern, reason in _BLOCKED_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return ErrorResponse(
                    message=f"Command blocked: {reason}",
                    error="blocked_command",
                    session_id=session_id,
                )

        # When kernel network sandbox unavailable, block network commands
        if not has_network_sandbox():
            words = set(re.findall(r"\b\w+\b", command))
            blocked = words & _NETWORK_COMMANDS
            if blocked:
                return ErrorResponse(
                    message=(
                        f"Network commands not available: {', '.join(sorted(blocked))}. "
                        "Use web_fetch instead."
                    ),
                    error="network_blocked",
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
