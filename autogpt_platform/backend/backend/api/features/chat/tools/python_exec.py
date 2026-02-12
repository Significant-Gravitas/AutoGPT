"""Python execution tool — run Python code in a network-isolated sandbox."""

import logging
import os
from typing import Any

from backend.api.features.chat.model import ChatSession
from backend.api.features.chat.tools.base import BaseTool
from backend.api.features.chat.tools.models import (
    ErrorResponse,
    PythonExecResponse,
    ToolResponseBase,
)
from backend.api.features.chat.tools.sandbox import (
    get_workspace_dir,
    has_network_sandbox,
    run_sandboxed,
)

logger = logging.getLogger(__name__)

# Modules blocked via import hook when kernel network sandbox is unavailable
_BLOCKED_MODULES = {
    "socket",
    "ssl",
    "http",
    "urllib",
    "requests",
    "httpx",
    "aiohttp",
    "ftplib",
    "smtplib",
    "poplib",
    "imaplib",
    "telnetlib",
    "xmlrpc",
    "subprocess",
    "ctypes",
    "multiprocessing",
}

# Security prelude injected before user code (only when unshare unavailable)
_SECURITY_PRELUDE = """\
import builtins as _b
_BLOCKED = {blocked}
_orig = _b.__import__
def _si(name, *a, **k):
    if name.split(".")[0] in _BLOCKED:
        raise ImportError(f"Module '{{name}}' is not available in the sandbox")
    return _orig(name, *a, **k)
_b.__import__ = _si
import os as _os
_os.system = lambda *a, **k: (_ for _ in ()).throw(
    PermissionError("os.system is blocked")
)
_os.popen = lambda *a, **k: (_ for _ in ()).throw(
    PermissionError("os.popen is blocked")
)
del _b, _BLOCKED, _orig, _si, _os
"""


class PythonExecTool(BaseTool):
    """Execute Python code in a sandboxed environment."""

    @property
    def name(self) -> str:
        return "python_exec"

    @property
    def description(self) -> str:
        return (
            "Execute Python code in a sandboxed environment. "
            "SECURITY: All internet/network access is blocked at the kernel level "
            "(no HTTP, sockets, DNS, or any outbound connections). "
            "To fetch web content, use the web_fetch tool instead. "
            "Code runs in an isolated per-session workspace directory — "
            "it cannot read or write files outside that directory. "
            "Execution is killed after the timeout (default 30s, max 120s). "
            "Returns stdout and stderr. "
            "Useful for data processing, calculations, text manipulation, "
            "JSON/CSV parsing, and generating files in the workspace."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute.",
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        "Max execution time in seconds (default 30, max 120)."
                    ),
                    "default": 30,
                },
            },
            "required": ["code"],
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
        code: str = (kwargs.get("code") or "").strip()
        timeout: int = kwargs.get("timeout", 30)
        session_id = session.session_id if session else None

        if not code:
            return ErrorResponse(
                message="No code provided.",
                error="empty_code",
                session_id=session_id,
            )

        workspace = get_workspace_dir(session_id or "default")

        # Add security prelude when kernel network isolation is unavailable
        if not has_network_sandbox():
            prelude = _SECURITY_PRELUDE.format(blocked=repr(_BLOCKED_MODULES))
            full_code = prelude + "\n" + code
        else:
            full_code = code

        script_path = os.path.join(workspace, "_exec.py")
        try:
            with open(script_path, "w") as f:
                f.write(full_code)

            stdout, stderr, exit_code, timed_out = await run_sandboxed(
                command=["python3", "-I", "-u", script_path],
                cwd=workspace,
                timeout=timeout,
            )

            return PythonExecResponse(
                message=(
                    "Execution timed out"
                    if timed_out
                    else f"Code executed (exit {exit_code})"
                ),
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                timed_out=timed_out,
                session_id=session_id,
            )
        finally:
            try:
                os.unlink(script_path)
            except OSError:
                pass
