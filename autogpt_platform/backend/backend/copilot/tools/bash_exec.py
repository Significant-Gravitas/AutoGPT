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
import shlex
from typing import Any

from e2b import AsyncSandbox
from e2b.exceptions import TimeoutException

from backend.copilot.context import E2B_WORKDIR, get_current_sandbox
from backend.copilot.integration_creds import get_integration_env_vars
from backend.copilot.model import ChatSession

from .base import BaseTool
from .models import BashExecResponse, ErrorResponse, ToolResponseBase
from .sandbox import get_workspace_dir, has_full_sandbox, run_sandboxed

logger = logging.getLogger(__name__)


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
            "Killed after timeout (default 30s, max 120s)."
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
                    "description": "Max seconds (default 30, max 120).",
                    "default": 30,
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

        command: str = (kwargs.get("command") or "").strip()
        timeout: int = int(kwargs.get("timeout", 30))

        if not command:
            return ErrorResponse(
                message="No command provided.",
                error="empty_command",
                session_id=session_id,
            )

        sandbox = get_current_sandbox()
        if sandbox is not None:
            return await self._execute_on_e2b(
                sandbox, command, timeout, session_id, user_id
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
        sandbox: AsyncSandbox,
        command: str,
        timeout: int,
        session_id: str | None,
        user_id: str | None = None,
    ) -> ToolResponseBase:
        """Execute *command* on the E2B sandbox via commands.run().

        Integration tokens (e.g. GH_TOKEN) are injected into the sandbox env
        for any user with connected accounts. E2B has full internet access, so
        CLI tools like ``gh`` work without manual authentication.
        """
        envs: dict[str, str] = {
            "PATH": "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        }
        # Collect injected secret values so we can scrub them from output.
        secret_values: list[str] = []
        if user_id is not None:
            integration_env = await get_integration_env_vars(user_id)
            secret_values = [v for v in integration_env.values() if v]
            envs.update(integration_env)

        try:
            result = await sandbox.commands.run(
                f"bash -c {shlex.quote(command)}",
                cwd=E2B_WORKDIR,
                timeout=timeout,
                envs=envs,
            )
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            # Scrub injected tokens from command output to prevent exfiltration
            # via `echo $GH_TOKEN`, `env`, `printenv`, etc.
            for secret in secret_values:
                stdout = stdout.replace(secret, "[REDACTED]")
                stderr = stderr.replace(secret, "[REDACTED]")
            return BashExecResponse(
                message=f"Command executed on E2B (exit {result.exit_code})",
                stdout=stdout,
                stderr=stderr,
                exit_code=result.exit_code,
                timed_out=False,
                session_id=session_id,
            )
        except Exception as exc:
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
