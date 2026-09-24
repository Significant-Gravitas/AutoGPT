"""Bash execution tool — run shell commands on E2B or in a bubblewrap sandbox.

When an E2B sandbox is available in the current execution context the command
runs directly on the remote E2B cloud environment.  This means:

- **Persistent filesystem**: files survive across turns via HTTP-based sync
  with the sandbox's ``/home/user`` directory (E2B files API), shared with
  SDK Read/Write/Edit tools.
- **Full internet access**: E2B sandboxes have unrestricted outbound network.
- **Execution isolation**: E2B provides a fresh, containerised Linux environment.

Connected accounts reach the box as environment variables (``GH_TOKEN`` and
``GITHUB_TOKEN`` for GitHub).  When the box's egress goes through the
credential swap proxy (``backend.util.e2b_network``), those variables hold a
placeholder and the proxy puts the real value in on the way out, so no stored
credential is ever in the box.  Without the proxy they hold the real token,
as they always have.

When E2B is *not* configured the tool falls back to **bubblewrap** (bwrap):
OS-level isolation with a whitelist-only filesystem, no network, and resource
limits.  Requires bubblewrap to be installed (Linux only).
"""

import logging
import shlex
from typing import Any

from e2b import AsyncSandbox, CommandExitException
from e2b.exceptions import TimeoutException

from backend.copilot.context import (
    E2B_WORKDIR,
    get_current_permissions,
    get_current_sandbox,
    looks_like_sdk_tool_result_path,
    sdk_tool_result_redirect_hint,
)
from backend.copilot.credential_selection import selected_credentials
from backend.copilot.integration_creds import (
    get_github_user_git_identity,
    get_integration_env_vars,
    grant_to_box,
    placeholder_env,
    placeholder_grants,
)
from backend.copilot.model import ChatSession
from backend.copilot.permissions import allowed_providers
from backend.util.e2b_network import proxy_address

from .base import BaseTool
from .connect_integration import requested_scopes
from .models import BashExecResponse, ErrorResponse, ToolResponseBase
from .sandbox import get_workspace_dir, has_full_sandbox, run_sandboxed

logger = logging.getLogger(__name__)


def _build_completion_response(
    stdout: str | None,
    stderr: str | None,
    exit_code: int,
    secret_values: list[str],
    session_id: str | None,
) -> BashExecResponse:
    """The command's result, with any injected token replaced by [REDACTED].

    Not a security control: a literal match on one command's output, beaten by
    base64, ``rev`` or a newline, blind to files read back and to anything sent
    over the network.  It only keeps an accidental ``env`` from printing a
    token into the chat.  Behind the swap proxy there is no token in the box
    and nothing to redact.
    """
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
        # True because _execute_on_e2b hands the box the user's connected
        # accounts (GH_TOKEN etc.): the tokens themselves without the swap
        # proxy, placeholders the proxy turns into them with it.  Either way
        # the command acts with the user's accounts, so only an authenticated
        # user may run one.
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
    ) -> ToolResponseBase:
        """Execute *command* on the E2B sandbox via commands.run().

        Connected accounts go into the command's env (e.g. GH_TOKEN), so CLI
        tools like ``gh`` work without manual authentication: as placeholders
        when the box egresses through the swap proxy, as the real tokens when
        it does not.
        """
        envs: dict[str, str] = {
            "PATH": "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        }
        # Injected token values, redacted from the output (see
        # _build_completion_response).  Placeholders are not secret.
        secret_values: list[str] = []
        if user_id is not None:
            selected = await selected_credentials(session_id)
            # The run's ceiling on connected accounts.  Without the proxy,
            # leaving a provider's token out is what keeps it from the box.
            # Behind it, a provider outside the ceiling gets no grant (and so
            # empty variables); the ceiling recorded when the box's egress was
            # pinned, enforced by the swap service, is what holds.
            providers = allowed_providers(get_current_permissions())
            if proxy_address() is not None:
                grants = await placeholder_grants(
                    user_id, required_scopes, selected, providers
                )
                try:
                    # Before the command runs: its placeholders resolve only
                    # for credentials granted to this box.
                    await grant_to_box(sandbox.sandbox_id, grants)
                except Exception:
                    # The command still runs; its placeholders go out as they
                    # are and fail at the provider, which is the safe side.
                    logger.warning(
                        "[E2B] Could not grant credentials to %.12s",
                        sandbox.sandbox_id,
                        exc_info=True,
                    )
                envs.update(placeholder_env(grants))
            else:
                integration_env = await get_integration_env_vars(
                    user_id, required_scopes, selected, providers
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

        try:
            result = await sandbox.commands.run(
                f"bash -c {shlex.quote(command)}",
                cwd=E2B_WORKDIR,
                timeout=timeout,
                envs=envs,
            )
            return _build_completion_response(
                result.stdout,
                result.stderr,
                result.exit_code,
                secret_values,
                session_id,
            )
        except CommandExitException as exc:
            return _build_completion_response(
                exc.stdout, exc.stderr, exc.exit_code, secret_values, session_id
            )
        except TimeoutException:
            return BashExecResponse(
                message="Execution timed out",
                stdout="",
                stderr=f"Timed out after {timeout}s",
                exit_code=-1,
                timed_out=True,
                session_id=session_id,
            )
        except Exception as exc:
            logger.error("[E2B] bash_exec failed: %s", exc, exc_info=True)
            return ErrorResponse(
                message=f"E2B execution failed: {exc}",
                error="e2b_execution_error",
                session_id=session_id,
            )
