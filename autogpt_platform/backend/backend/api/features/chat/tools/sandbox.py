"""Sandbox execution utilities for code execution tools.

Provides network-isolated command execution using Linux ``unshare --net``
(kernel-level, no bypass possible) with a fallback for development on macOS.
"""

import asyncio
import logging
import os
import platform
import shutil

logger = logging.getLogger(__name__)

# Output limits — prevent blowing up LLM context
_MAX_OUTPUT_CHARS = 50_000
_DEFAULT_TIMEOUT = 30
_MAX_TIMEOUT = 120


def _check_unshare() -> bool:
    """Check if ``unshare --net`` is available for kernel-level network isolation."""
    if platform.system() != "Linux":
        return False
    return shutil.which("unshare") is not None


# Cached at import time so we don't shell out on every call
_UNSHARE_AVAILABLE: bool | None = None


def has_network_sandbox() -> bool:
    """Return True if kernel-level network isolation is available."""
    global _UNSHARE_AVAILABLE
    if _UNSHARE_AVAILABLE is None:
        _UNSHARE_AVAILABLE = _check_unshare()
    return _UNSHARE_AVAILABLE


def get_workspace_dir(session_id: str) -> str:
    """Get or create the workspace directory for a session."""
    workspace = f"/tmp/copilot-{session_id}"
    os.makedirs(workspace, exist_ok=True)
    return workspace


async def run_sandboxed(
    command: list[str],
    cwd: str,
    timeout: int = _DEFAULT_TIMEOUT,
    env: dict[str, str] | None = None,
) -> tuple[str, str, int, bool]:
    """Run a command in a sandboxed environment.

    Returns:
        (stdout, stderr, exit_code, timed_out)

    Security layers:
    - Network isolation via ``unshare --net`` (Linux)
    - Restricted working directory
    - Minimal environment variables
    - Hard timeout
    """
    timeout = min(max(timeout, 1), _MAX_TIMEOUT)

    safe_env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": cwd,
        "TMPDIR": cwd,
        "LANG": "en_US.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
    }
    if env:
        safe_env.update(env)

    # Wrap with unshare --net on Linux for kernel-level network isolation
    if has_network_sandbox():
        full_command = ["unshare", "--net", *command]
    else:
        full_command = command

    try:
        proc = await asyncio.create_subprocess_exec(
            *full_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=safe_env,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            stdout = stdout_bytes.decode("utf-8", errors="replace")[:_MAX_OUTPUT_CHARS]
            stderr = stderr_bytes.decode("utf-8", errors="replace")[:_MAX_OUTPUT_CHARS]
            return stdout, stderr, proc.returncode or 0, False
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return "", f"Execution timed out after {timeout}s", -1, True

    except Exception as e:
        return "", f"Sandbox error: {e}", -1, False
