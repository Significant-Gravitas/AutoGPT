"""Sandbox execution utilities for code execution tools.

Provides filesystem + network isolated command execution using **bubblewrap**
(``bwrap``): whitelist-only filesystem (only system dirs visible read-only),
writable workspace only, clean environment, network blocked.

Tools that call :func:`run_sandboxed` must first check :func:`has_full_sandbox`
and refuse to run if bubblewrap is not available.
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


# ---------------------------------------------------------------------------
# Sandbox capability detection (cached at first call)
# ---------------------------------------------------------------------------

_BWRAP_AVAILABLE: bool | None = None


def has_full_sandbox() -> bool:
    """Return True if bubblewrap is available (filesystem + network isolation).

    On non-Linux platforms (macOS), always returns False.
    """
    global _BWRAP_AVAILABLE
    if _BWRAP_AVAILABLE is None:
        _BWRAP_AVAILABLE = (
            platform.system() == "Linux" and shutil.which("bwrap") is not None
        )
    return _BWRAP_AVAILABLE


WORKSPACE_PREFIX = "/tmp/copilot-"


def make_session_path(session_id: str) -> str:
    """Build a sanitized, session-specific path under :data:`WORKSPACE_PREFIX`.

    Shared by both the SDK working-directory setup and the sandbox tools so
    they always resolve to the same directory for a given session.

    Steps:
        1. Strip all characters except ``[A-Za-z0-9-]``.
        2. Construct ``/tmp/copilot-<safe_id>``.
        3. Validate via ``os.path.normpath`` + ``startswith`` (CodeQL-recognised
           sanitizer) to prevent path traversal.

    Raises:
        ValueError: If the resulting path escapes the prefix.
    """
    import re

    safe_id = re.sub(r"[^A-Za-z0-9-]", "", session_id)
    if not safe_id:
        safe_id = "default"
    path = os.path.normpath(f"{WORKSPACE_PREFIX}{safe_id}")
    if not path.startswith(WORKSPACE_PREFIX):
        raise ValueError(f"Session path escaped prefix: {path}")
    return path


def get_workspace_dir(session_id: str) -> str:
    """Get or create the workspace directory for a session.

    Uses :func:`make_session_path` — the same path the SDK uses — so that
    bash_exec shares the workspace with the SDK file tools.
    """
    workspace = make_session_path(session_id)
    os.makedirs(workspace, exist_ok=True)
    return workspace


# ---------------------------------------------------------------------------
# Bubblewrap command builder
# ---------------------------------------------------------------------------

# System directories mounted read-only inside the sandbox.
# ONLY these are visible — /app, /root, /home, /opt, /var etc. are NOT accessible.
_SYSTEM_RO_BINDS = [
    "/usr",  # binaries, libraries, Python interpreter
    "/etc",  # system config: ld.so, locale, passwd, alternatives
]

# Symlinks to /usr/* on modern Debian, may be real dirs on older systems.
_COMPAT_RO_BINDS = [
    "/bin",  # -> /usr/bin on Debian 13
    "/sbin",  # -> /usr/sbin on Debian 13
    "/lib",  # -> /usr/lib on Debian 13
    "/lib64",  # 64-bit libraries (may not exist)
]


def _build_bwrap_command(
    command: list[str], cwd: str, env: dict[str, str]
) -> list[str]:
    """Build a bubblewrap command with strict filesystem + network isolation.

    Security model:
    - **Whitelist-only filesystem**: only system directories (``/usr``, ``/etc``,
      ``/bin``, ``/lib``) are mounted read-only.  Application code (``/app``),
      home directories, ``/var``, ``/opt``, etc. are NOT accessible at all.
    - **Writable workspace only**: the per-session workspace is the sole
      writable path.
    - **Clean environment**: ``--clearenv`` wipes all inherited env vars.
      Only the explicitly-passed safe env vars are set inside the sandbox.
    - **Network isolation**: ``--unshare-net`` blocks all network access.
    - **New session**: prevents terminal control escape.
    - **Die with parent**: prevents orphaned sandbox processes.
    """
    cmd = [
        "bwrap",
        # Wipe all inherited environment variables (API keys, secrets, etc.)
        "--clearenv",
    ]

    # Set only the safe env vars inside the sandbox
    for key, value in env.items():
        cmd.extend(["--setenv", key, value])

    # System directories: read-only
    for path in _SYSTEM_RO_BINDS:
        cmd.extend(["--ro-bind", path, path])

    # Compat paths: bind only if they exist on the host
    for path in _COMPAT_RO_BINDS:
        if os.path.exists(path):
            cmd.extend(["--ro-bind", path, path])

    cmd.extend(
        [
            # Writable workspace only
            "--bind",
            cwd,
            cwd,
            # Fresh virtual filesystems
            "--dev",
            "/dev",
            "--proc",
            "/proc",
            "--tmpdir",
            "/tmp",
            # Isolation
            "--unshare-net",
            "--die-with-parent",
            "--new-session",
            "--chdir",
            cwd,
            "--",
            *command,
        ]
    )

    return cmd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def run_sandboxed(
    command: list[str],
    cwd: str,
    timeout: int = _DEFAULT_TIMEOUT,
    env: dict[str, str] | None = None,
) -> tuple[str, str, int, bool]:
    """Run a command inside a bubblewrap sandbox.

    Callers **must** check :func:`has_full_sandbox` before calling this
    function.  If bubblewrap is not available, this function raises
    :class:`RuntimeError` rather than running unsandboxed.

    Returns:
        (stdout, stderr, exit_code, timed_out)
    """
    if not has_full_sandbox():
        raise RuntimeError(
            "run_sandboxed() requires bubblewrap but bwrap is not available. "
            "Callers must check has_full_sandbox() before calling this function."
        )

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

    full_command = _build_bwrap_command(command, cwd, safe_env)

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

    except RuntimeError:
        raise
    except Exception as e:
        return "", f"Sandbox error: {e}", -1, False
