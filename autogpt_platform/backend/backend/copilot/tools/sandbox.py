"""Sandbox execution utilities for code execution tools.

Provides filesystem + network isolated command execution using **bubblewrap**
(``bwrap``): whitelist-only filesystem (only system dirs visible read-only),
writable workspace only, clean environment, network blocked.

Tools that call :func:`run_sandboxed` must first check :func:`has_full_sandbox`
and refuse to run if bubblewrap is not available.
"""

import asyncio
import contextlib
import logging
import os
import platform
import shutil
import signal
from typing import BinaryIO

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30
_MAX_TIMEOUT = 120
# Kept per output stream (bytes here, characters for E2B's decoded text); a
# command that prints more is stopped.
MAX_OUTPUT_SIZE = 1_000_000
_READ_CHUNK = 65536
# A killed bwrap is reaped at once; this only bounds a wait that should not block.
_REAP_TIMEOUT = 5


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

# Compat paths: symlinks to /usr/* on modern Debian, real dirs on older systems.
# On Debian 13 these are symlinks (e.g. /bin -> usr/bin).  bwrap --ro-bind
# can't create a symlink target, so we detect and use --symlink instead.
# /lib64 is critical: the ELF dynamic linker lives at /lib64/ld-linux-x86-64.so.2.
_COMPAT_PATHS = [
    ("/bin", "usr/bin"),  # -> /usr/bin on Debian 13
    ("/sbin", "usr/sbin"),  # -> /usr/sbin on Debian 13
    ("/lib", "usr/lib"),  # -> /usr/lib on Debian 13
    ("/lib64", "usr/lib64"),  # 64-bit libraries / ELF interpreter
]

# Resource limits to prevent fork bombs, memory exhaustion, and disk abuse.
# Applied via ulimit inside the sandbox before exec'ing the user command.
_RESOURCE_LIMITS = (
    "ulimit -u 64"  # max 64 processes  (prevents fork bombs)
    " -v 524288"  # 512 MB virtual memory
    " -f 51200"  # 50 MB max file size  (1024-byte blocks)
    " -n 256"  # 256 open file descriptors
    " 2>/dev/null"
)


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
    - **Resource limits**: ulimit caps on processes (64), memory (512MB),
      file size (50MB), and open FDs (256) to prevent fork bombs and abuse.
    - **New session**: prevents terminal control escape.
    - **Die with parent**: prevents orphaned sandbox processes.
    """
    cmd = [
        "bwrap",
        # Create a new user namespace so bwrap can set up sandboxing
        # inside unprivileged Docker containers (no CAP_SYS_ADMIN needed).
        "--unshare-user",
        # Wipe all inherited environment variables (API keys, secrets, etc.)
        "--clearenv",
    ]

    # Set only the safe env vars inside the sandbox
    for key, value in env.items():
        cmd.extend(["--setenv", key, value])

    # System directories: read-only
    for path in _SYSTEM_RO_BINDS:
        cmd.extend(["--ro-bind", path, path])

    # Compat paths: use --symlink when host path is a symlink (Debian 13),
    # --ro-bind when it's a real directory (older distros).
    for path, symlink_target in _COMPAT_PATHS:
        if os.path.islink(path):
            cmd.extend(["--symlink", symlink_target, path])
        elif os.path.exists(path):
            cmd.extend(["--ro-bind", path, path])

    # Wrap the user command with resource limits:
    #   sh -c 'ulimit ...; exec "$@"' -- <original command>
    # `exec "$@"` replaces the shell so there's no extra process overhead,
    # and properly handles arguments with spaces.
    limited_command = [
        "sh",
        "-c",
        f'{_RESOURCE_LIMITS}; exec "$@"',
        "--",
        *command,
    ]

    cmd.extend(
        [
            # Fresh virtual filesystems
            "--dev",
            "/dev",
            "--proc",
            "/proc",
            "--tmpfs",
            "/tmp",
            # Workspace bind AFTER --tmpfs /tmp so it's visible through the tmpfs.
            # (workspace lives under /tmp/copilot-<session>)
            "--bind",
            cwd,
            cwd,
            # Isolation
            "--unshare-net",
            "--die-with-parent",
            "--new-session",
            "--chdir",
            cwd,
            "--",
            *limited_command,
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
        return await _run_capped(full_command, cwd, safe_env, timeout)
    except RuntimeError:
        raise
    except Exception as e:
        return "", f"Sandbox error: {e}", -1, False


async def _run_capped(
    full_command: list[str], cwd: str, env: dict[str, str], timeout: int
) -> tuple[str, str, int, bool]:
    """Run *full_command*, keeping at most ``MAX_OUTPUT_SIZE`` per stream.

    The output pipes are ours rather than the subprocess's, so a kill can close
    them: anything the kill missed then stops feeding this process.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    out_r, out_w = os.pipe()
    err_r, err_w = os.pipe()
    pipes = [os.fdopen(out_r, "rb", 0), os.fdopen(err_r, "rb", 0)]
    outputs = [bytearray(), bytearray()]
    readers: list[asyncio.Task[bool]] = []
    transports: list[asyncio.ReadTransport] = []
    try:
        try:
            proc = await asyncio.create_subprocess_exec(
                *full_command,
                stdout=out_w,
                stderr=err_w,
                cwd=cwd,
                env=env,
                start_new_session=True,  # Own process group for clean kill
            )
        finally:
            os.close(out_w)
            os.close(err_w)
        outcome = "error"
        try:
            for pipe, output in zip(pipes, outputs):
                reader, transport = await _open_reader(pipe)
                transports.append(transport)
                readers.append(asyncio.create_task(_read_capped(reader, output)))
            outcome = await _until_done(proc, readers, deadline)
        finally:
            if outcome != "done":
                _kill_sandbox(proc)
            for task in readers:
                task.cancel()
            for transport in transports:
                transport.close()
            if outcome != "done":
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(proc.wait(), _REAP_TIMEOUT)
    finally:
        for pipe in pipes:
            pipe.close()

    if outcome == "timeout":
        return "", f"Execution timed out after {timeout}s", -1, True
    stdout, stderr = (o.decode("utf-8", errors="replace") for o in outputs)
    if outcome == "exceeded":
        note = (
            f"Printed more than {MAX_OUTPUT_SIZE:,} bytes and was killed; "
            "redirect output to a file and read parts of it"
        )
        return stdout, f"{note}\n{stderr}".rstrip("\n"), -1, False
    return stdout, stderr, proc.returncode or 0, False


async def _until_done(
    proc: asyncio.subprocess.Process,
    readers: "list[asyncio.Task[bool]]",
    deadline: float,
) -> str:
    """Wait for both streams to end and the process to exit: ``"done"``, or
    ``"exceeded"`` as soon as a stream passes the cap, or ``"timeout"``."""
    loop = asyncio.get_running_loop()
    pending = set(readers)
    while pending:
        done, pending = await asyncio.wait(
            pending,
            timeout=max(deadline - loop.time(), 0),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if not done:
            return "timeout"
        if any(task.result() for task in done):
            return "exceeded"
    try:
        await asyncio.wait_for(proc.wait(), max(deadline - loop.time(), 0))
    except asyncio.TimeoutError:
        return "timeout"
    return "done"


async def _open_reader(
    pipe: BinaryIO,
) -> tuple[asyncio.StreamReader, asyncio.ReadTransport]:
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader(limit=_READ_CHUNK)
    transport, _ = await loop.connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(reader), pipe
    )
    return reader, transport


async def _read_capped(reader: asyncio.StreamReader, output: bytearray) -> bool:
    """Read into *output* to end of stream, or stop at the first byte past the
    cap and return True."""
    while chunk := await reader.read(_READ_CHUNK):
        room = MAX_OUTPUT_SIZE - len(output)
        if len(chunk) > room:
            output += chunk[:room]
            return True
        output += chunk
    return False


def _kill_sandbox(proc: asyncio.subprocess.Process) -> None:
    """Kill bwrap's process group; ``proc.kill()`` alone would leave its
    children running until they finish."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass  # Already exited
    except OSError as kill_err:
        logger.warning(f"Failed to kill process group {proc.pid}: {kill_err}")
