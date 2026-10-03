"""Writing files into the turn's working directory.

The model's shell and file tools run on an E2B sandbox in production and in a
bubblewrap-isolated ``/tmp/copilot-<session>`` directory locally. Both branches
live here so every tool that puts a file where the model can reach it —
``read_workspace_file``'s ``save_to_path``, ``read_skill``'s package
materialisation — resolves and validates the destination the same way.
"""

import logging
import os
import shlex

from backend.copilot.context import (
    E2B_WORKDIR,
    get_current_sandbox,
    resolve_sandbox_path,
)
from backend.copilot.tools.sandbox import make_session_path

from .models import ErrorResponse

logger = logging.getLogger(__name__)


def workdir_root(session_id: str) -> str:
    """Directory the model's shell starts in for this turn."""
    if get_current_sandbox() is not None:
        return E2B_WORKDIR
    return make_session_path(session_id)


async def save_to_workdir(
    path: str, content: bytes, session_id: str
) -> str | ErrorResponse:
    """Write *content* to *path* on E2B sandbox or local ephemeral directory.

    Returns the resolved path on success, or an ``ErrorResponse`` on failure.
    """

    sandbox = get_current_sandbox()
    if sandbox is not None:
        remote = resolve_sandbox_path_or_error(path, session_id, "save_to_path")
        if isinstance(remote, ErrorResponse):
            return remote
        try:
            await sandbox.files.write(remote, content)
        except Exception as exc:
            return ErrorResponse(
                message=f"Failed to write to sandbox: {path} ({exc})",
                session_id=session_id,
            )
        return remote

    validated = validate_ephemeral_path(
        path, param_name="save_to_path", session_id=session_id
    )
    if isinstance(validated, ErrorResponse):
        return validated
    try:
        dir_path = os.path.dirname(validated)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(validated, "wb") as f:
            f.write(content)
    except Exception as exc:
        return ErrorResponse(
            message=f"Failed to write to local path: {path} ({exc})",
            session_id=session_id,
        )
    return validated


async def read_workdir_bytes(path: str, session_id: str) -> bytes | None:
    """Read *path* back out of the working directory, or ``None`` when it is
    absent or unreadable — callers use this for bookkeeping files whose
    absence is the ordinary first-run case."""
    sandbox = get_current_sandbox()
    try:
        if sandbox is not None:
            remote = resolve_sandbox_path_or_error(path, session_id, "path")
            if isinstance(remote, ErrorResponse):
                return None
            return bytes(await sandbox.files.read(remote, format="bytes"))
        validated = validate_ephemeral_path(
            path, param_name="path", session_id=session_id
        )
        if isinstance(validated, ErrorResponse):
            return None
        with open(validated, "rb") as f:
            return f.read()
    except Exception:
        return None


async def remove_from_workdir(paths: list[str], session_id: str) -> list[str]:
    """Best-effort delete, returning the paths that are still there.

    A file left behind is one the model can still read or run after it has
    left the package, so the caller needs to know which ones to try again
    rather than record as gone.  The sandbox runs one command for the batch,
    so a failure there is reported against all of them.
    """
    if not paths:
        return []
    sandbox = get_current_sandbox()
    try:
        if sandbox is not None:
            quoted = " ".join(shlex.quote(p) for p in paths)
            await sandbox.commands.run(f"rm -f {quoted}")
            return []
    except Exception:
        logger.warning(
            "[workdir] failed to remove %d stale file(s) in session %s",
            len(paths),
            session_id,
            exc_info=True,
        )
        return list(paths)
    failed: list[str] = []
    for path in paths:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except Exception:
            logger.warning(
                "[workdir] failed to remove %s in session %s",
                path,
                session_id,
                exc_info=True,
            )
            failed.append(path)
    return failed


async def set_executable(
    paths: list[str], executable: bool, session_id: str
) -> list[str]:
    """Best-effort ``chmod +x`` / ``-x``, returning the paths it could not set.

    Rewriting a file does not change an existing mode, so clearing the bit
    needs its own call.  A package whose scripts are not executable is still
    usable through ``python script.py``, so this never raises — but the caller
    must not record a mode that was not applied, or the retry never comes.
    The sandbox runs one command for the batch, so a failure there is reported
    against all of them.
    """
    if not paths:
        return []
    sandbox = get_current_sandbox()
    try:
        if sandbox is not None:
            quoted = " ".join(shlex.quote(p) for p in paths)
            await sandbox.commands.run(f"chmod {'+' if executable else '-'}x {quoted}")
            return []
    except Exception:
        logger.warning(
            "[workdir] failed to set executable=%s on %d file(s) in session %s",
            executable,
            len(paths),
            session_id,
            exc_info=True,
        )
        return list(paths)
    failed: list[str] = []
    for path in paths:
        try:
            mode = os.stat(path).st_mode
            os.chmod(path, mode | 0o111 if executable else mode & ~0o111)
        except Exception:
            logger.warning(
                "[workdir] failed to set executable=%s on %s in session %s",
                executable,
                path,
                session_id,
                exc_info=True,
            )
            failed.append(path)
    return failed


def resolve_sandbox_path_or_error(
    path: str, session_id: str | None, param_name: str
) -> str | ErrorResponse:
    """Normalize *path* to an absolute sandbox path under :data:`E2B_WORKDIR`.

    Delegates to :func:`~backend.copilot.context.resolve_sandbox_path` and
    wraps any ``ValueError`` into an :class:`ErrorResponse`.
    """
    try:
        return resolve_sandbox_path(path)
    except ValueError:
        return ErrorResponse(
            message=f"{param_name} must be within {E2B_WORKDIR}",
            session_id=session_id,
        )


def validate_ephemeral_path(
    path: str, *, param_name: str, session_id: str
) -> ErrorResponse | str:
    """Validate that *path* is inside the session's ephemeral directory.

    Uses the session-specific directory (``make_session_path(session_id)``)
    rather than the bare prefix, so ``/tmp/copilot-evil/...`` is rejected.

    Returns the resolved real path on success, or an ``ErrorResponse`` when the
    path escapes the session directory.
    """
    session_dir = os.path.realpath(make_session_path(session_id)) + os.sep
    real = os.path.realpath(path)
    if not real.startswith(session_dir):
        return ErrorResponse(
            message=(
                f"{param_name} must be within the ephemeral working "
                f"directory ({make_session_path(session_id)})"
            ),
            session_id=session_id,
        )
    return real
