"""Shared execution context for copilot SDK tool handlers.

All context variables and their accessors live here so that
``tool_adapter``, ``file_ref``, and ``e2b_file_tools`` can import them
without creating circular dependencies.
"""

import os
import re
from contextvars import ContextVar
from typing import TYPE_CHECKING

from backend.copilot.model import ChatSession

if TYPE_CHECKING:
    from e2b import AsyncSandbox

# Allowed base directory for the Read tool.  Public so service.py can use it
# for sweep operations without depending on a private implementation detail.
SDK_PROJECTS_DIR = os.path.realpath(os.path.expanduser("~/.claude/projects"))

# Compiled UUID-v4 pattern for validating conversation directory names.
# Kept as a module-level constant so the security-relevant pattern is easy
# to audit in one place and avoids recompilation on every call.
_UUID_RE = re.compile(r"^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$", re.IGNORECASE)

# Encoded project-directory name for the current session (e.g.
# "-private-tmp-copilot-<uuid>").  Set by set_execution_context() so path
# validation can scope tool-results reads to the current session.
_current_project_dir: ContextVar[str] = ContextVar("_current_project_dir", default="")

_current_user_id: ContextVar[str | None] = ContextVar("current_user_id", default=None)
_current_session: ContextVar[ChatSession | None] = ContextVar(
    "current_session", default=None
)
_current_sandbox: ContextVar["AsyncSandbox | None"] = ContextVar(
    "_current_sandbox", default=None
)
_current_sdk_cwd: ContextVar[str] = ContextVar("_current_sdk_cwd", default="")


def _encode_cwd_for_cli(cwd: str) -> str:
    """Encode a working directory path the same way the Claude CLI does."""
    return re.sub(r"[^a-zA-Z0-9]", "-", os.path.realpath(cwd))


def set_execution_context(
    user_id: str | None,
    session: ChatSession,
    sandbox: "AsyncSandbox | None" = None,
    sdk_cwd: str | None = None,
) -> None:
    """Set per-turn context variables used by file-resolution tool handlers."""
    _current_user_id.set(user_id)
    _current_session.set(session)
    _current_sandbox.set(sandbox)
    _current_sdk_cwd.set(sdk_cwd or "")
    _current_project_dir.set(_encode_cwd_for_cli(sdk_cwd) if sdk_cwd else "")


def get_execution_context() -> tuple[str | None, ChatSession | None]:
    """Return the current (user_id, session) pair for the active request."""
    return _current_user_id.get(), _current_session.get()


def get_current_sandbox() -> "AsyncSandbox | None":
    """Return the E2B sandbox for the current session, or None if not active."""
    return _current_sandbox.get()


def get_sdk_cwd() -> str:
    """Return the SDK working directory for the current session (empty string if unset)."""
    return _current_sdk_cwd.get()


E2B_WORKDIR = "/home/user"


def resolve_sandbox_path(path: str) -> str:
    """Normalise *path* to an absolute sandbox path under ``/home/user``.

    Raises :class:`ValueError` if the resolved path escapes the sandbox.
    """
    candidate = path if os.path.isabs(path) else os.path.join(E2B_WORKDIR, path)
    normalized = os.path.normpath(candidate)
    if normalized != E2B_WORKDIR and not normalized.startswith(E2B_WORKDIR + "/"):
        raise ValueError(f"Path must be within {E2B_WORKDIR}: {path}")
    return normalized


def is_allowed_local_path(path: str, sdk_cwd: str | None = None) -> bool:
    """Return True if *path* is within an allowed host-filesystem location.

    Allowed:
    - Files under *sdk_cwd* (``/tmp/copilot-<session>/``)
    - Files under ``~/.claude/projects/<encoded-cwd>/<uuid>/tool-results/...``.
      The SDK nests tool-results under a conversation UUID directory;
      the UUID segment is validated with ``_UUID_RE``.
    """
    if not path:
        return False

    if path.startswith("~"):
        resolved = os.path.realpath(os.path.expanduser(path))
    elif not os.path.isabs(path) and sdk_cwd:
        resolved = os.path.realpath(os.path.join(sdk_cwd, path))
    else:
        resolved = os.path.realpath(path)

    if sdk_cwd:
        norm_cwd = os.path.realpath(sdk_cwd)
        if resolved == norm_cwd or resolved.startswith(norm_cwd + os.sep):
            return True

    encoded = _current_project_dir.get("")
    if encoded:
        project_dir = os.path.join(SDK_PROJECTS_DIR, encoded)
        # Only allow: <encoded-cwd>/<uuid>/tool-results/<file>
        # The SDK always creates a conversation UUID directory between
        # the project dir and tool-results/.
        if resolved.startswith(project_dir + os.sep):
            relative = resolved[len(project_dir) + 1 :]
            parts = relative.split(os.sep)
            # Require exactly: [<uuid>, "tool-results", <file>, ...]
            if (
                len(parts) >= 3
                and _UUID_RE.match(parts[0])
                and parts[1] == "tool-results"
            ):
                return True

    return False
