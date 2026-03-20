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
from backend.data.db_accessors import workspace_db
from backend.util.workspace import WorkspaceManager

if TYPE_CHECKING:
    from e2b import AsyncSandbox

# Allowed base directory for the Read tool.  Public so service.py can use it
# for sweep operations without depending on a private implementation detail.
# Respects CLAUDE_CONFIG_DIR env var, consistent with transcript.py's
# _projects_base() function.
_config_dir = os.environ.get("CLAUDE_CONFIG_DIR") or os.path.expanduser("~/.claude")
SDK_PROJECTS_DIR = os.path.realpath(os.path.join(_config_dir, "projects"))

# Compiled UUID pattern for validating conversation directory names.
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


def encode_cwd_for_cli(cwd: str) -> str:
    """Encode a working directory path the same way the Claude CLI does.

    The Claude CLI encodes the absolute cwd as a directory name by replacing
    every non-alphanumeric character with ``-``.  For example
    ``/tmp/copilot-abc`` becomes ``-tmp-copilot-abc``.
    """
    return re.sub(r"[^a-zA-Z0-9]", "-", os.path.realpath(cwd))


# Keep the private alias for internal callers (backwards compat).
_encode_cwd_for_cli = encode_cwd_for_cli


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
E2B_ALLOWED_DIRS: tuple[str, ...] = (E2B_WORKDIR, "/tmp")


def is_within_allowed_dirs(path: str) -> bool:
    """Return True if *path* is within one of the allowed sandbox directories."""
    for allowed in E2B_ALLOWED_DIRS:
        if path == allowed or path.startswith(allowed + "/"):
            return True
    return False


def resolve_sandbox_path(path: str) -> str:
    """Normalise *path* to an absolute sandbox path under an allowed directory.

    Allowed directories: ``/home/user`` and ``/tmp``.
    Relative paths are resolved against ``/home/user``.

    Raises :class:`ValueError` if the resolved path escapes the sandbox.
    """
    candidate = path if os.path.isabs(path) else os.path.join(E2B_WORKDIR, path)
    normalized = os.path.normpath(candidate)
    if not is_within_allowed_dirs(normalized):
        raise ValueError(f"Path must be within {' or '.join(E2B_ALLOWED_DIRS)}: {path}")
    return normalized


async def get_workspace_manager(user_id: str, session_id: str) -> WorkspaceManager:
    """Create a session-scoped :class:`WorkspaceManager`.

    Placed here (rather than in ``tools/workspace_files``) so that modules
    like ``sdk/file_ref`` can import it without triggering the heavy
    ``tools/__init__`` import chain.
    """
    workspace = await workspace_db().get_or_create_workspace(user_id)
    return WorkspaceManager(user_id, workspace.id, session_id)


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
        project_dir = os.path.realpath(os.path.join(SDK_PROJECTS_DIR, encoded))
        # Defence-in-depth: ensure project_dir didn't escape the base.
        if not project_dir.startswith(SDK_PROJECTS_DIR + os.sep):
            return False
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
