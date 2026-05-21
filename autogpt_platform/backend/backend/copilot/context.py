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

    from backend.copilot.permissions import CopilotPermissions


# Allowed base directory for the Read tool.  Public so service.py can use it
# for sweep operations without depending on a private implementation detail.
# Respects CLAUDE_CONFIG_DIR env var, consistent with transcript.py's
# projects_base() function.
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

# Current execution's capability filter.  None means "no restrictions".
# Set by set_execution_context(); read by run_block and service.py.
_current_permissions: "ContextVar[CopilotPermissions | None]" = ContextVar(
    "_current_permissions", default=None
)


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
    session: ChatSession | None,
    sandbox: "AsyncSandbox | None" = None,
    sdk_cwd: str | None = None,
    permissions: "CopilotPermissions | None" = None,
) -> None:
    """Set per-turn context variables used by file-resolution tool handlers."""
    _current_user_id.set(user_id)
    _current_session.set(session)
    _current_sandbox.set(sandbox)
    _current_sdk_cwd.set(sdk_cwd or "")
    _current_project_dir.set(_encode_cwd_for_cli(sdk_cwd) if sdk_cwd else "")
    _current_permissions.set(permissions)


def get_execution_context() -> tuple[str | None, ChatSession | None]:
    """Return the current (user_id, session) pair for the active request."""
    return _current_user_id.get(), _current_session.get()


def get_current_permissions() -> "CopilotPermissions | None":
    """Return the capability filter for the current execution, or None if unrestricted."""
    return _current_permissions.get()


def get_current_sandbox() -> "AsyncSandbox | None":
    """Return the E2B sandbox for the current session, or None if not active."""
    return _current_sandbox.get()


def get_sdk_cwd() -> str:
    """Return the SDK working directory for the current session (empty string if unset)."""
    return _current_sdk_cwd.get()


E2B_WORKDIR = "/home/user"
E2B_ALLOWED_DIRS: tuple[str, ...] = (E2B_WORKDIR, "/tmp")
E2B_ALLOWED_DIRS_STR: str = " or ".join(E2B_ALLOWED_DIRS)


def is_within_allowed_dirs(path: str) -> bool:
    """Return True if *path* is within one of the allowed sandbox directories."""
    for allowed in E2B_ALLOWED_DIRS:
        if path == allowed or path.startswith(allowed + "/"):
            return True
    return False


def is_sdk_tool_path(path: str) -> bool:
    """Return True if *path* is an SDK-internal tool-results or tool-outputs path.

    These paths exist on the host filesystem (not in the E2B sandbox) and are
    created by the Claude Agent SDK itself.  In E2B mode, only these paths should
    be read from the host; all other paths should be read from the sandbox.

    This is a strict subset of ``is_allowed_local_path`` — it intentionally
    excludes ``sdk_cwd`` paths because those are the agent's working directory,
    which in E2B mode is the sandbox, not the host.
    """
    if not path:
        return False

    if path.startswith("~"):
        resolved = os.path.realpath(os.path.expanduser(path))
    elif not os.path.isabs(path):
        # Relative paths cannot resolve to an absolute SDK-internal path
        return False
    else:
        resolved = os.path.realpath(path)

    encoded = _current_project_dir.get("")
    if not encoded:
        return False

    project_dir = os.path.realpath(os.path.join(SDK_PROJECTS_DIR, encoded))
    if not project_dir.startswith(SDK_PROJECTS_DIR + os.sep):
        return False
    if not resolved.startswith(project_dir + os.sep):
        return False

    relative = resolved[len(project_dir) + 1 :]
    parts = relative.split(os.sep)
    return (
        len(parts) >= 3
        and _UUID_RE.match(parts[0]) is not None
        and parts[1] in ("tool-results", "tool-outputs")
    )


# Anchored matches for host-side SDK tool-result paths. Used by
# wrong-tool detectors (bash_exec, read_workspace_file) to redirect the
# model to ``read_tool_result`` / ``@@agptfile:`` rather than letting it
# bounce off ``Permission denied`` repeatedly.
#
# Two shapes are matched:
#   (1) absolute SDK path:  /<...>/.claude/projects/<cwd>/<uuid>/tool-results/<file>.json
#   (2) model shorthand:    tool-(results|outputs)/<toolu|mcp>_<id>.json
#
# The shorthand branch requires the SDK's actual ID prefix so a user
# repo that happens to use a ``tool-outputs/`` directory name does NOT
# trigger a false redirect on legitimate paths like
# ``my-pipeline/tool-outputs/data.json``.
_SDK_TOOL_RESULT_RE = re.compile(
    r"(?:^|[/\s'\"])/?\.claude/projects/[^/\s]+/[^/\s]+/tool-(?:results|outputs)/"
    r"[\w-]+\.json"
    r"|(?:^|[/\s'\"])tool-(?:results|outputs)/(?:toolu|mcp)_[\w-]+\.json",
    re.IGNORECASE,
)


def looks_like_sdk_tool_result_path(text: str) -> bool:
    """Return True if *text* references a host-side SDK tool-result file.

    Anchored regex match — avoids false positives on user paths that
    happen to contain a ``tool-outputs/`` directory by requiring either
    the full ``.claude/projects/<…>/tool-(results|outputs)/<file>.json``
    shape or the SDK's ``(toolu|mcp)_<id>.json`` filename pattern.
    """
    if not text:
        return False
    return _SDK_TOOL_RESULT_RE.search(text) is not None


# Shell separators used by ``_extract_offending_segment`` to bound the
# path-like token that triggered the redirect. Quotes are included so a
# quoted SDK path inside a command (e.g. ``cat "/root/.claude/..."``)
# extracts the path without the surrounding quotes.
_SHELL_TOKEN_DELIMS = frozenset(" |;&<>'\"\n\t`()")


def _extract_offending_segment(text: str) -> str:
    """Return the path-shaped token in *text* that tripped a redirect hint.

    Uses the regex match position as the anchor (so false positives like
    ``my-pipeline/tool-outputs/data.json`` never reach this path) and
    walks outward to shell-token delimiters to capture the full
    surrounding path (handles ``--file=/path/...`` and quoted paths).
    Falls back to the first whitespace-separated token so callers that
    pass an isolated path (e.g. ``read_workspace_file``) still echo
    cleanly.
    """
    match = _SDK_TOOL_RESULT_RE.search(text)
    if match:
        # Walk outward from the matched span to shell-token boundaries
        # so we capture the full path token, not just the regex hit
        # (which excludes the leading slash on the absolute branch).
        end = match.end()
        while end < len(text) and text[end] not in _SHELL_TOKEN_DELIMS:
            end += 1
        start = match.start()
        # Skip any leading delimiter the regex consumed (e.g. the space
        # before "/root/.claude/...").
        while start < end and text[start] in _SHELL_TOKEN_DELIMS:
            start += 1
        while start > 0 and text[start - 1] not in _SHELL_TOKEN_DELIMS:
            start -= 1
        return text[start:end]
    stripped = text.strip()
    return stripped.split()[0] if stripped else ""


def sdk_tool_result_redirect_hint(path_or_command: str) -> str:
    """User-facing redirect message for wrong-tool access to SDK tool-results.

    Names the two right ways to get at the bytes so the model doesn't
    burn another turn retrying with yet another wrong tool, and quotes
    the *actual* matched SDK path (not just ``bash``/``cat``) so the
    model knows which fragment of its command tripped the redirect.
    """
    fragment = _extract_offending_segment(path_or_command) if path_or_command else ""
    return (
        "This path lives in the Claude Agent SDK's host-side "
        "tool-results directory, which is not mounted into the bash "
        "sandbox / workspace storage. Use one of:\n"
        "  - `read_tool_result(file_path=...)` for direct reads "
        "(supports offset/limit; auto-unwraps the MCP envelope).\n"
        "  - `@@agptfile:<absolute-path>[<start>-<end>]` inside another "
        "tool's argument to inline a slice of the file's content "
        '(e.g. `bash_exec(command="echo @@agptfile:/root/.claude/'
        'projects/.../result.json[1-200] | jq .field")`).\n'
        f"Offending fragment: {fragment!r}"
    )


def resolve_sandbox_path(path: str) -> str:
    """Normalise *path* to an absolute sandbox path under an allowed directory.

    Allowed directories: ``/home/user`` and ``/tmp``.
    Relative paths are resolved against ``/home/user``.

    Raises :class:`ValueError` if the resolved path escapes the sandbox.
    """
    candidate = path if os.path.isabs(path) else os.path.join(E2B_WORKDIR, path)
    normalized = os.path.normpath(candidate)
    if not is_within_allowed_dirs(normalized):
        raise ValueError(
            f"Path must be within {E2B_ALLOWED_DIRS_STR}: {os.path.basename(path)}"
        )
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
    - Files under ``~/.claude/projects/<encoded-cwd>/<uuid>/tool-results/...``
      or ``tool-outputs/...``.
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
        # Only allow: <encoded-cwd>/<uuid>/<tool-dir>/<file>
        # The SDK always creates a conversation UUID directory between
        # the project dir and the tool directory.
        # Accept both "tool-results" (SDK's persisted outputs) and
        # "tool-outputs" (the model sometimes confuses workspace paths
        # with filesystem paths and generates this variant).
        if resolved.startswith(project_dir + os.sep):
            relative = resolved[len(project_dir) + 1 :]
            parts = relative.split(os.sep)
            # Require exactly: [<uuid>, "tool-results"|"tool-outputs", <file>, ...]
            if (
                len(parts) >= 3
                and _UUID_RE.match(parts[0])
                and parts[1] in ("tool-results", "tool-outputs")
            ):
                return True

    return False
