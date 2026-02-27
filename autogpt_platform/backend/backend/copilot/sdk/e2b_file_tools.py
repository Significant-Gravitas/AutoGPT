"""MCP file-tool handlers that route to the E2B cloud sandbox.

When E2B is active, these tools replace the SDK built-in Read/Write/Edit/
Glob/Grep so that all file operations share the same ``/home/user``
filesystem as ``bash_exec``.  SDK-internal local paths (e.g.
``~/.claude/projects/…/tool-results/``) are transparently read from the
host filesystem instead.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import shlex
from typing import Any, Callable

logger = logging.getLogger(__name__)

_E2B_WORKDIR = "/home/user"


# ---------------------------------------------------------------------------
# Path routing
# ---------------------------------------------------------------------------


_ALLOWED_LOCAL_ROOTS = (
    os.path.realpath(os.path.expanduser("~/.claude")),
    "/tmp/copilot-",  # SDK ephemeral cwd prefix (matched by startswith)
)


def _is_local_path(path: str) -> bool:
    """Return True when *path* refers to an allowed SDK-internal host path.

    Only paths under ``~/.claude/`` (tool-results, transcripts) and
    ``/tmp/copilot-*/`` (SDK ephemeral cwd) are allowed locally.
    Everything else is treated as a sandbox path.
    """
    if not path:
        return False
    real = os.path.realpath(os.path.expanduser(path))
    return any(
        real == root or real.startswith(root + os.sep) or real.startswith(root)
        for root in _ALLOWED_LOCAL_ROOTS
    )


def _resolve_remote(path: str) -> str:
    """Normalise *path* to an absolute sandbox path under ``/home/user``."""
    if path.startswith(_E2B_WORKDIR):
        return path
    return os.path.join(_E2B_WORKDIR, path)


# ---------------------------------------------------------------------------
# MCP response helpers
# ---------------------------------------------------------------------------


def _mcp_ok(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "isError": False}


def _mcp_error(text: str) -> dict[str, Any]:
    return {
        "content": [
            {"type": "text", "text": json.dumps({"error": text, "type": "error"})}
        ],
        "isError": True,
    }


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


async def _handle_read_file(args: dict[str, Any]) -> dict[str, Any]:
    file_path: str = args.get("file_path", "")
    offset: int = max(0, int(args.get("offset", 0)))
    limit: int = max(1, int(args.get("limit", 2000)))

    if not file_path:
        return _mcp_error("file_path is required")

    # SDK-internal paths (tool-results, transcripts, …) stay local.
    if _is_local_path(file_path):
        return _read_local(file_path, offset, limit)

    from .tool_adapter import get_current_sandbox

    sandbox = get_current_sandbox()
    if sandbox is None:
        return _mcp_error("No E2B sandbox available")

    remote = _resolve_remote(file_path)
    try:
        content: str = await sandbox.files.read(remote, format="text")
    except Exception as exc:
        return _mcp_error(f"Failed to read {remote}: {exc}")

    lines = content.splitlines(keepends=True)
    selected = list(itertools.islice(lines, offset, offset + limit))
    numbered = "".join(
        f"{i + offset + 1:>6}\t{line}" for i, line in enumerate(selected)
    )
    return _mcp_ok(numbered)


async def _handle_write_file(args: dict[str, Any]) -> dict[str, Any]:
    file_path: str = args.get("file_path", "")
    content: str = args.get("content", "")

    if not file_path:
        return _mcp_error("file_path is required")

    from .tool_adapter import get_current_sandbox

    sandbox = get_current_sandbox()
    if sandbox is None:
        return _mcp_error("No E2B sandbox available")

    remote = _resolve_remote(file_path)
    try:
        parent = os.path.dirname(remote)
        if parent and parent != _E2B_WORKDIR:
            await sandbox.files.make_dir(parent)
        await sandbox.files.write(remote, content)
    except Exception as exc:
        return _mcp_error(f"Failed to write {remote}: {exc}")

    return _mcp_ok(f"Successfully wrote to {remote}")


async def _handle_edit_file(args: dict[str, Any]) -> dict[str, Any]:
    file_path: str = args.get("file_path", "")
    old_string: str = args.get("old_string", "")
    new_string: str = args.get("new_string", "")
    replace_all: bool = args.get("replace_all", False)

    if not file_path:
        return _mcp_error("file_path is required")
    if not old_string:
        return _mcp_error("old_string is required")

    from .tool_adapter import get_current_sandbox

    sandbox = get_current_sandbox()
    if sandbox is None:
        return _mcp_error("No E2B sandbox available")

    remote = _resolve_remote(file_path)
    try:
        content: str = await sandbox.files.read(remote, format="text")
    except Exception as exc:
        return _mcp_error(f"Failed to read {remote}: {exc}")

    count = content.count(old_string)
    if count == 0:
        return _mcp_error(f"old_string not found in {file_path}")
    if count > 1 and not replace_all:
        return _mcp_error(
            f"old_string appears {count} times in {file_path}. "
            "Use replace_all=true or provide a more unique string."
        )

    updated = (
        content.replace(old_string, new_string)
        if replace_all
        else content.replace(old_string, new_string, 1)
    )
    try:
        await sandbox.files.write(remote, updated)
    except Exception as exc:
        return _mcp_error(f"Failed to write {remote}: {exc}")

    return _mcp_ok(f"Edited {remote} ({count} replacement{'s' if count > 1 else ''})")


async def _handle_glob(args: dict[str, Any]) -> dict[str, Any]:
    pattern: str = args.get("pattern", "")
    path: str = args.get("path", "")

    if not pattern:
        return _mcp_error("pattern is required")

    from .tool_adapter import get_current_sandbox

    sandbox = get_current_sandbox()
    if sandbox is None:
        return _mcp_error("No E2B sandbox available")

    search_dir = _resolve_remote(path) if path else _E2B_WORKDIR
    cmd = f"find {shlex.quote(search_dir)} -name {shlex.quote(pattern)} -type f 2>/dev/null | head -500"
    try:
        result = await sandbox.commands.run(cmd, cwd=_E2B_WORKDIR, timeout=10)
    except Exception as exc:
        return _mcp_error(f"Glob failed: {exc}")

    files = [line for line in (result.stdout or "").strip().splitlines() if line]
    return _mcp_ok(json.dumps(files, indent=2))


async def _handle_grep(args: dict[str, Any]) -> dict[str, Any]:
    pattern: str = args.get("pattern", "")
    path: str = args.get("path", "")
    include: str = args.get("include", "")

    if not pattern:
        return _mcp_error("pattern is required")

    from .tool_adapter import get_current_sandbox

    sandbox = get_current_sandbox()
    if sandbox is None:
        return _mcp_error("No E2B sandbox available")

    search_dir = _resolve_remote(path) if path else _E2B_WORKDIR
    parts = ["grep", "-rn", "--color=never"]
    if include:
        parts.extend(["--include", include])
    parts.extend([pattern, search_dir])
    cmd = " ".join(shlex.quote(p) for p in parts) + " 2>/dev/null | head -200"

    try:
        result = await sandbox.commands.run(cmd, cwd=_E2B_WORKDIR, timeout=15)
    except Exception as exc:
        return _mcp_error(f"Grep failed: {exc}")

    output = (result.stdout or "").strip()
    return _mcp_ok(output if output else "No matches found.")


# ---------------------------------------------------------------------------
# Local read (for SDK-internal paths)
# ---------------------------------------------------------------------------


def _read_local(file_path: str, offset: int, limit: int) -> dict[str, Any]:
    """Read a file from the host filesystem (SDK-internal paths only)."""
    expanded = os.path.expanduser(file_path)
    real = os.path.realpath(expanded)
    if not any(
        real == root or real.startswith(root + os.sep) or real.startswith(root)
        for root in _ALLOWED_LOCAL_ROOTS
    ):
        return _mcp_error(f"Local path not allowed: {file_path}")
    try:
        with open(real) as fh:
            selected = list(itertools.islice(fh, offset, offset + limit))
        content = "".join(selected)
        return _mcp_ok(content)
    except FileNotFoundError:
        return _mcp_error(f"File not found: {file_path}")
    except Exception as exc:
        return _mcp_error(f"Error reading {file_path}: {exc}")


# ---------------------------------------------------------------------------
# Tool descriptors — schemas & descriptions
# ---------------------------------------------------------------------------

E2B_FILE_TOOLS: list[tuple[str, str, dict[str, Any], Callable[..., Any]]] = [
    (
        "read_file",
        (
            "Read a file from the cloud sandbox working directory (/home/user). "
            "Use offset and limit to read specific line ranges for large files."
        ),
        {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file (relative to /home/user, or absolute).",
                },
                "offset": {
                    "type": "integer",
                    "description": "Line to start reading from (0-indexed). Default: 0.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of lines to read. Default: 2000.",
                },
            },
            "required": ["file_path"],
        },
        _handle_read_file,
    ),
    (
        "write_file",
        (
            "Write or create a file in the cloud sandbox working directory (/home/user). "
            "Parent directories are created automatically."
        ),
        {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file (relative to /home/user, or absolute).",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file.",
                },
            },
            "required": ["file_path", "content"],
        },
        _handle_write_file,
    ),
    (
        "edit_file",
        (
            "Perform a targeted text replacement in a file on the cloud sandbox. "
            "The old_string must appear in the file and will be replaced with new_string."
        ),
        {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file (relative to /home/user, or absolute).",
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact text to find and replace.",
                },
                "new_string": {
                    "type": "string",
                    "description": "The replacement text.",
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace all occurrences (default: false, replace first only).",
                },
            },
            "required": ["file_path", "old_string", "new_string"],
        },
        _handle_edit_file,
    ),
    (
        "glob",
        (
            "Search for files by name pattern in the cloud sandbox. "
            "Returns a list of matching file paths under /home/user."
        ),
        {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": 'The glob pattern to match (e.g. "*.py", "test_*").',
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (relative to /home/user). Default: /home/user.",
                },
            },
            "required": ["pattern"],
        },
        _handle_glob,
    ),
    (
        "grep",
        (
            "Search file contents by regex pattern in the cloud sandbox. "
            "Returns matching lines with file paths and line numbers."
        ),
        {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The regex pattern to search for.",
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search (relative to /home/user). Default: /home/user.",
                },
                "include": {
                    "type": "string",
                    "description": 'Glob to filter files (e.g. "*.py"). Default: all files.',
                },
            },
            "required": ["pattern"],
        },
        _handle_grep,
    ),
]

E2B_FILE_TOOL_NAMES: list[str] = [name for name, *_ in E2B_FILE_TOOLS]
