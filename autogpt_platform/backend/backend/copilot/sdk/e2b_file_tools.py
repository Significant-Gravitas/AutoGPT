"""MCP file-tool handlers that route to the E2B cloud sandbox.

When E2B is active, these tools replace the SDK built-in Read/Write/Edit/
Glob/Grep so that all file operations share the same ``/home/user``
filesystem as ``bash_exec``.

SDK-internal paths (``~/.claude/projects/…/tool-results/``) are handled
by the separate ``Read`` MCP tool registered in ``tool_adapter.py``.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import shlex
from typing import Any, Callable

from backend.copilot.tools.e2b_sandbox import E2B_WORKDIR

logger = logging.getLogger(__name__)


# Lazy imports to break circular dependency with tool_adapter.


def _get_sandbox():  # type: ignore[return]
    from .tool_adapter import get_current_sandbox  # noqa: E402

    return get_current_sandbox()


def _is_allowed_local(path: str) -> bool:
    from .tool_adapter import is_allowed_local_path  # noqa: E402

    return is_allowed_local_path(path)


def _resolve_remote(path: str) -> str:
    """Normalise *path* to an absolute sandbox path under ``/home/user``.

    Raises :class:`ValueError` if the resolved path escapes the sandbox.
    """
    candidate = path if os.path.isabs(path) else os.path.join(E2B_WORKDIR, path)
    normalized = os.path.normpath(candidate)
    if normalized != E2B_WORKDIR and not normalized.startswith(E2B_WORKDIR + "/"):
        raise ValueError(f"Path must be within {E2B_WORKDIR}: {path}")
    return normalized


def _mcp(text: str, *, error: bool = False) -> dict[str, Any]:
    if error:
        text = json.dumps({"error": text, "type": "error"})
    return {"content": [{"type": "text", "text": text}], "isError": error}


def _get_sandbox_and_path(
    file_path: str,
) -> tuple[Any, str] | dict[str, Any]:
    """Common preamble: get sandbox + resolve path, or return MCP error."""
    sandbox = _get_sandbox()
    if sandbox is None:
        return _mcp("No E2B sandbox available", error=True)
    try:
        remote = _resolve_remote(file_path)
    except ValueError as exc:
        return _mcp(str(exc), error=True)
    return sandbox, remote


# Tool handlers


async def _handle_read_file(args: dict[str, Any]) -> dict[str, Any]:
    file_path: str = args.get("file_path", "")
    offset: int = max(0, int(args.get("offset", 0)))
    limit: int = max(1, int(args.get("limit", 2000)))

    if not file_path:
        return _mcp("file_path is required", error=True)

    # SDK-internal paths (tool-results, ephemeral working dir) stay on the host.
    if _is_allowed_local(file_path):
        return _read_local(file_path, offset, limit)

    result = _get_sandbox_and_path(file_path)
    if isinstance(result, dict):
        return result
    sandbox, remote = result

    try:
        content: str = await sandbox.files.read(remote, format="text")
    except Exception as exc:
        return _mcp(f"Failed to read {remote}: {exc}", error=True)

    lines = content.splitlines(keepends=True)
    selected = list(itertools.islice(lines, offset, offset + limit))
    numbered = "".join(
        f"{i + offset + 1:>6}\t{line}" for i, line in enumerate(selected)
    )
    return _mcp(numbered)


async def _handle_write_file(args: dict[str, Any]) -> dict[str, Any]:
    file_path: str = args.get("file_path", "")
    content: str = args.get("content", "")

    if not file_path:
        return _mcp("file_path is required", error=True)

    result = _get_sandbox_and_path(file_path)
    if isinstance(result, dict):
        return result
    sandbox, remote = result

    try:
        parent = os.path.dirname(remote)
        if parent and parent != E2B_WORKDIR:
            await sandbox.files.make_dir(parent)
        await sandbox.files.write(remote, content)
    except Exception as exc:
        return _mcp(f"Failed to write {remote}: {exc}", error=True)

    return _mcp(f"Successfully wrote to {remote}")


async def _handle_edit_file(args: dict[str, Any]) -> dict[str, Any]:
    file_path: str = args.get("file_path", "")
    old_string: str = args.get("old_string", "")
    new_string: str = args.get("new_string", "")
    replace_all: bool = args.get("replace_all", False)

    if not file_path:
        return _mcp("file_path is required", error=True)
    if not old_string:
        return _mcp("old_string is required", error=True)

    result = _get_sandbox_and_path(file_path)
    if isinstance(result, dict):
        return result
    sandbox, remote = result

    try:
        content: str = await sandbox.files.read(remote, format="text")
    except Exception as exc:
        return _mcp(f"Failed to read {remote}: {exc}", error=True)

    count = content.count(old_string)
    if count == 0:
        return _mcp(f"old_string not found in {file_path}", error=True)
    if count > 1 and not replace_all:
        return _mcp(
            f"old_string appears {count} times in {file_path}. "
            "Use replace_all=true or provide a more unique string.",
            error=True,
        )

    updated = (
        content.replace(old_string, new_string)
        if replace_all
        else content.replace(old_string, new_string, 1)
    )
    try:
        await sandbox.files.write(remote, updated)
    except Exception as exc:
        return _mcp(f"Failed to write {remote}: {exc}", error=True)

    return _mcp(f"Edited {remote} ({count} replacement{'s' if count > 1 else ''})")


async def _handle_glob(args: dict[str, Any]) -> dict[str, Any]:
    pattern: str = args.get("pattern", "")
    path: str = args.get("path", "")

    if not pattern:
        return _mcp("pattern is required", error=True)

    sandbox = _get_sandbox()
    if sandbox is None:
        return _mcp("No E2B sandbox available", error=True)

    try:
        search_dir = _resolve_remote(path) if path else E2B_WORKDIR
    except ValueError as exc:
        return _mcp(str(exc), error=True)

    cmd = f"find {shlex.quote(search_dir)} -name {shlex.quote(pattern)} -type f 2>/dev/null | head -500"
    try:
        result = await sandbox.commands.run(cmd, cwd=E2B_WORKDIR, timeout=10)
    except Exception as exc:
        return _mcp(f"Glob failed: {exc}", error=True)

    files = [line for line in (result.stdout or "").strip().splitlines() if line]
    return _mcp(json.dumps(files, indent=2))


async def _handle_grep(args: dict[str, Any]) -> dict[str, Any]:
    pattern: str = args.get("pattern", "")
    path: str = args.get("path", "")
    include: str = args.get("include", "")

    if not pattern:
        return _mcp("pattern is required", error=True)

    sandbox = _get_sandbox()
    if sandbox is None:
        return _mcp("No E2B sandbox available", error=True)

    try:
        search_dir = _resolve_remote(path) if path else E2B_WORKDIR
    except ValueError as exc:
        return _mcp(str(exc), error=True)

    parts = ["grep", "-rn", "--color=never"]
    if include:
        parts.extend(["--include", include])
    parts.extend([pattern, search_dir])
    cmd = " ".join(shlex.quote(p) for p in parts) + " 2>/dev/null | head -200"

    try:
        result = await sandbox.commands.run(cmd, cwd=E2B_WORKDIR, timeout=15)
    except Exception as exc:
        return _mcp(f"Grep failed: {exc}", error=True)

    output = (result.stdout or "").strip()
    return _mcp(output if output else "No matches found.")


# Local read (for SDK-internal paths)


def _read_local(file_path: str, offset: int, limit: int) -> dict[str, Any]:
    """Read from the host filesystem (defence-in-depth path check)."""
    if not _is_allowed_local(file_path):
        return _mcp(f"Path not allowed: {file_path}", error=True)
    expanded = os.path.realpath(os.path.expanduser(file_path))
    try:
        with open(expanded) as fh:
            selected = list(itertools.islice(fh, offset, offset + limit))
        numbered = "".join(
            f"{i + offset + 1:>6}\t{line}" for i, line in enumerate(selected)
        )
        return _mcp(numbered)
    except FileNotFoundError:
        return _mcp(f"File not found: {file_path}", error=True)
    except Exception as exc:
        return _mcp(f"Error reading {file_path}: {exc}", error=True)


# Tool descriptors (name, description, schema, handler)

E2B_FILE_TOOLS: list[tuple[str, str, dict[str, Any], Callable[..., Any]]] = [
    (
        "read_file",
        "Read a file from the cloud sandbox (/home/user). "
        "Use offset and limit for large files.",
        {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path (relative to /home/user, or absolute).",
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
        "Write or create a file in the cloud sandbox (/home/user). "
        "Parent directories are created automatically.",
        {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path (relative to /home/user, or absolute).",
                },
                "content": {"type": "string", "description": "Content to write."},
            },
            "required": ["file_path", "content"],
        },
        _handle_write_file,
    ),
    (
        "edit_file",
        "Targeted text replacement in a sandbox file. "
        "old_string must appear in the file and is replaced with new_string.",
        {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path (relative to /home/user, or absolute).",
                },
                "old_string": {"type": "string", "description": "Text to find."},
                "new_string": {"type": "string", "description": "Replacement text."},
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace all occurrences (default: false).",
                },
            },
            "required": ["file_path", "old_string", "new_string"],
        },
        _handle_edit_file,
    ),
    (
        "glob",
        "Search for files by name pattern in the cloud sandbox.",
        {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern (e.g. *.py).",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search. Default: /home/user.",
                },
            },
            "required": ["pattern"],
        },
        _handle_glob,
    ),
    (
        "grep",
        "Search file contents by regex in the cloud sandbox.",
        {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern."},
                "path": {
                    "type": "string",
                    "description": "File or directory. Default: /home/user.",
                },
                "include": {
                    "type": "string",
                    "description": "Glob to filter files (e.g. *.py).",
                },
            },
            "required": ["pattern"],
        },
        _handle_grep,
    ),
]

E2B_FILE_TOOL_NAMES: list[str] = [name for name, *_ in E2B_FILE_TOOLS]
