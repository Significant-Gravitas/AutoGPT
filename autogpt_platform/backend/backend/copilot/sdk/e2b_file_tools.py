"""MCP file-tool handlers that route to the E2B cloud sandbox.

When E2B is active, these tools replace the SDK built-in Read/Write/Edit/
Glob/Grep so that all file operations share the same ``/home/user``
and ``/tmp`` filesystems as ``bash_exec``.

SDK-internal paths (``~/.claude/projects/…/tool-results/``) are handled
by the separate ``Read`` MCP tool registered in ``tool_adapter.py``.
"""

import itertools
import json
import logging
import os
import shlex
from typing import Any, Callable

from backend.copilot.context import (
    E2B_ALLOWED_DIRS,
    E2B_ALLOWED_DIRS_STR,
    E2B_WORKDIR,
    get_current_sandbox,
    get_sdk_cwd,
    is_allowed_local_path,
    is_within_allowed_dirs,
    resolve_sandbox_path,
)

logger = logging.getLogger(__name__)


async def _check_sandbox_symlink_escape(
    sandbox: Any,
    parent: str,
) -> str | None:
    """Resolve the canonical parent path inside the sandbox to detect symlink escapes.

    ``normpath`` (used by ``resolve_sandbox_path``) only normalises the string;
    ``readlink -f`` follows actual symlinks on the sandbox filesystem.

    Returns the canonical parent path, or ``None`` if the path escapes
    the allowed sandbox directories.

    Note: There is an inherent TOCTOU window between this check and the
    subsequent ``sandbox.files.write()``.  A symlink could theoretically be
    replaced between the two operations.  This is acceptable in the E2B
    sandbox model since the sandbox is single-user and ephemeral.
    """
    canonical_res = await sandbox.commands.run(
        f"readlink -f {shlex.quote(parent or E2B_WORKDIR)}",
        cwd=E2B_WORKDIR,
        timeout=5,
    )
    canonical_parent = (canonical_res.stdout or "").strip()
    if (
        canonical_res.exit_code != 0
        or not canonical_parent
        or not is_within_allowed_dirs(canonical_parent)
    ):
        return None
    return canonical_parent


def _get_sandbox():
    return get_current_sandbox()


def _is_allowed_local(path: str) -> bool:
    return is_allowed_local_path(path, get_sdk_cwd())


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
        remote = resolve_sandbox_path(file_path)
    except ValueError as exc:
        return _mcp(str(exc), error=True)
    return sandbox, remote


async def _sandbox_write(sandbox: Any, path: str, content: str) -> None:
    """Write *content* to *path* inside the sandbox.

    The E2B filesystem API (``sandbox.files.write``) and the command API
    (``sandbox.commands.run``) run as **different users**.  On ``/tmp``
    (which has the sticky bit set) this means ``sandbox.files.write`` can
    create new files but cannot overwrite files previously created by
    ``sandbox.commands.run`` (or itself), because the sticky bit restricts
    deletion/rename to the file owner.

    To work around this, writes targeting ``/tmp`` are performed via
    ``tee`` through the command API, which runs as the sandbox ``user``
    and can therefore always overwrite user-owned files.
    """
    if path == "/tmp" or path.startswith("/tmp/"):
        import base64 as _b64

        encoded = _b64.b64encode(content.encode()).decode()
        result = await sandbox.commands.run(
            f"echo {shlex.quote(encoded)} | base64 -d > {shlex.quote(path)}",
            cwd=E2B_WORKDIR,
            timeout=10,
        )
        if result.exit_code != 0:
            raise RuntimeError(
                f"shell write failed (exit {result.exit_code}): "
                + (result.stderr or "").strip()
            )
    else:
        await sandbox.files.write(path, content)


# Tool handlers


async def _handle_read_file(args: dict[str, Any]) -> dict[str, Any]:
    """Read lines from a sandbox file, falling back to the local host for SDK-internal paths."""
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
        raw: bytes = await sandbox.files.read(remote, format="bytes")
        content = raw.decode("utf-8", errors="replace")
    except Exception as exc:
        return _mcp(f"Failed to read {remote}: {exc}", error=True)

    lines = content.splitlines(keepends=True)
    selected = list(itertools.islice(lines, offset, offset + limit))
    numbered = "".join(
        f"{i + offset + 1:>6}\t{line}" for i, line in enumerate(selected)
    )
    return _mcp(numbered)


async def _handle_write_file(args: dict[str, Any]) -> dict[str, Any]:
    """Write content to a sandbox file, creating parent directories as needed."""
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
        if parent and parent not in E2B_ALLOWED_DIRS:
            await sandbox.files.make_dir(parent)
        canonical_parent = await _check_sandbox_symlink_escape(sandbox, parent)
        if canonical_parent is None:
            return _mcp(
                f"Path must be within {E2B_ALLOWED_DIRS_STR}: {os.path.basename(parent)}",
                error=True,
            )
        remote = os.path.join(canonical_parent, os.path.basename(remote))
        await _sandbox_write(sandbox, remote, content)
    except Exception as exc:
        return _mcp(f"Failed to write {remote}: {exc}", error=True)

    return _mcp(f"Successfully wrote to {remote}")


async def _handle_edit_file(args: dict[str, Any]) -> dict[str, Any]:
    """Replace a substring in a sandbox file, with optional replace-all support."""
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

    parent = os.path.dirname(remote)
    canonical_parent = await _check_sandbox_symlink_escape(sandbox, parent)
    if canonical_parent is None:
        return _mcp(
            f"Path must be within {E2B_ALLOWED_DIRS_STR}: {os.path.basename(parent)}",
            error=True,
        )
    remote = os.path.join(canonical_parent, os.path.basename(remote))

    try:
        raw: bytes = await sandbox.files.read(remote, format="bytes")
        content = raw.decode("utf-8", errors="replace")
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
        await _sandbox_write(sandbox, remote, updated)
    except Exception as exc:
        return _mcp(f"Failed to write {remote}: {exc}", error=True)

    return _mcp(f"Edited {remote} ({count} replacement{'s' if count > 1 else ''})")


async def _handle_glob(args: dict[str, Any]) -> dict[str, Any]:
    """Find files matching a name pattern inside the sandbox using ``find``."""
    pattern: str = args.get("pattern", "")
    path: str = args.get("path", "")

    if not pattern:
        return _mcp("pattern is required", error=True)

    sandbox = _get_sandbox()
    if sandbox is None:
        return _mcp("No E2B sandbox available", error=True)

    try:
        search_dir = resolve_sandbox_path(path) if path else E2B_WORKDIR
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
    """Search file contents by regex inside the sandbox using ``grep -rn``."""
    pattern: str = args.get("pattern", "")
    path: str = args.get("path", "")
    include: str = args.get("include", "")

    if not pattern:
        return _mcp("pattern is required", error=True)

    sandbox = _get_sandbox()
    if sandbox is None:
        return _mcp("No E2B sandbox available", error=True)

    try:
        search_dir = resolve_sandbox_path(path) if path else E2B_WORKDIR
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
        with open(expanded, encoding="utf-8", errors="replace") as fh:
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
        "Read a file from the cloud sandbox (/home/user or /tmp). "
        "Use offset and limit for large files.",
        {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path (relative to /home/user, or absolute under /home/user or /tmp).",
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
        "Write or create a file in the cloud sandbox (/home/user or /tmp). "
        "Parent directories are created automatically. "
        "To copy a workspace file into the sandbox, use "
        "read_workspace_file with save_to_path instead.",
        {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path (relative to /home/user, or absolute under /home/user or /tmp).",
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
                    "description": "Path (relative to /home/user, or absolute under /home/user or /tmp).",
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
