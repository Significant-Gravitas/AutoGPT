"""Unified MCP file-tool handlers for both E2B (sandbox) and non-E2B (local) modes.

When E2B is active, Read/Write/Edit/Glob/Grep route to the sandbox so that
all file operations share the same ``/home/user`` and ``/tmp`` filesystems
as ``bash_exec``.

In non-E2B mode (no sandbox), Read/Write/Edit operate on the SDK working
directory (``/tmp/copilot-<session>/``), providing the same truncation
detection and path-validation guarantees.

SDK-internal paths (``~/.claude/projects/…/tool-results/``) are handled
by the separate ``Read`` MCP tool registered in ``tool_adapter.py``.
"""

import asyncio
import base64
import collections
import hashlib
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
    is_sdk_tool_path,
    is_within_allowed_dirs,
    resolve_sandbox_path,
)

logger = logging.getLogger(__name__)

# Default number of lines returned by ``read_file`` when the caller does not
# specify a limit.  Also used as the threshold in ``bridge_to_sandbox`` to
# decide whether the model is requesting the full file (and thus whether the
# bridge copy is worthwhile).
_DEFAULT_READ_LIMIT = 2000

# Per-path lock for edit operations to prevent parallel lost updates.
# When MCP tools are dispatched in parallel (readOnlyHint=True annotation),
# two Edit calls on the same file could race through read-modify-write
# and silently drop one change.  Keyed by resolved absolute path.
# Bounded to _EDIT_LOCKS_MAX entries (LRU eviction) to prevent unbounded
# memory growth across long-running server processes.
_EDIT_LOCKS_MAX = 1_000
_edit_locks: collections.OrderedDict[str, asyncio.Lock] = collections.OrderedDict()

# Inline content above this threshold triggers a warning — it survived this
# time but is dangerously close to the API output-token truncation limit.
_LARGE_CONTENT_WARN_CHARS = 50_000

_READ_BINARY_EXTENSIONS = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".ico",
        ".webp",
        ".pdf",
        ".zip",
        ".gz",
        ".tar",
        ".bz2",
        ".xz",
        ".7z",
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".bin",
        ".o",
        ".a",
        ".pyc",
        ".pyo",
        ".class",
        ".wasm",
        ".mp3",
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".wav",
        ".flac",
        ".sqlite",
        ".db",
    }
)


def _is_likely_binary(path: str) -> bool:
    """Heuristic check for binary files by extension."""
    _, ext = os.path.splitext(path)
    return ext.lower() in _READ_BINARY_EXTENSIONS


_PARTIAL_TRUNCATION_MSG = (
    "Your Write call was truncated (file_path missing but content "
    "was present). The content was too large for a single tool call. "
    "Write in chunks: use bash_exec with "
    "'cat > file << \"EOF\"\\n...\\nEOF' for the first section, "
    "'cat >> file << \"EOF\"\\n...\\nEOF' to append subsequent "
    "sections, then reference the file with "
    "@@agptfile:/path/to/file if needed."
)

_COMPLETE_TRUNCATION_MSG = (
    "Your Write call had empty arguments — this means your previous "
    "response was too long and the tool call was truncated by the API. "
    "Break your work into smaller steps. For large content, write "
    "section-by-section using bash_exec with "
    "'cat > file << \"EOF\"\\n...\\nEOF' and "
    "'cat >> file << \"EOF\"\\n...\\nEOF'."
)

_EDIT_PARTIAL_TRUNCATION_MSG = (
    "Your Edit call was truncated (file_path missing but old_string/new_string "
    "were present). The arguments were too large for a single tool call. "
    "Break your edit into smaller replacements, or use bash_exec with "
    "'sed' for large-scale find-and-replace."
)


def _check_truncation(file_path: str, content: str) -> dict[str, Any] | None:
    """Return an error response if the args look truncated, else ``None``."""
    if not file_path:
        if content:
            return _mcp(_PARTIAL_TRUNCATION_MSG, error=True)
        return _mcp(_COMPLETE_TRUNCATION_MSG, error=True)
    return None


def _resolve_and_validate(
    file_path: str, sdk_cwd: str
) -> tuple[str, None] | tuple[None, dict[str, Any]]:
    """Resolve *file_path* against *sdk_cwd* and validate it stays within bounds.

    Returns ``(resolved_path, None)`` on success, or ``(None, error_response)``
    on failure.
    """
    if not os.path.isabs(file_path):
        resolved = os.path.realpath(os.path.join(sdk_cwd, file_path))
    else:
        resolved = os.path.realpath(file_path)

    if not is_allowed_local_path(resolved, sdk_cwd):
        return None, _mcp(
            f"Path must be within the working directory: {os.path.basename(file_path)}",
            error=True,
        )
    return resolved, None


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


async def _sandbox_write(sandbox: Any, path: str, content: str | bytes) -> None:
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

    *content* may be ``str`` (text) or ``bytes`` (binary).  Both paths
    are handled correctly: text is encoded to bytes for the base64 shell
    pipe, and raw bytes are passed through without any encoding.
    """
    if path == "/tmp" or path.startswith("/tmp/"):
        raw = content.encode() if isinstance(content, str) else content
        encoded = base64.b64encode(raw).decode()
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
    """Read lines from a file — E2B sandbox, local SDK working dir, or SDK-internal paths."""
    if not args:
        return _mcp(
            "Your read_file call had empty arguments \u2014 this means your previous "
            "response was too long and the tool call was truncated by the API. "
            "Break your work into smaller steps.",
            error=True,
        )
    file_path: str = args.get("file_path", "")
    try:
        offset: int = max(0, int(args.get("offset", 0)))
        limit: int = max(1, int(args.get("limit", _DEFAULT_READ_LIMIT)))
    except (ValueError, TypeError):
        return _mcp("Invalid offset/limit \u2014 must be integers.", error=True)

    if not file_path:
        if "offset" in args or "limit" in args:
            return _mcp(
                "Your read_file call was truncated (file_path missing but "
                "offset/limit were present). Resend with the full file_path.",
                error=True,
            )
        return _mcp("file_path is required", error=True)

    # SDK-internal tool-results/tool-outputs paths are on the host filesystem in
    # both E2B and non-E2B mode — always read them locally.
    # When E2B is active, also copy the file into the sandbox so bash_exec can
    # process it further.
    # NOTE: when E2B is active we intentionally use `is_sdk_tool_path` (not
    # `_is_allowed_local`) so that sdk_cwd-relative paths (e.g. "output.txt")
    # are NOT captured here.  In E2B mode the agent's working directory is the
    # sandbox, not sdk_cwd on the host, so relative paths should be read from
    # the sandbox below.
    sandbox_active = _get_sandbox() is not None
    local_check = (
        is_sdk_tool_path(file_path) if sandbox_active else _is_allowed_local(file_path)
    )
    if local_check:
        result = _read_local(file_path, offset, limit)
        if not result.get("isError"):
            sandbox = _get_sandbox()
            if sandbox is not None:
                annotation = await bridge_and_annotate(
                    sandbox, file_path, offset, limit
                )
                if annotation:
                    result["content"][0]["text"] += annotation
        return result

    sandbox = _get_sandbox()
    if sandbox is not None:
        # E2B path — read from sandbox filesystem
        result = _get_sandbox_and_path(file_path)
        if isinstance(result, dict):
            return result
        sandbox, remote = result

        try:
            raw: bytes = await sandbox.files.read(remote, format="bytes")
            content = raw.decode("utf-8", errors="replace")
        except Exception as exc:
            return _mcp(f"Failed to read {os.path.basename(remote)}: {exc}", error=True)

        lines = content.splitlines(keepends=True)
        selected = list(itertools.islice(lines, offset, offset + limit))
        numbered = "".join(
            f"{i + offset + 1:>6}\t{line}" for i, line in enumerate(selected)
        )
        return _mcp(numbered)

    # Non-E2B path — read from SDK working directory
    sdk_cwd = get_sdk_cwd()
    if not sdk_cwd:
        return _mcp("No SDK working directory available", error=True)

    resolved, err = _resolve_and_validate(file_path, sdk_cwd)
    if err is not None:
        return err
    assert resolved is not None

    if _is_likely_binary(resolved):
        return _mcp(
            f"Cannot read binary file: {os.path.basename(resolved)}. "
            "Use bash_exec with 'xxd' or 'file' to inspect binary files.",
            error=True,
        )

    try:
        with open(resolved, encoding="utf-8", errors="replace") as f:
            selected = list(itertools.islice(f, offset, offset + limit))
    except FileNotFoundError:
        return _mcp(f"File not found: {file_path}", error=True)
    except PermissionError:
        return _mcp(f"Permission denied: {file_path}", error=True)
    except Exception as exc:
        return _mcp(f"Failed to read {file_path}: {exc}", error=True)

    numbered = "".join(
        f"{i + offset + 1:>6}\t{line}" for i, line in enumerate(selected)
    )
    return _mcp(numbered)


async def _handle_write_file(args: dict[str, Any]) -> dict[str, Any]:
    """Write content to a file — E2B sandbox or local SDK working directory."""
    if not args:
        return _mcp(_COMPLETE_TRUNCATION_MSG, error=True)
    file_path: str = args.get("file_path", "")
    content: str = args.get("content", "")

    truncation_err = _check_truncation(file_path, content)
    if truncation_err is not None:
        return truncation_err

    sandbox = _get_sandbox()
    if sandbox is not None:
        # E2B path — write to sandbox filesystem
        try:
            remote = resolve_sandbox_path(file_path)
        except ValueError as exc:
            return _mcp(str(exc), error=True)

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
            return _mcp(
                f"Failed to write {os.path.basename(remote)}: {exc}", error=True
            )

        msg = f"Successfully wrote to {file_path}"
        if len(content) > _LARGE_CONTENT_WARN_CHARS:
            logger.warning(
                "[Write] large inline content (%d chars) for %s",
                len(content),
                remote,
            )
            msg += (
                f"\n\nWARNING: The content was very large ({len(content)} chars). "
                "Next time, write large files in sections using bash_exec with "
                "'cat > file << EOF ... EOF' and 'cat >> file << EOF ... EOF' "
                "to avoid output-token truncation."
            )
        return _mcp(msg)

    # Non-E2B path — write to SDK working directory
    sdk_cwd = get_sdk_cwd()
    if not sdk_cwd:
        return _mcp("No SDK working directory available", error=True)

    resolved, err = _resolve_and_validate(file_path, sdk_cwd)
    if err is not None:
        return err
    assert resolved is not None

    try:
        parent = os.path.dirname(resolved)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(resolved, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as exc:
        logger.error("Write failed for %s: %s", resolved, exc, exc_info=True)
        return _mcp(
            f"Failed to write {os.path.basename(resolved)}: {type(exc).__name__}",
            error=True,
        )

    msg = f"Successfully wrote to {file_path}"
    if len(content) > _LARGE_CONTENT_WARN_CHARS:
        logger.warning(
            "[Write] large inline content (%d chars) for %s",
            len(content),
            resolved,
        )
        msg += (
            f"\n\nWARNING: The content was very large ({len(content)} chars). "
            "Next time, write large files in sections using bash_exec with "
            "'cat > file << EOF ... EOF' and 'cat >> file << EOF ... EOF' "
            "to avoid output-token truncation."
        )
    return _mcp(msg)


async def _handle_edit_file(args: dict[str, Any]) -> dict[str, Any]:
    """Replace a substring in a file — E2B sandbox or local SDK working directory."""
    if not args:
        return _mcp(
            "Your Edit call had empty arguments \u2014 this means your previous "
            "response was too long and the tool call was truncated by the API. "
            "Break your work into smaller steps.",
            error=True,
        )
    file_path: str = args.get("file_path", "")
    old_string: str = args.get("old_string", "")
    new_string: str = args.get("new_string", "")
    replace_all: bool = args.get("replace_all", False)

    # Partial truncation: file_path missing but edit strings present
    if not file_path:
        if old_string or new_string:
            return _mcp(_EDIT_PARTIAL_TRUNCATION_MSG, error=True)
        return _mcp(
            "Your Edit call had empty arguments \u2014 this means your previous "
            "response was too long and the tool call was truncated by the API. "
            "Break your work into smaller steps.",
            error=True,
        )

    if not old_string:
        return _mcp("old_string is required", error=True)

    sandbox = _get_sandbox()
    if sandbox is not None:
        # E2B path — edit in sandbox filesystem
        try:
            remote = resolve_sandbox_path(file_path)
        except ValueError as exc:
            return _mcp(str(exc), error=True)

        parent = os.path.dirname(remote)
        canonical_parent = await _check_sandbox_symlink_escape(sandbox, parent)
        if canonical_parent is None:
            return _mcp(
                f"Path must be within {E2B_ALLOWED_DIRS_STR}: {os.path.basename(parent)}",
                error=True,
            )
        remote = os.path.join(canonical_parent, os.path.basename(remote))

        try:
            raw = bytes(await sandbox.files.read(remote, format="bytes"))
            content = raw.decode("utf-8", errors="replace")
        except Exception as exc:
            return _mcp(f"Failed to read {os.path.basename(remote)}: {exc}", error=True)

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
            return _mcp(
                f"Failed to write {os.path.basename(remote)}: {exc}", error=True
            )

        return _mcp(
            f"Edited {file_path} ({count} replacement{'s' if count > 1 else ''})"
        )

    # Non-E2B path — edit in SDK working directory
    sdk_cwd = get_sdk_cwd()
    if not sdk_cwd:
        return _mcp("No SDK working directory available", error=True)

    resolved, err = _resolve_and_validate(file_path, sdk_cwd)
    if err is not None:
        return err
    assert resolved is not None

    # Per-path lock prevents parallel edits from racing through
    # the read-modify-write cycle and silently dropping changes.
    # LRU-bounded: evict the oldest entry when the dict is full so that
    # _edit_locks does not grow unboundedly in long-running server processes.
    if resolved not in _edit_locks:
        if len(_edit_locks) >= _EDIT_LOCKS_MAX:
            _edit_locks.popitem(last=False)
        _edit_locks[resolved] = asyncio.Lock()
    else:
        _edit_locks.move_to_end(resolved)
    lock = _edit_locks[resolved]
    async with lock:
        try:
            with open(resolved, encoding="utf-8") as f:
                content = f.read()
        except FileNotFoundError:
            return _mcp(f"File not found: {file_path}", error=True)
        except PermissionError:
            return _mcp(f"Permission denied: {file_path}", error=True)
        except Exception as exc:
            return _mcp(f"Failed to read {file_path}: {exc}", error=True)

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

        # Yield to the event loop between the read and write phases so other
        # coroutines waiting on this lock can be scheduled.  The lock above
        # ensures they cannot enter the critical section until we release it.
        await asyncio.sleep(0)

        try:
            with open(resolved, "w", encoding="utf-8") as f:
                f.write(updated)
        except Exception as exc:
            return _mcp(f"Failed to write {file_path}: {exc}", error=True)

    return _mcp(f"Edited {file_path} ({count} replacement{'s' if count > 1 else ''})")


async def _handle_glob(args: dict[str, Any]) -> dict[str, Any]:
    """Find files matching a name pattern inside the sandbox using ``find``."""
    if not args:
        return _mcp(
            "Your glob call had empty arguments \u2014 this means your previous "
            "response was too long and the tool call was truncated by the API. "
            "Break your work into smaller steps.",
            error=True,
        )
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
    if not args:
        return _mcp(
            "Your grep call had empty arguments \u2014 this means your previous "
            "response was too long and the tool call was truncated by the API. "
            "Break your work into smaller steps.",
            error=True,
        )
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


# Bridging: copy SDK-internal files into E2B sandbox

# Files larger than this are written to /home/user/ via sandbox.files.write()
# instead of /tmp/ via shell base64, to avoid shell argument length limits
# and E2B command timeouts.  Base64 expands content by ~33%, so keep this
# well under the typical Linux ARG_MAX (128 KB).
_BRIDGE_SHELL_MAX_BYTES = 32 * 1024  # 32 KB
# Files larger than this are skipped entirely to avoid excessive transfer times.
_BRIDGE_SKIP_BYTES = 50 * 1024 * 1024  # 50 MB


async def bridge_to_sandbox(
    sandbox: Any, file_path: str, offset: int, limit: int
) -> str | None:
    """Best-effort copy of a host-side SDK file into the E2B sandbox.

    When the model reads an SDK-internal file (e.g. tool-results), it often
    wants to process the data with bash.  Copying the file into the sandbox
    under a stable name lets ``bash_exec`` access it without extra steps.

    Only copies when offset=0 and limit is large enough to indicate the model
    wants the full file.  Errors are logged but never propagated.

    Returns the sandbox path on success, or ``None`` on skip/failure.

    Size handling:
    - <= 32 KB: written to ``/tmp/<hash>-<basename>`` via shell base64
      (``_sandbox_write``).  Kept small to stay within ARG_MAX.
    - 32 KB - 50 MB: written to ``/home/user/<hash>-<basename>`` via
      ``sandbox.files.write()`` to avoid shell argument length limits.
    - > 50 MB: skipped entirely with a warning.

    The sandbox filename is prefixed with a short hash of the full source
    path to avoid collisions when different source files share the same
    basename (e.g. multiple ``result.json`` files).
    """
    if offset != 0 or limit < _DEFAULT_READ_LIMIT:
        return None
    try:
        expanded = os.path.realpath(os.path.expanduser(file_path))
        basename = os.path.basename(expanded)
        source_id = hashlib.sha256(expanded.encode()).hexdigest()[:12]
        unique_name = f"{source_id}-{basename}"
        file_size = os.path.getsize(expanded)
        if file_size > _BRIDGE_SKIP_BYTES:
            logger.warning(
                "[E2B] Skipping bridge for large file (%d bytes): %s",
                file_size,
                basename,
            )
            return None

        def _read_bytes() -> bytes:
            with open(expanded, "rb") as fh:
                return fh.read()

        raw_content = await asyncio.to_thread(_read_bytes)
        try:
            text_content: str | None = raw_content.decode("utf-8")
        except UnicodeDecodeError:
            text_content = None
        data: str | bytes = text_content if text_content is not None else raw_content
        if file_size <= _BRIDGE_SHELL_MAX_BYTES:
            sandbox_path = f"/tmp/{unique_name}"
            await _sandbox_write(sandbox, sandbox_path, data)
        else:
            sandbox_path = f"/home/user/{unique_name}"
            await sandbox.files.write(sandbox_path, data)
        logger.info(
            "[E2B] Bridged SDK file to sandbox: %s -> %s", basename, sandbox_path
        )
        return sandbox_path
    except Exception:
        logger.warning(
            "[E2B] Failed to bridge SDK file to sandbox: %s",
            file_path,
            exc_info=True,
        )
        return None


async def bridge_and_annotate(
    sandbox: Any, file_path: str, offset: int, limit: int
) -> str | None:
    """Bridge a host file to the sandbox and return a newline-prefixed annotation.

    Combines ``bridge_to_sandbox`` with the standard annotation suffix so
    callers don't need to duplicate the pattern.  Returns a string like
    ``"\\n[Sandbox copy available at /tmp/abc-file.txt]"`` on success, or
    ``None`` if bridging was skipped or failed.
    """
    sandbox_path = await bridge_to_sandbox(sandbox, file_path, offset, limit)
    if sandbox_path is None:
        return None
    return f"\n[Sandbox copy available at {sandbox_path}]"


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
        },
        _handle_grep,
    ),
]

E2B_FILE_TOOL_NAMES: list[str] = [name for name, *_ in E2B_FILE_TOOLS]


# ---------------------------------------------------------------------------
# Unified tool descriptors — used by tool_adapter.py in both E2B and non-E2B modes
# ---------------------------------------------------------------------------

WRITE_TOOL_NAME = "Write"
WRITE_TOOL_DESCRIPTION = (
    "Write or create a file. Parent directories are created automatically. "
    "For large content (>2000 words), prefer writing in sections using "
    "bash_exec with 'cat > file' and 'cat >> file' instead."
)
WRITE_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "file_path": {
            "type": "string",
            "description": (
                "The path to the file to write. "
                "Relative paths are resolved against the working directory."
            ),
        },
        "content": {
            "type": "string",
            "description": "The content to write to the file.",
        },
    },
}

READ_TOOL_NAME = "read_file"
READ_TOOL_DESCRIPTION = (
    "Read a file from the working directory. Returns content with line numbers "
    "(cat -n format). Use offset and limit to read specific ranges for large files."
)
READ_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "file_path": {
            "type": "string",
            "description": (
                "The path to the file to read. "
                "Relative paths are resolved against the working directory."
            ),
        },
        "offset": {
            "type": "integer",
            "description": (
                "Line number to start reading from (0-indexed). Default: 0."
            ),
        },
        "limit": {
            "type": "integer",
            "description": "Number of lines to read. Default: 2000.",
        },
    },
}

EDIT_TOOL_NAME = "Edit"
EDIT_TOOL_DESCRIPTION = (
    "Make targeted text replacements in a file. Finds old_string in the file "
    "and replaces it with new_string. For replacing all occurrences, set "
    "replace_all=true."
)
EDIT_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "file_path": {
            "type": "string",
            "description": (
                "The path to the file to edit. "
                "Relative paths are resolved against the working directory."
            ),
        },
        "old_string": {
            "type": "string",
            "description": "The text to find in the file.",
        },
        "new_string": {
            "type": "string",
            "description": "The replacement text.",
        },
        "replace_all": {
            "type": "boolean",
            "description": (
                "Replace all occurrences of old_string (default: false). "
                "When false, old_string must appear exactly once."
            ),
        },
    },
}


def get_write_tool_handler() -> Callable[..., Any]:
    """Return the Write handler for non-E2B mode."""
    return _handle_write_file


def get_read_tool_handler() -> Callable[..., Any]:
    """Return the Read handler for non-E2B mode."""
    return _handle_read_file


def get_edit_tool_handler() -> Callable[..., Any]:
    """Return the Edit handler for non-E2B mode."""
    return _handle_edit_file
