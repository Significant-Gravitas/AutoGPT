"""File reference protocol for tool call inputs.

Allows the LLM to pass a file reference instead of embedding large content
inline.  The processor expands ``@@agptfile:<uri>[<start>-<end>]`` tokens in tool
arguments before the tool is executed.

Protocol
--------

    @@agptfile:<uri>[<start>-<end>]

``<uri>`` (required)
    - ``workspace://<file_id>`` — workspace file by ID
    - ``workspace://<file_id>#<mime>`` — same, MIME hint is ignored for reads
    - ``workspace:///<path>`` — workspace file by virtual path
    - ``/absolute/local/path`` — ephemeral or sdk_cwd file (validated by
      :func:`~backend.copilot.sdk.tool_adapter.is_allowed_local_path`)
    - Any absolute path that resolves inside the E2B sandbox
      (``/home/user/...``) when a sandbox is active

``[<start>-<end>]`` (optional)
    Line range, 1-indexed inclusive.  Examples: ``[1-100]``, ``[50-200]``.
    Omit to read the entire file.

Examples
--------
    @@agptfile:workspace://abc123
    @@agptfile:workspace://abc123[10-50]
    @@agptfile:workspace:///reports/q1.md
    @@agptfile:/tmp/copilot-<session>/output.py[1-80]
    @@agptfile:/home/user/script.sh
"""

import itertools
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

from backend.copilot.context import (
    get_current_sandbox,
    get_sdk_cwd,
    is_allowed_local_path,
    resolve_sandbox_path,
)
from backend.copilot.model import ChatSession
from backend.copilot.tools.workspace_files import get_manager
from backend.util.file import parse_workspace_uri


class FileRefExpansionError(Exception):
    """Raised when a ``@@agptfile:`` reference in tool call args fails to resolve.

    Separating this from inline substitution lets callers (e.g. the MCP tool
    wrapper) block tool execution and surface a helpful error to the model
    rather than passing an ``[file-ref error: …]`` string as actual input.
    """


logger = logging.getLogger(__name__)

FILE_REF_PREFIX = "@@agptfile:"

# Matches:  @@agptfile:<uri>[start-end]?
#   Group 1 – URI; must start with '/' (absolute path) or 'workspace://'
#   Group 2 – start line (optional)
#   Group 3 – end line (optional)
_FILE_REF_RE = re.compile(
    re.escape(FILE_REF_PREFIX) + r"((?:workspace://|/)[^\[\s]*)(?:\[(\d+)-(\d+)\])?"
)

# Maximum characters returned for a single file reference expansion.
_MAX_EXPAND_CHARS = 200_000
# Maximum total characters across all @@agptfile: expansions in one string.
_MAX_TOTAL_EXPAND_CHARS = 1_000_000


@dataclass
class FileRef:
    uri: str
    start_line: int | None  # 1-indexed, inclusive
    end_line: int | None  # 1-indexed, inclusive


def parse_file_ref(text: str) -> FileRef | None:
    """Return a :class:`FileRef` if *text* is a bare file reference token.

    A "bare token" means the entire string matches the ``@@agptfile:...`` pattern
    (after stripping whitespace).  Use :func:`expand_file_refs_in_string` to
    expand references embedded in larger strings.
    """
    m = _FILE_REF_RE.fullmatch(text.strip())
    if not m:
        return None
    start = int(m.group(2)) if m.group(2) else None
    end = int(m.group(3)) if m.group(3) else None
    if start is not None and start < 1:
        return None
    if end is not None and end < 1:
        return None
    if start is not None and end is not None and end < start:
        return None
    return FileRef(uri=m.group(1), start_line=start, end_line=end)


def _apply_line_range(text: str, start: int | None, end: int | None) -> str:
    """Slice *text* to the requested 1-indexed line range (inclusive)."""
    if start is None and end is None:
        return text
    lines = text.splitlines(keepends=True)
    s = (start - 1) if start is not None else 0
    e = end if end is not None else len(lines)
    selected = list(itertools.islice(lines, s, e))
    return "".join(selected)


async def read_file_bytes(
    uri: str,
    user_id: str | None,
    session: ChatSession,
) -> bytes:
    """Resolve *uri* to raw bytes using workspace, local, or E2B path logic.

    Raises :class:`ValueError` if the URI cannot be resolved.
    """
    # Strip MIME fragment (e.g. workspace://id#mime) before dispatching.
    plain = uri.split("#")[0] if uri.startswith("workspace://") else uri

    if plain.startswith("workspace://"):
        if not user_id:
            raise ValueError("workspace:// file references require authentication")
        manager = await get_manager(user_id, session.session_id)
        ws = parse_workspace_uri(plain)
        try:
            return await (
                manager.read_file(ws.file_ref)
                if ws.is_path
                else manager.read_file_by_id(ws.file_ref)
            )
        except FileNotFoundError:
            raise ValueError(f"File not found: {plain}")
        except Exception as exc:
            raise ValueError(f"Failed to read {plain}: {exc}") from exc

    if is_allowed_local_path(plain, get_sdk_cwd()):
        resolved = os.path.realpath(os.path.expanduser(plain))
        try:
            with open(resolved, "rb") as fh:
                return fh.read()
        except FileNotFoundError:
            raise ValueError(f"File not found: {plain}")
        except Exception as exc:
            raise ValueError(f"Failed to read {plain}: {exc}") from exc

    sandbox = get_current_sandbox()
    if sandbox is not None:
        try:
            remote = resolve_sandbox_path(plain)
        except ValueError as exc:
            raise ValueError(
                f"Path is not allowed (not in workspace, sdk_cwd, or sandbox): {plain}"
            ) from exc
        try:
            return bytes(await sandbox.files.read(remote, format="bytes"))
        except Exception as exc:
            raise ValueError(f"Failed to read from sandbox: {plain}: {exc}") from exc

    raise ValueError(
        f"Path is not allowed (not in workspace, sdk_cwd, or sandbox): {plain}"
    )


async def resolve_file_ref(
    ref: FileRef,
    user_id: str | None,
    session: ChatSession,
) -> str:
    """Resolve a :class:`FileRef` to its text content."""
    raw = await read_file_bytes(ref.uri, user_id, session)
    return _apply_line_range(
        raw.decode("utf-8", errors="replace"), ref.start_line, ref.end_line
    )


async def expand_file_refs_in_string(
    text: str,
    user_id: str | None,
    session: "ChatSession",
    *,
    raise_on_error: bool = False,
) -> str:
    """Expand all ``@@agptfile:...`` tokens in *text*, returning the substituted string.

    Non-reference text is passed through unchanged.

    If *raise_on_error* is ``False`` (default), expansion errors are surfaced
    inline as ``[file-ref error: <message>]`` — useful for display/log contexts
    where partial expansion is acceptable.

    If *raise_on_error* is ``True``, any resolution failure raises
    :class:`FileRefExpansionError` immediately so the caller can block the
    operation and surface a clean error to the model.
    """
    if FILE_REF_PREFIX not in text:
        return text

    result: list[str] = []
    last_end = 0
    total_chars = 0
    for m in _FILE_REF_RE.finditer(text):
        result.append(text[last_end : m.start()])
        start = int(m.group(2)) if m.group(2) else None
        end = int(m.group(3)) if m.group(3) else None
        if (start is not None and start < 1) or (end is not None and end < 1):
            msg = f"line numbers must be >= 1: {m.group(0)}"
            if raise_on_error:
                raise FileRefExpansionError(msg)
            result.append(f"[file-ref error: {msg}]")
            last_end = m.end()
            continue
        if start is not None and end is not None and end < start:
            msg = f"end line must be >= start line: {m.group(0)}"
            if raise_on_error:
                raise FileRefExpansionError(msg)
            result.append(f"[file-ref error: {msg}]")
            last_end = m.end()
            continue
        ref = FileRef(uri=m.group(1), start_line=start, end_line=end)
        try:
            content = await resolve_file_ref(ref, user_id, session)
            if len(content) > _MAX_EXPAND_CHARS:
                content = content[:_MAX_EXPAND_CHARS] + "\n... [truncated]"
            remaining = _MAX_TOTAL_EXPAND_CHARS - total_chars
            if remaining <= 0:
                content = "[file-ref budget exhausted: total expansion limit reached]"
            elif len(content) > remaining:
                content = content[:remaining] + "\n... [total budget exhausted]"
            total_chars += len(content)
            result.append(content)
        except ValueError as exc:
            logger.warning("file-ref expansion failed for %r: %s", m.group(0), exc)
            if raise_on_error:
                raise FileRefExpansionError(str(exc)) from exc
            result.append(f"[file-ref error: {exc}]")
        last_end = m.end()

    result.append(text[last_end:])
    return "".join(result)


async def expand_file_refs_in_args(
    args: dict[str, Any],
    user_id: str | None,
    session: "ChatSession",
) -> dict[str, Any]:
    """Recursively expand ``@@agptfile:...`` references in tool call arguments.

    String values are expanded in-place.  Nested dicts and lists are
    traversed.  Non-string scalars are returned unchanged.

    Raises :class:`FileRefExpansionError` if any reference fails to resolve,
    so the tool is *not* executed with an error string as its input.  The
    caller (the MCP tool wrapper) should convert this into an MCP error
    response that lets the model correct the reference before retrying.
    """
    if not args:
        return args

    async def _expand(value: Any) -> Any:
        if isinstance(value, str):
            return await expand_file_refs_in_string(
                value, user_id, session, raise_on_error=True
            )
        if isinstance(value, dict):
            return {k: await _expand(v) for k, v in value.items()}
        if isinstance(value, list):
            return [await _expand(item) for item in value]
        return value

    return {k: await _expand(v) for k, v in args.items()}
