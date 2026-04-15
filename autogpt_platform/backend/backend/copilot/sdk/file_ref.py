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
    get_workspace_manager,
    is_allowed_local_path,
    resolve_sandbox_path,
)
from backend.copilot.model import ChatSession
from backend.util.file import parse_workspace_uri
from backend.util.file_content_parser import (
    BINARY_FORMATS,
    MIME_TO_FORMAT,
    PARSE_EXCEPTIONS,
    infer_format_from_uri,
    parse_file_content,
)
from backend.util.type import MediaFileType


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
# Maximum raw byte size for bare ref structured parsing (10 MB).
_MAX_BARE_REF_BYTES = 10_000_000


@dataclass
class FileRef:
    uri: str
    start_line: int | None  # 1-indexed, inclusive
    end_line: int | None  # 1-indexed, inclusive


# ---------------------------------------------------------------------------
# Public API  (top-down: main functions first, helpers below)
# ---------------------------------------------------------------------------


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
        manager = await get_workspace_manager(user_id, session.session_id)
        ws = parse_workspace_uri(plain)
        try:
            data = await (
                manager.read_file(ws.file_ref)
                if ws.is_path
                else manager.read_file_by_id(ws.file_ref)
            )
        except FileNotFoundError:
            raise ValueError(f"File not found: {plain}")
        except (PermissionError, OSError) as exc:
            raise ValueError(f"Failed to read {plain}: {exc}") from exc
        except (AttributeError, TypeError, RuntimeError) as exc:
            # AttributeError/TypeError: workspace manager returned an
            # unexpected type or interface; RuntimeError: async runtime issues.
            logger.warning("Unexpected error reading %s: %s", plain, exc)
            raise ValueError(f"Failed to read {plain}: {exc}") from exc
        # NOTE: Workspace API does not support pre-read size checks;
        # the full file is loaded before the size guard below.
        if len(data) > _MAX_BARE_REF_BYTES:
            raise ValueError(
                f"File too large ({len(data)} bytes, limit {_MAX_BARE_REF_BYTES})"
            )
        return data

    if is_allowed_local_path(plain, get_sdk_cwd()):
        resolved = os.path.realpath(os.path.expanduser(plain))
        try:
            # Read with a one-byte overshoot to detect files that exceed the limit
            # without a separate os.path.getsize call (avoids TOCTOU race).
            with open(resolved, "rb") as fh:
                data = fh.read(_MAX_BARE_REF_BYTES + 1)
            if len(data) > _MAX_BARE_REF_BYTES:
                raise ValueError(
                    f"File too large (>{_MAX_BARE_REF_BYTES} bytes, "
                    f"limit {_MAX_BARE_REF_BYTES})"
                )
            return data
        except FileNotFoundError:
            raise ValueError(f"File not found: {plain}")
        except OSError as exc:
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
            data = bytes(await sandbox.files.read(remote, format="bytes"))
        except (FileNotFoundError, OSError, UnicodeDecodeError) as exc:
            raise ValueError(f"Failed to read from sandbox: {plain}: {exc}") from exc
        except Exception as exc:
            # E2B SDK raises SandboxException subclasses (NotFoundException,
            # TimeoutException, NotEnoughSpaceException, etc.) which don't
            # inherit from standard exceptions.  Import lazily to avoid a
            # hard dependency on e2b at module level.
            try:
                from e2b.exceptions import SandboxException  # noqa: PLC0415

                if isinstance(exc, SandboxException):
                    raise ValueError(
                        f"Failed to read from sandbox: {plain}: {exc}"
                    ) from exc
            except ImportError:
                pass
            # Re-raise unexpected exceptions (TypeError, AttributeError, etc.)
            # so they surface as real bugs rather than being silently masked.
            raise
        # NOTE: E2B sandbox API does not support pre-read size checks;
        # the full file is loaded before the size guard below.
        if len(data) > _MAX_BARE_REF_BYTES:
            raise ValueError(
                f"File too large ({len(data)} bytes, limit {_MAX_BARE_REF_BYTES})"
            )
        return data

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
    return _apply_line_range(_to_str(raw), ref.start_line, ref.end_line)


async def expand_file_refs_in_string(
    text: str,
    user_id: str | None,
    session: ChatSession,
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
            # remaining == 0 means the budget was exactly exhausted by the
            # previous ref.  The elif below (len > remaining) won't catch
            # this since 0 > 0 is false, so we need the <= 0 check.
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
    session: ChatSession,
    *,
    input_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Recursively expand ``@@agptfile:...`` references in tool call arguments.

    String values are expanded in-place.  Nested dicts and lists are
    traversed.  Non-string scalars are returned unchanged.

    **Bare references** (the entire argument value is a single
    ``@@agptfile:...`` token with no surrounding text) are resolved and then
    parsed according to the file's extension or MIME type.  See
    :mod:`backend.util.file_content_parser` for the full list of supported
    formats (JSON, JSONL, CSV, TSV, YAML, TOML, Parquet, Excel).

    When *input_schema* is provided and the target property has
    ``"type": "string"``, structured parsing is skipped — the raw file content
    is returned as a plain string so blocks receive the original text.

    If the format is unrecognised or parsing fails, the content is returned as
    a plain string (the fallback).

    **Embedded references** (``@@agptfile:`` mixed with other text) always
    produce a plain string — structured parsing only applies to bare refs.

    Raises :class:`FileRefExpansionError` if any reference fails to resolve,
    so the tool is *not* executed with an error string as its input.  The
    caller (the MCP tool wrapper) should convert this into an MCP error
    response that lets the model correct the reference before retrying.
    """
    if not args:
        return args

    properties = (input_schema or {}).get("properties", {})

    async def _expand(
        value: Any,
        *,
        prop_schema: dict[str, Any] | None = None,
    ) -> Any:
        """Recursively expand a single argument value.

        Strings are checked for ``@@agptfile:`` references and expanded
        (bare refs get structured parsing; embedded refs get inline
        substitution).  Dicts and lists are traversed recursively,
        threading the corresponding sub-schema from *prop_schema* so
        that nested fields also receive correct type-aware expansion.
        Non-string scalars pass through unchanged.
        """
        if isinstance(value, str):
            ref = parse_file_ref(value)
            if ref is not None:
                # MediaFileType fields: return the raw URI immediately —
                # no file reading, no format inference, no content parsing.
                if _is_media_file_field(prop_schema):
                    return ref.uri

                fmt = infer_format_from_uri(ref.uri)
                # Workspace URIs by ID (workspace://abc123) have no extension.
                # When the MIME fragment is also missing, fall back to the
                # workspace file manager's metadata for format detection.
                if fmt is None and ref.uri.startswith("workspace://"):
                    fmt = await _infer_format_from_workspace(ref.uri, user_id, session)
                return await _expand_bare_ref(ref, fmt, user_id, session, prop_schema)

            # Not a bare ref — do normal inline expansion.
            return await expand_file_refs_in_string(
                value, user_id, session, raise_on_error=True
            )
        if isinstance(value, dict):
            # When the schema says this is an object but doesn't define
            # inner properties, skip expansion — the caller (e.g.
            # RunBlockTool) will expand with the actual nested schema.
            if (
                prop_schema is not None
                and prop_schema.get("type") == "object"
                and "properties" not in prop_schema
            ):
                return value
            nested_props = (prop_schema or {}).get("properties", {})
            return {
                k: await _expand(v, prop_schema=nested_props.get(k))
                for k, v in value.items()
            }
        if isinstance(value, list):
            items_schema = (prop_schema or {}).get("items")
            return [await _expand(item, prop_schema=items_schema) for item in value]
        return value

    return {k: await _expand(v, prop_schema=properties.get(k)) for k, v in args.items()}


# ---------------------------------------------------------------------------
# Private helpers  (used by the public functions above)
# ---------------------------------------------------------------------------


def _apply_line_range(text: str, start: int | None, end: int | None) -> str:
    """Slice *text* to the requested 1-indexed line range (inclusive).

    When the requested range extends beyond the file, a note is appended
    so the LLM knows it received the entire remaining content.
    """
    if start is None and end is None:
        return text
    lines = text.splitlines(keepends=True)
    total = len(lines)
    s = (start - 1) if start is not None else 0
    e = end if end is not None else total
    selected = list(itertools.islice(lines, s, e))
    result = "".join(selected)
    if end is not None and end > total:
        result += f"\n[Note: file has only {total} lines]\n"
    return result


def _to_str(content: str | bytes) -> str:
    """Decode *content* to a string if it is bytes, otherwise return as-is."""
    if isinstance(content, str):
        return content
    return content.decode("utf-8", errors="replace")


def _check_content_size(content: str | bytes) -> None:
    """Raise :class:`ValueError` if *content* exceeds the byte limit.

    Raises ``ValueError`` (not ``FileRefExpansionError``) so that the caller
    (``_expand_bare_ref``) can unify all resolution errors into a single
    ``except ValueError`` → ``FileRefExpansionError`` handler, keeping the
    error-flow consistent with ``read_file_bytes`` and ``resolve_file_ref``.

    For ``bytes``, the length is the byte count directly.  For ``str``,
    we encode to UTF-8 first because multi-byte characters (e.g. emoji)
    mean the byte size can be up to 4x the character count.
    """
    if isinstance(content, bytes):
        size = len(content)
    else:
        char_len = len(content)
        # Fast lower bound: UTF-8 byte count >= char count.
        # If char count already exceeds the limit, reject immediately
        # without allocating an encoded copy.
        if char_len > _MAX_BARE_REF_BYTES:
            size = char_len  # real byte size is even larger
        # Fast upper bound: each char is at most 4 UTF-8 bytes.
        # If worst-case is still under the limit, skip encoding entirely.
        elif char_len * 4 <= _MAX_BARE_REF_BYTES:
            return
        else:
            # Edge case: char count is under limit but multibyte chars
            # might push byte count over. Encode to get exact size.
            size = len(content.encode("utf-8"))
    if size > _MAX_BARE_REF_BYTES:
        raise ValueError(
            f"File too large for structured parsing "
            f"({size} bytes, limit {_MAX_BARE_REF_BYTES})"
        )


async def _infer_format_from_workspace(
    uri: str,
    user_id: str | None,
    session: ChatSession,
) -> str | None:
    """Look up workspace file metadata to infer the format.

    Workspace URIs by ID (``workspace://abc123``) have no file extension.
    When the MIME fragment is also absent, we query the workspace file
    manager for the file's stored MIME type and original filename.
    """
    if not user_id:
        return None
    try:
        ws = parse_workspace_uri(uri)
        manager = await get_workspace_manager(user_id, session.session_id)
        info = await (
            manager.get_file_info(ws.file_ref)
            if not ws.is_path
            else manager.get_file_info_by_path(ws.file_ref)
        )
        if info is None:
            return None
        # Try MIME type first, then filename extension.
        mime = (info.mime_type or "").split(";", 1)[0].strip().lower()
        return MIME_TO_FORMAT.get(mime) or infer_format_from_uri(info.name)
    except (
        ValueError,
        FileNotFoundError,
        OSError,
        PermissionError,
        AttributeError,
        TypeError,
    ):
        # Expected failures: bad URI, missing file, permission denied, or
        # workspace manager returning unexpected types.  Propagate anything
        # else (e.g. programming errors) so they don't get silently swallowed.
        logger.debug("workspace metadata lookup failed for %s", uri, exc_info=True)
        return None


def _is_media_file_field(prop_schema: dict[str, Any] | None) -> bool:
    """Return True if *prop_schema* describes a MediaFileType field (format: file)."""
    if prop_schema is None:
        return False
    return (
        prop_schema.get("type") == "string"
        and prop_schema.get("format") == MediaFileType.string_format
    )


async def _expand_bare_ref(
    ref: FileRef,
    fmt: str | None,
    user_id: str | None,
    session: ChatSession,
    prop_schema: dict[str, Any] | None,
) -> Any:
    """Resolve and parse a bare ``@@agptfile:`` reference.

    This is the structured-parsing path: the file is read, optionally parsed
    according to *fmt*, and adapted to the target *prop_schema*.

    Raises :class:`FileRefExpansionError` on resolution or parsing failure.

    Note: MediaFileType fields (format: "file") are handled earlier in
    ``_expand`` to avoid unnecessary format inference and file I/O.
    """
    try:
        if fmt is not None and fmt in BINARY_FORMATS:
            # Binary formats need raw bytes, not UTF-8 text.
            # Line ranges are meaningless for binary formats (parquet/xlsx)
            # — ignore them and parse full bytes.  Warn so the caller/model
            # knows the range was silently dropped.
            if ref.start_line is not None or ref.end_line is not None:
                logger.warning(
                    "Line range [%s-%s] ignored for binary format %s (%s); "
                    "binary formats are always parsed in full.",
                    ref.start_line,
                    ref.end_line,
                    fmt,
                    ref.uri,
                )
            content: str | bytes = await read_file_bytes(ref.uri, user_id, session)
        else:
            content = await resolve_file_ref(ref, user_id, session)
    except ValueError as exc:
        raise FileRefExpansionError(str(exc)) from exc

    # For known formats this rejects files >10 MB before parsing.
    # For unknown formats _MAX_EXPAND_CHARS (200K chars) below is stricter,
    # but this check still guards the parsing path which has no char limit.
    # _check_content_size raises ValueError, which we unify here just like
    # resolution errors above.
    try:
        _check_content_size(content)
    except ValueError as exc:
        raise FileRefExpansionError(str(exc)) from exc

    # When the schema declares this parameter as "string",
    # return raw file content — don't parse into a structured
    # type that would need json.dumps() serialisation.
    expect_string = (prop_schema or {}).get("type") == "string"
    if expect_string:
        if isinstance(content, bytes):
            raise FileRefExpansionError(
                f"Cannot use {fmt} file as text input: "
                f"binary formats (parquet, xlsx) must be passed "
                f"to a block that accepts structured data (list/object), "
                f"not a string-typed parameter."
            )
        return content

    if fmt is not None:
        # Use strict mode for binary formats so we surface the
        # actual error (e.g. missing pyarrow/openpyxl, corrupt
        # file) instead of silently returning garbled bytes.
        strict = fmt in BINARY_FORMATS
        try:
            parsed = parse_file_content(content, fmt, strict=strict)
        except PARSE_EXCEPTIONS as exc:
            raise FileRefExpansionError(f"Failed to parse {fmt} file: {exc}") from exc
        # Normalize bytes fallback to str so tools never
        # receive raw bytes when parsing fails.
        if isinstance(parsed, bytes):
            parsed = _to_str(parsed)
        return _adapt_to_schema(parsed, prop_schema)

    # Unknown format — return as plain string, but apply
    # the same per-ref character limit used by inline refs
    # to prevent injecting unexpectedly large content.
    text = _to_str(content)
    if len(text) > _MAX_EXPAND_CHARS:
        text = text[:_MAX_EXPAND_CHARS] + "\n... [truncated]"
    return text


def _adapt_to_schema(parsed: Any, prop_schema: dict[str, Any] | None) -> Any:
    """Adapt a parsed file value to better fit the target schema type.

    When the parser returns a natural type (e.g. dict from YAML, list from CSV)
    that doesn't match the block's expected type, this function converts it to
    a more useful representation instead of relying on pydantic's generic
    coercion (which can produce awkward results like flattened dicts → lists).

    Returns *parsed* unchanged when no adaptation is needed.
    """
    if prop_schema is None:
        return parsed

    target_type = prop_schema.get("type")

    # Dict → array: delegate to helper.
    if isinstance(parsed, dict) and target_type == "array":
        return _adapt_dict_to_array(parsed, prop_schema)

    # List → object: delegate to helper (raises for non-tabular lists).
    if isinstance(parsed, list) and target_type == "object":
        return _adapt_list_to_object(parsed)

    # Tabular list → Any (no type): convert to list of dicts.
    # Blocks like FindInDictionaryBlock have `input: Any` which produces
    # a schema with no "type" key.  Tabular [[header],[rows]] is unusable
    # for key lookup, but [{col: val}, ...] works with FindInDict's
    # list-of-dicts branch (line 195-199 in data_manipulation.py).
    if isinstance(parsed, list) and target_type is None and _is_tabular(parsed):
        return _tabular_to_list_of_dicts(parsed)

    return parsed


def _adapt_dict_to_array(parsed: dict, prop_schema: dict[str, Any]) -> Any:
    """Adapt a parsed dict to an array-typed field.

    Extracts list-valued entries when the target item type is ``array``,
    passes through unchanged when item type is ``string`` (lets pydantic error),
    or wraps in ``[parsed]`` as a fallback.
    """
    items_type = (prop_schema.get("items") or {}).get("type")
    if items_type == "array":
        # Target is List[List[Any]] — extract list-typed values from the
        # dict as inner lists.  E.g. YAML {"fruits": [{...},...]}} with
        # ConcatenateLists (List[List[Any]]) → [[{...},...]].
        list_values = [v for v in parsed.values() if isinstance(v, list)]
        if list_values:
            return list_values
    if items_type == "string":
        # Target is List[str] — wrapping a dict would give [dict]
        # which can't coerce to strings.  Return unchanged and let
        # pydantic surface a clear validation error.
        return parsed
    # Fallback: wrap in a single-element list so the block gets [dict]
    # instead of pydantic flattening keys/values into a flat list.
    return [parsed]


def _adapt_list_to_object(parsed: list) -> Any:
    """Adapt a parsed list to an object-typed field.

    Converts tabular lists to column-dicts; raises for non-tabular lists.
    """
    if _is_tabular(parsed):
        return _tabular_to_column_dict(parsed)
    # Non-tabular list (e.g. a plain Python list from a YAML file) cannot
    # be meaningfully coerced to an object.  Raise explicitly so callers
    # get a clear error rather than pydantic silently wrapping the list.
    raise FileRefExpansionError(
        "Cannot adapt a non-tabular list to an object-typed field. "
        "Expected a tabular structure ([[header], [row1], ...]) or a dict."
    )


def _is_tabular(parsed: Any) -> bool:
    """Check if parsed data is in tabular format: [[header], [row1], ...].

    Uses isinstance checks because this is a structural type guard on
    opaque parser output (Any), not duck typing.  A Protocol wouldn't
    help here — we need to verify exact list-of-lists shape.
    """
    if not isinstance(parsed, list) or len(parsed) < 2:
        return False
    header = parsed[0]
    if not isinstance(header, list) or not header:
        return False
    if not all(isinstance(h, str) for h in header):
        return False
    return all(isinstance(row, list) for row in parsed[1:])


def _tabular_to_list_of_dicts(parsed: list) -> list[dict[str, Any]]:
    """Convert [[header], [row1], ...] → [{header[0]: row[0], ...}, ...].

    Ragged rows (fewer columns than the header) get None for missing values.
    Extra values beyond the header length are silently dropped.
    """
    header = parsed[0]
    return [
        dict(itertools.zip_longest(header, row[: len(header)], fillvalue=None))
        for row in parsed[1:]
    ]


def _tabular_to_column_dict(parsed: list) -> dict[str, list]:
    """Convert [[header], [row1], ...] → {"col1": [val1, ...], ...}.

    Ragged rows (fewer columns than the header) get None for missing values,
    ensuring all columns have equal length.
    """
    header = parsed[0]
    return {
        col: [row[i] if i < len(row) else None for row in parsed[1:]]
        for i, col in enumerate(header)
    }
