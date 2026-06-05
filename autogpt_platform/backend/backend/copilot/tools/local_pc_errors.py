"""
Translation layer for autogpt-local-executor shim error codes.

The shim's wire `ERROR` envelope is structured:

    {
      "type": "ERROR",
      "id": "...",
      "ts": ...,
      "payload": {
        "code": "PATH_OUTSIDE_ALLOWED_ROOT",
        "message": "Path /etc/passwd is outside allowed root ...",
        "fatal": false,
        "details": {...}     # optional, shape varies per code
      }
    }

Those `code` values are opaque enums when surfaced to an LLM. This module
maps each code to an actionable English string that uses the shim's actual
HELLO metadata (allowed_root, platform, arch, ...) plus the per-code
`details` dict so the message tells the LLM *what to do next*, not just
*what went wrong*.

See `experimental/local-pc-executor/docs/PROTOCOL.md` (Error codes) and
`experimental/local-pc-executor/docs/COMPUTER_USE.md` (per-code `details`
examples) for the source of truth.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .local_pc_shim import LocalPCShim


def _shim_allowed_root(shim: "LocalPCShim | None") -> str:
    return getattr(shim, "allowed_root", "") or "the workspace root"


def _shim_platform(shim: "LocalPCShim | None") -> str:
    return getattr(shim, "platform", "") or "this OS"


def _shim_arch(shim: "LocalPCShim | None") -> str:
    return getattr(shim, "arch", "") or "unknown"


def _details_path(details: dict, message: str) -> str:
    """Pick the most useful path-ish value from the details dict."""
    for key in ("path", "requested_path", "dst", "src"):
        val = details.get(key)
        if val:
            return str(val)
    return message or "<unknown path>"


def _details_coordinate(details: dict) -> str:
    coord = details.get("requested_coordinate") or details.get("coordinate")
    if isinstance(coord, (list, tuple)) and len(coord) == 2:
        return f"({coord[0]}, {coord[1]})"
    return str(coord) if coord is not None else "<unknown coordinate>"


def _details_display_rects(details: dict) -> str:
    displays = details.get("displays") or details.get("display_rects") or []
    if not displays:
        return "<no displays reported>"
    parts: list[str] = []
    for d in displays:
        if not isinstance(d, dict):
            continue
        origin = d.get("origin") or [0, 0]
        size = d.get("size") or [0, 0]
        idx = d.get("index")
        prefix = f"#{idx} " if idx is not None else ""
        parts.append(
            f"{prefix}origin=({origin[0]}, {origin[1]}) size=({size[0]}x{size[1]})"
        )
    return "; ".join(parts) if parts else "<malformed display rects>"


def _path_outside_allowed_root(
    code: str, message: str, details: dict, shim: "LocalPCShim | None"
) -> str:
    return (
        f"The path you tried (`{_details_path(details, message)}`) is outside "
        f"the user's local workspace. Write inside `{_shim_allowed_root(shim)}` "
        f"instead. The shim only has access to paths under that root."
    )


def _path_reserved_name(
    code: str, message: str, details: dict, shim: "LocalPCShim | None"
) -> str:
    return (
        "The path uses a Windows reserved name (CON, PRN, NUL, COM1-9, "
        "LPT1-9). Pick a different basename — these are blocked at the OS "
        "level on Windows even if the rest of the path is valid."
    )


def _path_invalid_chars(
    code: str, message: str, details: dict, shim: "LocalPCShim | None"
) -> str:
    return (
        f"The path contains characters illegal on {_shim_platform(shim)} "
        '(e.g. `<`, `>`, `:`, `"`, `|`, `?`, `*` on Windows; null bytes '
        "anywhere). Sanitize the filename and retry."
    )


def _path_not_found(
    code: str, message: str, details: dict, shim: "LocalPCShim | None"
) -> str:
    path = _details_path(details, message)
    return (
        f"Path `{path}` doesn't exist. Use `local_pc_list_windows` or list "
        "the directory first (FILE_LIST, or `ls`/`dir` in a shell) to find "
        "the right path."
    )


def _path_not_empty(
    code: str, message: str, details: dict, shim: "LocalPCShim | None"
) -> str:
    path = _details_path(details, message)
    return (
        f"Directory `{path}` is not empty — pass `recursive=true` to delete "
        "it and its contents, or empty it first."
    )


def _path_exists(
    code: str, message: str, details: dict, shim: "LocalPCShim | None"
) -> str:
    path = _details_path(details, message)
    return f"Destination `{path}` already exists — pass `overwrite=true` to replace it."


def _command_timeout(
    code: str, message: str, details: dict, shim: "LocalPCShim | None"
) -> str:
    timeout = details.get("timeout_seconds")
    timeout_str = f"{timeout}s" if timeout is not None else "configured"
    return (
        f"The command exceeded its {timeout_str} timeout. Either increase the "
        "timeout or break the work into smaller commands. The shim killed "
        "the process; partial output may be in stdout/stderr."
    )


def _shell_not_available(
    code: str, message: str, details: dict, shim: "LocalPCShim | None"
) -> str:
    requested = details.get("requested_shell") or details.get("shell") or "<unknown>"
    return (
        f"Shell `{requested}` isn't installed on {_shim_platform(shim)}. "
        'Try `shell: "auto"` to use the OS default, or use `argv: [...]` '
        "to skip shell entirely. Note bash isn't always present on Windows."
    )


def _window_stale(
    code: str, message: str, details: dict, shim: "LocalPCShim | None"
) -> str:
    window_id = details.get("window_id") or "<unknown>"
    return (
        f"Window id `{window_id}` no longer exists or has been recycled. "
        "Call `local_pc_list_windows` again to get fresh ids and retry."
    )


def _permission_pending(
    code: str, message: str, details: dict, shim: "LocalPCShim | None"
) -> str:
    permission = details.get("permission") or "Accessibility / Screen Recording"
    return (
        f"macOS {permission} permission is not granted (or was revoked). "
        "The shim daemon will re-launch shortly to pick up the new consent. "
        "Ask the user to open System Settings → Privacy & Security → "
        f"{permission} and enable autogpt-shim, then retry on the next turn."
    )


def _clipboard_concealed(
    code: str, message: str, details: dict, shim: "LocalPCShim | None"
) -> str:
    reason = details.get("reason")
    reason_str = f" (reason: {reason})" if reason else ""
    return (
        f"The current clipboard contents are marked confidential{reason_str} "
        "(likely from a password manager, or the writeback window expired). "
        "The shim refuses to read concealed clipboard data. Skip this read "
        "or ask the user to copy the value again from a different source."
    )


def _input_out_of_bounds(
    code: str, message: str, details: dict, shim: "LocalPCShim | None"
) -> str:
    return (
        f"Coordinate `{_details_coordinate(details)}` is outside the "
        f"connected display bounds [{_details_display_rects(details)}]. "
        "Re-screenshot to get fresh display dimensions and clamp "
        "coordinates to inside one of the listed rectangles."
    )


def _feature_not_supported(
    code: str, message: str, details: dict, shim: "LocalPCShim | None"
) -> str:
    op = details.get("op") or details.get("feature") or "<requested op>"
    reason = message or "feature not advertised in HELLO.computer_use_features"
    return (
        f"Operation `{op}` is not available on this shim's platform "
        f"({_shim_platform(shim)}). Reason: {reason}. The shim's "
        "`computer_use_features` in HELLO lists what IS available."
    )


def _capability_not_granted(
    code: str, message: str, details: dict, shim: "LocalPCShim | None"
) -> str:
    capability = details.get("capability") or "<requested capability>"
    return (
        f"The user didn't grant the `{capability}` capability when they "
        "installed the shim. They'd need to re-run `autogpt-shim auth` with "
        "the additional scope to enable it. Skip this for now."
    )


def _shim_overloaded(
    code: str, message: str, details: dict, shim: "LocalPCShim | None"
) -> str:
    max_concurrent = details.get("max_concurrent") or "the"
    return (
        f"The shim is at its concurrency cap ({max_concurrent} concurrent "
        "ops). Retry in a moment, or batch smaller ops to reduce concurrency."
    )


def _auth_failed(
    code: str, message: str, details: dict, shim: "LocalPCShim | None"
) -> str:
    return (
        "The shim's OAuth token is invalid or expired. The user needs to "
        "run `autogpt-shim auth` again. The session can't proceed until "
        "they do."
    )


def _unsupported_arch(
    code: str, message: str, details: dict, shim: "LocalPCShim | None"
) -> str:
    return (
        f"The shim is running on an architecture the platform doesn't "
        f"support (`{_shim_arch(shim)}`). x86_64 and arm64 are supported."
    )


def _write_unconfirmed(
    code: str, message: str, details: dict, shim: "LocalPCShim | None"
) -> str:
    return (
        "The FILE_WRITE was sent but the connection dropped before the "
        "shim acknowledged. The file may or may not have been written. "
        "Verify with a FILE_STAT and re-write if needed."
    )


def _file_too_large(
    code: str, message: str, details: dict, shim: "LocalPCShim | None"
) -> str:
    max_bytes = details.get("max_file_size_bytes")
    attempted = details.get("attempted_bytes")
    op = details.get("op") or "read/write"
    if max_bytes is not None:
        max_str = f"{max_bytes} bytes"
    else:
        max_str = "the shim's configured limit"
    attempted_str = f" (attempted {attempted} bytes)" if attempted is not None else ""
    return (
        f"File exceeded the shim's max size ({max_str}){attempted_str}. "
        f"Chunk the {op} by reading/writing smaller sections with "
        "offset+length, or compress on the user's side first."
    )


def _dependency_missing(
    code: str, message: str, details: dict, shim: "LocalPCShim | None"
) -> str:
    dep = details.get("dep")
    extra = details.get("extra")
    op = details.get("op")
    if not dep:
        # Fallback: passthrough message — the shim's wording is usually
        # already specific (e.g. "Pillow is required for screenshots").
        return (
            f"The shim host is missing a runtime dependency for this op: "
            f"{message or '<no detail>'}. Ask the user to install the "
            "needed extra with `pipx install autogpt-local-executor[<extra>]`."
        )
    op_clause = f" for {op}" if op else ""
    extra_clause = f"[{extra}]" if extra else "[<extra>]"
    return (
        f"The shim host needs the '{dep}' Python package installed{op_clause}. "
        f"Ask the user to run `pipx install autogpt-local-executor{extra_clause}` "
        "to add it. Common deps: pyautogui (input), Pillow (screenshot), "
        "pyperclip (clipboard), pyserial (hardware)."
    )


def _internal_error(
    code: str, message: str, details: dict, shim: "LocalPCShim | None"
) -> str:
    return (
        f"The shim hit an unexpected internal error: `{message or '<no message>'}`. "
        "Try again; if it persists, ask the user to check "
        "`~/Library/Logs/autogpt-local-executor/audit.log` (or the equivalent "
        "on their OS) for details."
    )


_Translator = Callable[[str, str, dict, "LocalPCShim | None"], str]


_TRANSLATIONS: dict[str, _Translator] = {
    "PATH_OUTSIDE_ALLOWED_ROOT": _path_outside_allowed_root,
    "PATH_RESERVED_NAME": _path_reserved_name,
    "PATH_INVALID_CHARS": _path_invalid_chars,
    "PATH_NOT_FOUND": _path_not_found,
    "PATH_NOT_EMPTY": _path_not_empty,
    "PATH_EXISTS": _path_exists,
    "COMMAND_TIMEOUT": _command_timeout,
    "SHELL_NOT_AVAILABLE": _shell_not_available,
    "WINDOW_STALE": _window_stale,
    "PERMISSION_PENDING": _permission_pending,
    "CLIPBOARD_CONCEALED": _clipboard_concealed,
    "INPUT_OUT_OF_BOUNDS": _input_out_of_bounds,
    "FEATURE_NOT_SUPPORTED": _feature_not_supported,
    "CAPABILITY_NOT_GRANTED": _capability_not_granted,
    "SHIM_OVERLOADED": _shim_overloaded,
    "AUTH_FAILED": _auth_failed,
    "UNSUPPORTED_ARCH": _unsupported_arch,
    "WRITE_UNCONFIRMED": _write_unconfirmed,
    "FILE_TOO_LARGE": _file_too_large,
    "DEPENDENCY_MISSING": _dependency_missing,
    "INTERNAL_ERROR": _internal_error,
}


def translate_shim_error(
    code: str,
    message: str,
    details: dict | None,
    shim: "LocalPCShim | None",
) -> str:
    """
    Convert a shim wire ERROR envelope into an LLM-friendly recovery hint.

    Never raises — unknown codes fall through to a passthrough string so
    the LLM at least sees the raw enum and the shim-supplied message.
    """
    safe_details: dict = details if isinstance(details, dict) else {}
    safe_message: str = message or ""
    safe_code: str = code or "INTERNAL_ERROR"
    translator = _TRANSLATIONS.get(safe_code)
    if translator is None:
        return f"shim error {safe_code}: {safe_message}"
    try:
        return translator(safe_code, safe_message, safe_details, shim)
    except Exception as exc:
        # Defensive: a malformed details shape should never escape as an
        # uncaught exception — it would just be replaced with another
        # opaque error string in the upstream handler.
        return f"shim error {safe_code}: {safe_message} (translator fallback: {exc})"
