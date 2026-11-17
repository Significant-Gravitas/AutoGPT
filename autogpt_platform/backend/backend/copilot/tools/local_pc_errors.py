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

See `autogpt-local-executor/docs/PROTOCOL.md` (Error codes) and
`autogpt-local-executor/docs/COMPUTER_USE.md` (per-code `details`
examples) for the source of truth.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .local_pc_shim import LocalPCShim


def _shim_allowed_root(shim: "LocalPCShim | None") -> str:
    return (
        shim.allowed_root
        if shim is not None and shim.allowed_root
        else "the workspace root"
    )


def _shim_platform(shim: "LocalPCShim | None") -> str:
    return shim.platform if shim is not None and shim.platform else "this OS"


def _shim_arch(shim: "LocalPCShim | None") -> str:
    return shim.arch if shim is not None and shim.arch else "unknown"


def _details_path(details: dict, message: str) -> str:
    """Pick the most useful path-ish value from the details dict."""
    for key in ("path", "requested_path", "dst", "src"):
        val = details.get(key)
        if val:
            return str(val)
    return message or "<unknown path>"


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
        f"Path `{path}` doesn't exist. Read the directory's listing first "
        "(FILE_LIST, or `ls`/`dir` in a shell) to find the right path."
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
    tool = details.get("tool")
    if tool and details.get("started") is False:
        return (
            f"`{tool}` didn't start: the computer was busy with another call until "
            f"its {timeout_str} limit passed. Nothing ran; call it again."
        )
    if tool:
        return (
            f"`{tool}` didn't finish within its {timeout_str} limit, so the Local PC "
            "executor stopped cua-driver to end it. Anything it already did stays "
            "done: read the window's state before trying again."
        )
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


def _permission_pending(
    code: str, message: str, details: dict, shim: "LocalPCShim | None"
) -> str:
    permission = details.get("permission") or "Accessibility / Screen Recording"
    return (
        f"macOS {permission} permission is not granted (or was revoked). "
        "Ask the user to run `autogpt-shim doctor` on that computer, which "
        "requests it, or to enable it for CuaDriver in System Settings → "
        "Privacy & Security, then retry on the next turn."
    )


def _clipboard_concealed(
    code: str, message: str, details: dict, shim: "LocalPCShim | None"
) -> str:
    reason = details.get("reason")
    if reason in ("writeback_only", "writeback_overwritten"):
        return (
            "On this computer `clipboard_read` returns only text you put on the "
            "clipboard with `clipboard_write` in the last 30 seconds"
            + (
                ", and something replaced it since"
                if reason == "writeback_overwritten"
                else ""
            )
            + ". Read the value from the app's window instead, or ask the user to "
            "paste it into the chat."
        )
    if reason == "changed":
        return "The clipboard changed while it was being read. Read it again."
    marker = details.get("marker")
    marker_str = f" ({marker})" if marker else ""
    return (
        f"A password manager marked the clipboard's contents private{marker_str}, "
        "so the local executor won't read them. Skip this read, or ask the user "
        "to paste the value if it's meant to be shared."
    )


def _feature_not_supported(
    code: str, message: str, details: dict, shim: "LocalPCShim | None"
) -> str:
    tool = details.get("tool") or details.get("op") or details.get("feature")
    subject = f"`{tool}`" if tool else "That operation"
    reason = message or "not offered on this computer"
    return (
        f"{subject} isn't available on this computer ({_shim_platform(shim)}): "
        f"{reason}. Call `local_pc_computer_tools` to see the tools it offers."
    )


def _approval_required(
    code: str, message: str, details: dict, shim: "LocalPCShim | None"
) -> str:
    tool = details.get("tool") or "that tool"
    argument = details.get("argument") or "an argument"
    return (
        f"`{tool}` with `{argument}` reaches the computer's disk, so it runs only "
        "after the user approves that call, and this one wasn't approved. Nothing "
        "ran. Don't reach the file another way."
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
    if details.get("dependency") == "cua-driver":
        return (
            f"The local executor can't run cua-driver: {message or 'not installed'}. "
            "Ask the user to reinstall it with "
            "`pipx install 'autogpt-local-executor[computer-use]'` and run "
            "`autogpt-shim doctor`."
        )
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
        "to add it. Common extras: computer-use (cua-driver), hardware "
        "(pyserial)."
    )


def _recording_not_found(
    code: str, message: str, details: dict, shim: "LocalPCShim | None"
) -> str:
    rec_id = details.get("recording_id") or "<unknown>"
    return (
        f"No recording with id `{rec_id}` exists on the shim. It may have "
        "been secure-erased after skill generation, or never started. Use "
        "`list_recordings` to see what's still buffered, or ask the user to "
        "start a new recording from the AutoGPT recording control."
    )


def _recording_channel_unavailable(
    code: str, message: str, details: dict, shim: "LocalPCShim | None"
) -> str:
    channel = details.get("channel") or "<requested channel>"
    return (
        f"The `{channel}` capture channel isn't available on this machine "
        "(e.g. the browser companion extension isn't connected, or the "
        "desktop accessibility tree isn't exposed for the active app). The "
        "recording can still proceed on the screenshot+action floor — drop "
        f"`{channel}` from the channels list and retry. Steps in that app "
        "will degrade to visual replay, not fail."
    )


def _recording_already_active(
    code: str, message: str, details: dict, shim: "LocalPCShim | None"
) -> str:
    rec_id = details.get("recording_id") or "<active recording>"
    return (
        f"A recording (`{rec_id}`) is already in progress on this machine. "
        "Only one recording can be active at a time. Ask the user to stop it "
        "from the AutoGPT recording control before starting another."
    )


def _consent_required(
    code: str, message: str, details: dict, shim: "LocalPCShim | None"
) -> str:
    return (
        "Recording can't start without a consent token the shim issues "
        "after the user confirms in an OS-native, shim-rendered dialog. The "
        "platform cannot self-assert this consent. Ask the user to confirm "
        "recording in the shim's prompt; the shim then supplies the token "
        "for START_RECORDING."
    )


def _interpretation_unavailable(
    code: str, message: str, details: dict, shim: "LocalPCShim | None"
) -> str:
    route = details.get("interpretation_route") or details.get("route") or "<route>"
    return (
        f"The interpretation route `{route}` needs a local model this "
        "machine doesn't have (no local extractor / OCR / vision model), and "
        "the user declined the cloud fallback. Either ask the user to "
        "install a local model and re-record, or — if they consent — retry "
        "with the `screenshots_to_cloud` route behind the shim's consent "
        "gate."
    )


def _internal_error(
    code: str, message: str, details: dict, shim: "LocalPCShim | None"
) -> str:
    return (
        f"The shim hit an unexpected internal error: `{message or '<no message>'}`. "
        "Try again; if it persists, ask the user to inspect the local audit log "
        "with `autogpt-shim audit tail`."
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
    "PERMISSION_PENDING": _permission_pending,
    "CLIPBOARD_CONCEALED": _clipboard_concealed,
    "FEATURE_NOT_SUPPORTED": _feature_not_supported,
    "APPROVAL_REQUIRED": _approval_required,
    "CAPABILITY_NOT_GRANTED": _capability_not_granted,
    "SHIM_OVERLOADED": _shim_overloaded,
    "AUTH_FAILED": _auth_failed,
    "UNSUPPORTED_ARCH": _unsupported_arch,
    "WRITE_UNCONFIRMED": _write_unconfirmed,
    "FILE_TOO_LARGE": _file_too_large,
    "DEPENDENCY_MISSING": _dependency_missing,
    "RECORDING_NOT_FOUND": _recording_not_found,
    "RECORDING_CHANNEL_UNAVAILABLE": _recording_channel_unavailable,
    "RECORDING_ALREADY_ACTIVE": _recording_already_active,
    "CONSENT_REQUIRED": _consent_required,
    "INTERPRETATION_UNAVAILABLE": _interpretation_unavailable,
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
