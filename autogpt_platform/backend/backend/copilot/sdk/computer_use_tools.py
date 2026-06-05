"""MCP tools that expose the LocalPC shim's computer-use surface to Claude.

These tools are registered alongside the existing file/exec MCP tools
when (a) the active executor is a :class:`LocalPCShim` and (b) the shim's
HELLO advertised the ``computer_use`` capability.

Why expose them as plain MCP tools — even though the Claude Code CLI can
also opt into Anthropic's native ``computer_20251124`` beta tool (via
``ANTHROPIC_BETAS`` in :mod:`env.py`)? Because the native beta tool
covers only screenshot+click+type+key+scroll — it can't list windows,
launch apps, or read the clipboard. Exposing those shim ops as MCP tools
gives Claude leverage the native tool family doesn't have. See
``experimental/local-pc-executor/docs/COMPUTER_USE.md`` for the full
wire surface, and ``PLATFORM_HOOKS.md §10.9`` for the integration
rationale.

Each handler returns a structured MCP result:

* On success — a ``text`` block with a compact JSON payload, suitable
  for Claude to parse on the next turn.
* On a structured shim ERROR (PERMISSION_PENDING, INPUT_OUT_OF_BOUNDS,
  WINDOW_STALE, CLIPBOARD_CONCEALED, FEATURE_NOT_SUPPORTED, ...) — an
  ``isError=True`` block with the human-readable message produced by
  :mod:`local_pc_errors` plus the structured code so the LLM can branch
  on retry vs. give-up without parsing English.

Screenshot returns an Anthropic-format ``image`` content block (base64
payload + mime type) so the model can actually look at the pixels.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from backend.copilot.context import get_current_sandbox
from backend.copilot.tools.local_pc_shim import LocalPCShim, ShimComputerUseError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MCP result helpers
# ---------------------------------------------------------------------------


def _ok(payload: dict | list | str | int | None) -> dict[str, Any]:
    text = payload if isinstance(payload, str) else json.dumps(payload, default=str)
    return {"content": [{"type": "text", "text": text}], "isError": False}


def _err(code: str, message: str, details: dict | None = None) -> dict[str, Any]:
    """Surface a structured error to Claude.

    The text block carries both the LLM-friendly ``message`` (already
    translated by :mod:`local_pc_errors`) and the structured ``code``,
    so Claude can branch on the code without re-parsing the prose.
    """
    body: dict[str, Any] = {"code": code, "error": message}
    if details:
        body["details"] = details
    return {
        "content": [{"type": "text", "text": json.dumps(body, default=str)}],
        "isError": True,
    }


def _get_local_pc_shim() -> LocalPCShim | None:
    sb = get_current_sandbox()
    return sb if isinstance(sb, LocalPCShim) else None


def _require_computer_use() -> tuple[LocalPCShim | None, dict[str, Any] | None]:
    """Resolve the active LocalPCShim, refusing if capability isn't granted."""
    shim = _get_local_pc_shim()
    if shim is None:
        return None, _err(
            "NO_LOCAL_PC_EXECUTOR",
            "No LocalPC shim is connected for this session. The computer-use "
            "tools require the autogpt-local-executor daemon to be running on "
            "the user's machine.",
        )
    if "computer_use" not in (shim.capabilities or []):
        return None, _err(
            "CAPABILITY_NOT_GRANTED",
            "The connected shim did not advertise the `computer_use` capability. "
            "Either the shim's OS denied the necessary permissions "
            "(Accessibility / Screen Recording) or the user disabled the "
            "capability at install time. Skip computer-use for this turn.",
            details={"capabilities": list(shim.capabilities or [])},
        )
    return shim, None


def _handle_shim_error(exc: ShimComputerUseError) -> dict[str, Any]:
    return _err(exc.code, str(exc), exc.details)


# ---------------------------------------------------------------------------
# Handlers — one per wire op family
# ---------------------------------------------------------------------------


async def _h_screenshot(args: dict[str, Any]) -> dict[str, Any]:
    shim, gate = _require_computer_use()
    if shim is None:
        return gate  # type: ignore[return-value]
    try:
        payload = await shim.computer.screenshot(
            monitor=int(args.get("monitor", 0)),
            region=args.get("region"),
            window_id=args.get("window_id"),
            format=str(args.get("format", "png")),
            include_cursor=bool(args.get("include_cursor", False)),
            quality=int(args.get("quality", 75)),
        )
    except ShimComputerUseError as exc:
        return _handle_shim_error(exc)
    except ValueError as exc:
        return _err("INVALID_ARGUMENT", str(exc))

    image_b64 = payload.get("image_base64") or ""
    mime = payload.get("mime_type") or (
        "image/png" if str(args.get("format", "png")).lower() == "png" else "image/jpeg"
    )
    meta = {
        "width": payload.get("width"),
        "height": payload.get("height"),
        "monitor": payload.get("monitor"),
        "region": payload.get("region"),
        "display_scale": payload.get("display_scale"),
        "logical_size": payload.get("logical_size"),
        "meta": payload.get("meta"),
    }
    content: list[dict[str, Any]] = [
        {"type": "text", "text": json.dumps(meta, default=str)},
    ]
    if image_b64:
        content.append(
            {
                "type": "image",
                "data": image_b64,
                "mimeType": mime,
            }
        )
    return {"content": content, "isError": False}


def _coord(args: dict[str, Any]) -> list[int]:
    raw = args.get("coordinate")
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError("coordinate must be a [x, y] pair")
    return [int(raw[0]), int(raw[1])]


async def _h_click(args: dict[str, Any]) -> dict[str, Any]:
    shim, gate = _require_computer_use()
    if shim is None:
        return gate  # type: ignore[return-value]
    try:
        await shim.computer.click(
            _coord(args),
            button=str(args.get("button", "left")),
            modifiers=args.get("modifiers"),
        )
    except ValueError as exc:
        return _err("INVALID_ARGUMENT", str(exc))
    except ShimComputerUseError as exc:
        return _handle_shim_error(exc)
    return _ok({"ok": True})


async def _h_type(args: dict[str, Any]) -> dict[str, Any]:
    shim, gate = _require_computer_use()
    if shim is None:
        return gate  # type: ignore[return-value]
    text = args.get("text", "")
    if not isinstance(text, str):
        return _err("INVALID_ARGUMENT", "text must be a string")
    try:
        await shim.computer.type(
            text,
            paste=bool(args.get("paste", False)),
            preserve_clipboard=bool(args.get("preserve_clipboard", False)),
        )
    except ShimComputerUseError as exc:
        return _handle_shim_error(exc)
    return _ok({"ok": True})


async def _h_key(args: dict[str, Any]) -> dict[str, Any]:
    shim, gate = _require_computer_use()
    if shim is None:
        return gate  # type: ignore[return-value]
    key = args.get("key", "")
    if not isinstance(key, str) or not key:
        return _err("INVALID_ARGUMENT", "key is required (e.g. 'enter', 'ctrl+s')")
    try:
        await shim.computer.key(key)
    except ShimComputerUseError as exc:
        return _handle_shim_error(exc)
    return _ok({"ok": True})


async def _h_scroll(args: dict[str, Any]) -> dict[str, Any]:
    shim, gate = _require_computer_use()
    if shim is None:
        return gate  # type: ignore[return-value]
    try:
        await shim.computer.scroll(
            _coord(args),
            direction=str(args.get("direction", "down")),
            scroll_amount=int(args.get("scroll_amount", 1)),
            modifiers=args.get("modifiers"),
        )
    except ValueError as exc:
        return _err("INVALID_ARGUMENT", str(exc))
    except ShimComputerUseError as exc:
        return _handle_shim_error(exc)
    return _ok({"ok": True})


async def _h_cursor_position(args: dict[str, Any]) -> dict[str, Any]:
    shim, gate = _require_computer_use()
    if shim is None:
        return gate  # type: ignore[return-value]
    try:
        payload = await shim.computer.cursor_position()
    except ShimComputerUseError as exc:
        return _handle_shim_error(exc)
    return _ok(payload)


async def _h_list_windows(args: dict[str, Any]) -> dict[str, Any]:
    shim, gate = _require_computer_use()
    if shim is None:
        return gate  # type: ignore[return-value]
    try:
        windows = await shim.computer.list_windows(
            app_bundle_id=args.get("app_bundle_id"),
            include_minimized=bool(args.get("include_minimized", False)),
            include_offscreen=bool(args.get("include_offscreen", False)),
        )
    except ShimComputerUseError as exc:
        return _handle_shim_error(exc)
    return _ok({"windows": windows})


async def _h_focus_window(args: dict[str, Any]) -> dict[str, Any]:
    shim, gate = _require_computer_use()
    if shim is None:
        return gate  # type: ignore[return-value]
    window_id = args.get("window_id", "")
    if not isinstance(window_id, str) or not window_id:
        return _err("INVALID_ARGUMENT", "window_id is required")
    try:
        await shim.computer.focus_window(
            window_id, raise_=bool(args.get("raise", True))
        )
    except ShimComputerUseError as exc:
        return _handle_shim_error(exc)
    return _ok({"ok": True})


async def _h_list_apps(args: dict[str, Any]) -> dict[str, Any]:
    shim, gate = _require_computer_use()
    if shim is None:
        return gate  # type: ignore[return-value]
    try:
        apps = await shim.computer.list_apps(
            include_background=bool(args.get("include_background", False))
        )
    except ShimComputerUseError as exc:
        return _handle_shim_error(exc)
    return _ok({"apps": apps})


async def _h_launch_app(args: dict[str, Any]) -> dict[str, Any]:
    shim, gate = _require_computer_use()
    if shim is None:
        return gate  # type: ignore[return-value]
    try:
        payload = await shim.computer.launch_app(
            bundle_id=args.get("bundle_id"),
            executable_path=args.get("executable_path"),
            args=args.get("args") or [],
            activate=bool(args.get("activate", True)),
        )
    except ValueError as exc:
        return _err("INVALID_ARGUMENT", str(exc))
    except ShimComputerUseError as exc:
        return _handle_shim_error(exc)
    return _ok(payload)


async def _h_clipboard_read(args: dict[str, Any]) -> dict[str, Any]:
    shim, gate = _require_computer_use()
    if shim is None:
        return gate  # type: ignore[return-value]
    try:
        content = await shim.computer.clipboard_read(
            format=str(args.get("format", "text"))
        )
    except ShimComputerUseError as exc:
        return _handle_shim_error(exc)
    return _ok({"content": content})


async def _h_clipboard_write(args: dict[str, Any]) -> dict[str, Any]:
    shim, gate = _require_computer_use()
    if shim is None:
        return gate  # type: ignore[return-value]
    content = args.get("content", "")
    if not isinstance(content, str):
        return _err("INVALID_ARGUMENT", "content must be a string")
    try:
        await shim.computer.clipboard_write(
            content, format=str(args.get("format", "text"))
        )
    except ShimComputerUseError as exc:
        return _handle_shim_error(exc)
    return _ok({"ok": True})


async def _h_permissions_check(args: dict[str, Any]) -> dict[str, Any]:
    shim, gate = _require_computer_use()
    if shim is None:
        return gate  # type: ignore[return-value]
    try:
        result = await shim.computer.permissions_check(args.get("permissions"))
    except ShimComputerUseError as exc:
        return _handle_shim_error(exc)
    return _ok(result)


# ---------------------------------------------------------------------------
# Tool descriptors — registered by tool_adapter.create_copilot_mcp_server
# when ``use_local_pc_computer`` is True.
#
# Schemas follow the same minimal-JSON-Schema style as E2B_FILE_TOOLS.
# ---------------------------------------------------------------------------


_COORDINATE_SCHEMA = {
    "type": "array",
    "items": {"type": "integer"},
    "minItems": 2,
    "maxItems": 2,
    "description": "Display-global, top-left-origin [x, y] in unscaled virtual pixels.",
}


LOCAL_PC_COMPUTER_TOOLS: list[
    tuple[str, str, dict[str, Any], Callable[[dict[str, Any]], Any]]
] = [
    (
        "local_pc_screenshot",
        "Capture a screenshot of the user's screen via the LocalPC shim. "
        "Returns an image content block plus metadata (resolution, scale, "
        "monitor index). Use `region: [x1, y1, x2, y2]` to crop, or "
        "`window_id` from `local_pc_list_windows` to capture one window.",
        {
            "type": "object",
            "properties": {
                "monitor": {
                    "type": "integer",
                    "description": "Monitor index (0 = primary). Default: 0.",
                },
                "region": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": "[x1, y1, x2, y2] crop in display-global pixels.",
                },
                "window_id": {
                    "type": "string",
                    "description": "Opaque window id from local_pc_list_windows.",
                },
                "format": {
                    "type": "string",
                    "enum": ["png", "jpeg"],
                    "description": "Image format. Default: png.",
                },
                "include_cursor": {
                    "type": "boolean",
                    "description": "Render the cursor in the capture. Default: false.",
                },
                "quality": {
                    "type": "integer",
                    "description": "JPEG quality 1-100 (ignored for png). Default: 75.",
                },
            },
        },
        _h_screenshot,
    ),
    (
        "local_pc_click",
        "Click at a display-global coordinate via the LocalPC shim. "
        "`button` defaults to 'left'; pass 'right' or 'middle' for the other "
        "buttons. `modifiers` is an optional subset of "
        "['shift', 'ctrl', 'alt', 'super'].",
        {
            "type": "object",
            "properties": {
                "coordinate": _COORDINATE_SCHEMA,
                "button": {
                    "type": "string",
                    "enum": ["left", "right", "middle"],
                    "description": "Mouse button. Default: 'left'.",
                },
                "modifiers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Modifier keys held during the click.",
                },
            },
            "required": ["coordinate"],
        },
        _h_click,
    ),
    (
        "local_pc_type",
        "Type text via the LocalPC shim. Set `paste: true` for strings "
        ">= 200 characters to avoid IME / autocomplete issues — the shim "
        "stashes the text on the OS clipboard and sends Cmd/Ctrl+V. "
        "If you also pass `preserve_clipboard: true`, the shim restores "
        "the prior clipboard contents unless another app overwrote them "
        "during the paste.",
        {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to type."},
                "paste": {
                    "type": "boolean",
                    "description": "Use OS paste for long strings. Default: false.",
                },
                "preserve_clipboard": {
                    "type": "boolean",
                    "description": "Restore prior clipboard after paste. Default: false.",
                },
            },
            "required": ["text"],
        },
        _h_type,
    ),
    (
        "local_pc_key",
        "Press a key or chord (e.g. 'enter', 'tab', 'esc', 'ctrl+s', "
        "'cmd+shift+t') via the LocalPC shim. Modifiers: shift, ctrl, "
        "alt, super (alias: meta).",
        {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Key name or chord, e.g. 'enter' or 'ctrl+s'.",
                },
            },
            "required": ["key"],
        },
        _h_key,
    ),
    (
        "local_pc_scroll",
        "Scroll at a display-global coordinate. `direction` is one of "
        "'up'/'down'/'left'/'right'; `scroll_amount` is in OS-native ticks.",
        {
            "type": "object",
            "properties": {
                "coordinate": _COORDINATE_SCHEMA,
                "direction": {
                    "type": "string",
                    "enum": ["up", "down", "left", "right"],
                    "description": "Scroll direction. Default: 'down'.",
                },
                "scroll_amount": {
                    "type": "integer",
                    "description": "Scroll ticks. Default: 1.",
                },
                "modifiers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Modifier keys held during the scroll.",
                },
            },
            "required": ["coordinate"],
        },
        _h_scroll,
    ),
    (
        "local_pc_cursor_position",
        "Return the current cursor position via the LocalPC shim. "
        "Use to verify a mouse_move landed without paying for a screenshot.",
        {"type": "object", "properties": {}},
        _h_cursor_position,
    ),
    (
        "local_pc_list_windows",
        "List visible windows via the LocalPC shim. Returns objects with "
        "`window_id` (opaque, pass to local_pc_focus_window / "
        "local_pc_screenshot), title, app name, pid, bounds, monitor, and "
        "focus/minimized/fullscreen state. Filter by `app_bundle_id` to "
        "narrow the result. Window ids are invalidated on shim reconnect.",
        {
            "type": "object",
            "properties": {
                "app_bundle_id": {
                    "type": "string",
                    "description": "Filter to one application by bundle id.",
                },
                "include_minimized": {
                    "type": "boolean",
                    "description": "Include minimized windows. Default: false.",
                },
                "include_offscreen": {
                    "type": "boolean",
                    "description": "Include offscreen windows. Default: false.",
                },
            },
        },
        _h_list_windows,
    ),
    (
        "local_pc_focus_window",
        "Bring a window to the foreground via the LocalPC shim. "
        "`window_id` is an opaque id from local_pc_list_windows; if the id "
        "has been recycled the call returns WINDOW_STALE and you should "
        "re-list windows.",
        {
            "type": "object",
            "properties": {
                "window_id": {
                    "type": "string",
                    "description": "Opaque window id from local_pc_list_windows.",
                },
                "raise": {
                    "type": "boolean",
                    "description": "Raise (bring to front) vs. focus-only. Default: true.",
                },
            },
            "required": ["window_id"],
        },
        _h_focus_window,
    ),
    (
        "local_pc_list_apps",
        "List currently-running applications via the LocalPC shim "
        "(not installed apps). Returns objects with pid, name, bundle id, "
        "executable path, frontmost flag, and window count. Useful for "
        "'is Slack open?' before launching it.",
        {
            "type": "object",
            "properties": {
                "include_background": {
                    "type": "boolean",
                    "description": "Include background-only processes. Default: false.",
                },
            },
        },
        _h_list_apps,
    ),
    (
        "local_pc_launch_app",
        "Launch an application via the LocalPC shim. Pass `bundle_id` on "
        "macOS (preferred) or `executable_path` on Windows/Linux. "
        "Returns the launched pid.",
        {
            "type": "object",
            "properties": {
                "bundle_id": {
                    "type": "string",
                    "description": "macOS bundle id, e.g. 'com.apple.Safari'.",
                },
                "executable_path": {
                    "type": "string",
                    "description": "Full path to the executable (Windows/Linux).",
                },
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Command-line arguments to pass at launch.",
                },
                "activate": {
                    "type": "boolean",
                    "description": "Bring the app to front after launch. Default: true.",
                },
            },
        },
        _h_launch_app,
    ),
    (
        "local_pc_clipboard_read",
        "Read the current clipboard via the LocalPC shim. Only available "
        "when the shim was started with --enable-clipboard. Returns "
        "CLIPBOARD_CONCEALED for password-manager / writeback-expired "
        "contents — that is not a bug, that's the security boundary.",
        {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "enum": ["text"],
                    "description": "Clipboard format (only 'text' in v1).",
                },
            },
        },
        _h_clipboard_read,
    ),
    (
        "local_pc_clipboard_write",
        "Write text to the system clipboard via the LocalPC shim. Only "
        "available when the shim was started with --enable-clipboard. "
        "Note: this overwrites whatever the user had on their clipboard.",
        {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Text to write."},
                "format": {
                    "type": "string",
                    "enum": ["text"],
                    "description": "Clipboard format (only 'text' in v1).",
                },
            },
            "required": ["content"],
        },
        _h_clipboard_write,
    ),
    (
        "local_pc_permissions_check",
        "Probe OS-level permissions on the user's machine via the LocalPC "
        "shim (no UI prompt). Returns a dict mapping each requested "
        "permission to 'granted'/'denied'/'unknown'/'not_applicable'. Use "
        "to pre-flight before issuing a doomed INPUT_ACTION on macOS.",
        {
            "type": "object",
            "properties": {
                "permissions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Subset of: screen_recording, accessibility, "
                        "input_monitoring. Default: all three."
                    ),
                },
            },
        },
        _h_permissions_check,
    ),
]


LOCAL_PC_COMPUTER_TOOL_NAMES: list[str] = [name for name, *_ in LOCAL_PC_COMPUTER_TOOLS]
