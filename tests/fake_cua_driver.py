"""A stand-in for `cua-driver mcp`, run as `python fake_cua_driver.py mcp`.

It speaks MCP over stdio like the real driver, one request at a time and
dropping every notification as cua-driver 0.38 does, and enforces the YAML
policy named by CUA_DRIVER_POLICY_FILE (an `allow.tools` list) the same way. What
it does is set through environment variables, and it logs what it saw, one
JSON object per line, to FAKE_CUA_LOG:

  FAKE_CUA_VERSION       serverInfo.version (default 0.38.0)
  FAKE_CUA_EXTRA_TOOLS   comma-separated tools listed beyond the catalog
  FAKE_CUA_PERMISSIONS   JSON for check_permissions' structured result
  FAKE_CUA_PERMISSIONS_SLOW_SECONDS  how long check_permissions takes to answer (default
                         at once)
  FAKE_CUA_SCREEN        JSON for get_screen_size's structured result
  FAKE_CUA_CLIPBOARD     file holding the clipboard text
  FAKE_CUA_CLIPBOARD_TYPES  comma-separated clipboard types
  FAKE_CUA_IMAGE_BYTES   size of the image get_window_state returns
  FAKE_CUA_EXIT_AT_START exit before answering initialize
  FAKE_CUA_RPC_ERROR     comma-separated tools answered with a JSON-RPC error
  FAKE_CUA_NEXT_CURSOR   tools/list pages without end: `same` cursor or a `new` one
  FAKE_CUA_SLOW_SECONDS  how long the slow drag (pid -2) takes (default 30)
  FAKE_CUA_INITIALIZE    JSON object merged into initialize's result
  FAKE_CUA_TOOLS_LIST    JSON object merged into tools/list's result
  FAKE_CUA_SCREEN_ERROR  get_screen_size answers with this error
  FAKE_CUA_SCREEN_SLOW_SECONDS  how long get_screen_size takes to answer (default at once)
  FAKE_CUA_IGNORE_TERM   ignore SIGTERM (POSIX), so only a kill stops the fake
  FAKE_CUA_ENDED_SESSIONS  comma-separated session labels refused as ended
  FAKE_CUA_APPS          JSON list of the installed apps list_apps returns: names, or
                         {"name", "bundle_id"} objects
  FAKE_CUA_APPS_FILE     a file holding that list, read for each list_apps, so the
                         installed apps can change while the fake runs
  FAKE_CUA_LIST_APPS_ERROR  list_apps answers with this error
"""

from __future__ import annotations

import base64
import json
import os
import signal
import sys
import threading
import time

CATALOG = [
    "bring_to_front", "browser_click", "browser_dialog", "browser_navigate",
    "browser_pointer", "browser_prepare", "browser_select", "browser_set_input_files",
    "browser_type",
    "check_permissions", "click", "clipboard_read", "clipboard_write",
    "double_click", "drag", "get_browser_state", "get_cursor_position",
    "get_desktop_state", "get_guide", "get_screen_size", "get_window_state",
    "hotkey", "invoke_menu", "launch_app", "list_apps", "list_windows",
    "mouse_button_down", "mouse_button_up", "mouse_drag", "move_cursor",
    "press_key", "right_click", "run_script", "run_steps", "scroll",
    "set_value", "set_window_frame", "type_text", "verify_state", "zoom",
]  # fmt: skip

_write_lock = threading.Lock()


def log(**record: object) -> None:
    path = os.environ.get("FAKE_CUA_LOG")
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


def send(message: dict) -> None:
    with _write_lock:
        sys.stdout.write(json.dumps(message) + "\n")
        sys.stdout.flush()


def allowed_tools() -> set[str] | None:
    path = os.environ.get("CUA_DRIVER_POLICY_FILE")
    if not path:
        return None
    with open(path, encoding="utf-8") as handle:
        return set(json.load(handle)["allow"]["tools"])


def tool_list() -> list[dict]:
    extra = [t for t in os.environ.get("FAKE_CUA_EXTRA_TOOLS", "").split(",") if t]
    allowed = allowed_tools()
    names = [n for n in CATALOG + extra if allowed is None or n in allowed]
    return [
        {
            "name": name,
            "description": f"The {name} tool.",
            "inputSchema": {
                "type": "object",
                "properties": {"session": {"type": "string"}, "pid": {"type": "integer"}},
                "required": ["session"] if name == "zoom" else [],
            },
            "annotations": {"readOnlyHint": name.startswith("get_")},
        }
        for name in names
    ]


def text_result(text: str, structured: dict | None = None, *, error: bool = False) -> dict:
    result: dict = {"content": [{"type": "text", "text": text}]}
    if structured is not None:
        result["structuredContent"] = structured
    if error:
        result["isError"] = True
    return result


def call_tool(request_id: int, name: str, arguments: dict) -> dict:
    log(event="call", tool=name, arguments=arguments)
    allowed = allowed_tools()
    if allowed is not None and name not in allowed:
        return text_result(
            f"Permission denied: user policy: tool '{name}' is not allowed by the YAML policy",
            {"code": "permission_denied"},
            error=True,
        )
    session = arguments.get("session")
    if session in os.environ.get("FAKE_CUA_ENDED_SESSIONS", "").split(","):
        # cua-driver 0.38.0's refusal of a named session it ended for idling.
        message = (
            f"session '{session}' has ended; call start_session with session "
            f"'{session}' to start it again, or use a new session label"
        )
        return text_result(
            message,
            {
                "hint": "That session has ended. Start it again with start_session "
                "(same label), or use a new session label.",
                "next_call": {"arguments": {"session": session}, "tool": "start_session"},
                "refusal": {"code": "session_ended", "message": message},
                "status": "refused",
            },
            error=True,
        )
    if name == "list_apps" and (
        os.environ.get("FAKE_CUA_APPS")
        or os.environ.get("FAKE_CUA_APPS_FILE")
        or os.environ.get("FAKE_CUA_LIST_APPS_ERROR")
    ):
        if os.environ.get("FAKE_CUA_LIST_APPS_ERROR"):
            return text_result(os.environ["FAKE_CUA_LIST_APPS_ERROR"], error=True)
        if os.environ.get("FAKE_CUA_APPS_FILE"):
            with open(os.environ["FAKE_CUA_APPS_FILE"], encoding="utf-8") as handle:
                listed = json.load(handle)
        else:
            listed = json.loads(os.environ["FAKE_CUA_APPS"])
        # Each a name, or {"name", "bundle_id"} (on Linux the desktop-file id).
        installed = [app if isinstance(app, dict) else {"name": app} for app in listed]
        # cua-driver 0.38 on Windows and Linux: installed apps carry their
        # launch_path; a running process no installed app matched has none.
        apps = [
            {
                "pid": 0,
                "name": app["name"],
                "bundle_id": app.get("bundle_id"),
                "running": False,
                "kind": "desktop",
                "launch_path": f"C:\\Apps\\{app['name']}.exe",
            }
            for app in installed
        ] + [
            {
                "pid": 77,
                "name": "logoff",
                "bundle_id": None,
                "running": True,
                "kind": "desktop",
                "launch_path": None,
            }
        ]
        return text_result(f"Found {len(apps)} app(s).", {"apps": apps, "processes": []})
    if name == "get_screen_size":
        if os.environ.get("FAKE_CUA_SCREEN_SLOW_SECONDS"):
            time.sleep(float(os.environ["FAKE_CUA_SCREEN_SLOW_SECONDS"]))
        if os.environ.get("FAKE_CUA_SCREEN_ERROR"):
            return text_result(os.environ["FAKE_CUA_SCREEN_ERROR"], error=True)
        size = json.loads(
            os.environ.get("FAKE_CUA_SCREEN")
            or '{"width": 1920, "height": 1080, "scale_factor": 1.0}'
        )
        return text_result("1920x1080", size)
    if name == "check_permissions":
        if os.environ.get("FAKE_CUA_PERMISSIONS_SLOW_SECONDS"):
            time.sleep(float(os.environ["FAKE_CUA_PERMISSIONS_SLOW_SECONDS"]))
        granted = json.loads(
            os.environ.get("FAKE_CUA_PERMISSIONS")
            or '{"accessibility": true, "screen_recording": true}'
        )
        return text_result("permissions", granted)
    if name == "clipboard_write":
        value = arguments.get("text")
        if value is None:
            value = "file://" + (arguments.get("file_path") or arguments.get("image_path") or "")
        with open(os.environ["FAKE_CUA_CLIPBOARD"], "w", encoding="utf-8") as handle:
            handle.write(value)
        return text_result("written", {"ok": True})
    if name == "clipboard_read":
        try:
            with open(os.environ["FAKE_CUA_CLIPBOARD"], encoding="utf-8") as handle:
                text = handle.read()
        except FileNotFoundError:
            text = None
        types = [t for t in os.environ.get("FAKE_CUA_CLIPBOARD_TYPES", "").split(",") if t]
        return text_result(
            f"clipboard: {text}",
            {"supported": True, "types": types, "text": text, "privacy_sensitive": True},
        )
    if name == "get_window_state":
        size = int(os.environ.get("FAKE_CUA_IMAGE_BYTES", "12"))
        data = base64.b64encode(b"\x89PNG" + b"x" * max(size - 4, 0)).decode()
        return {
            "content": [
                {"type": "text", "text": "window"},
                {"type": "image", "data": data, "mimeType": "image/png"},
                {"type": "resource", "resource": {"uri": "x"}},
            ],
            "structuredContent": {"elements": []},
        }
    if name == "zoom" and arguments.get("pid") == -1:
        return text_result("no such window", {"code": "stale"}, error=True)
    if name == "drag" and arguments.get("pid") == -2:
        # A slow action, done at its end: nothing stops it but the process
        # stopping, as in cua-driver 0.38.
        time.sleep(float(os.environ.get("FAKE_CUA_SLOW_SECONDS", "30")))
        log(event="slow_done")
        return text_result("slow")
    if name == "drag" and arguments.get("pid") == -3:
        log(event="exiting")
        os._exit(3)
    return text_result(json.dumps(arguments), {"arguments": arguments})


def handle(message: dict) -> None:
    method = message.get("method")
    request_id = message.get("id")
    if method == "initialize":
        log(
            event="initialize",
            argv=sys.argv[1:],
            env={
                key: os.environ.get(key)
                for key in (
                    "CUA_DRIVER_POLICY_FILE",
                    "CUA_DRIVER_PERMISSION_MODE",
                    "CUA_DRIVER_DISABLE_UNRESTRICTED",
                    "DO_NOT_TRACK",
                    "CUA_DRIVER_RS_TELEMETRY_ENABLED",
                    "CUA_DRIVER_RS_UPDATE_CHECK",
                    "CUA_DRIVER_DANGEROUSLY_BYPASS_APPROVALS",
                    "CUA_DRIVER_CAPABILITY_MANIFEST_FILE",
                    "DISPLAY",
                    "XAUTHORITY",
                )
            },
            policy=sorted(allowed_tools() or []),
        )
        initialized = {
            "protocolVersion": message["params"]["protocolVersion"],
            "serverInfo": {
                "name": "cua-driver",
                "version": os.environ.get("FAKE_CUA_VERSION", "0.38.0"),
            },
            "capabilities": {"tools": {}},
            "instructions": "Fake cua-driver guidance.",
        }
        initialized.update(json.loads(os.environ.get("FAKE_CUA_INITIALIZE") or "{}"))
        send({"jsonrpc": "2.0", "id": request_id, "result": initialized})
    elif method == "tools/list":
        result: dict = {"tools": tool_list()}
        paging = os.environ.get("FAKE_CUA_NEXT_CURSOR")
        if paging:
            result["nextCursor"] = "page" if paging == "same" else f"page-{request_id}"
        result.update(json.loads(os.environ.get("FAKE_CUA_TOOLS_LIST") or "{}"))
        send({"jsonrpc": "2.0", "id": request_id, "result": result})
    elif method == "tools/call":
        params = message["params"]
        pid = (params.get("arguments") or {}).get("pid")
        if params["name"] in os.environ.get("FAKE_CUA_RPC_ERROR", "").split(","):
            send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32603, "message": f"{params['name']} broke"},
                }
            )
            return
        if params["name"] == "set_window_frame" and pid in (-4, -5):
            # A JSON-RPC error instead of a tool result.
            code, text = (-32602, "width must be positive") if pid == -4 else (-32603, "boom")
            send({"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": text}})
            return

        # One request at a time, read off stdin only once the last is done.
        result = call_tool(request_id, params["name"], params.get("arguments") or {})
        send({"jsonrpc": "2.0", "id": request_id, "result": result})
    elif request_id is None and isinstance(method, str) and method.startswith("notifications/"):
        # cua-driver drops every notification, cancellation included.
        log(event="notification_dropped", method=method)
    elif method == "ping-me":
        # Asks the client a question, as a server may.
        send({"jsonrpc": "2.0", "id": "server-1", "method": "ping"})
        send({"jsonrpc": "2.0", "id": request_id, "result": {}})
    elif method is None:
        log(event="client_reply", message=message)
    elif request_id is not None and method is not None:
        send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"{method} not found"},
            }
        )


def main() -> None:
    if os.environ.get("FAKE_CUA_EXIT_AT_START"):
        sys.stderr.write("fake driver failed to start\n")
        sys.stderr.flush()
        sys.exit(4)
    if os.environ.get("FAKE_CUA_IGNORE_TERM"):
        # POSIX: slow to stop, so the shim waits KILL_GRACE_SECONDS to kill it.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    for line in sys.stdin:
        line = line.strip()
        if line:
            handle(json.loads(line))
    log(event="stdin_closed")


if __name__ == "__main__":
    main()
