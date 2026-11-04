"""Otto's computer use on a Local PC: two MCP tools over the shim's cua-driver.

The shim relays cua-driver's own tools (window-targeted, in the background,
accessibility first; autogpt-local-executor ``docs/CUA_DRIVER.md``).
``local_pc_computer_tools`` returns the driver's operating guidance and the
tools this computer offers, with full input schemas for the ones named;
``local_pc_computer`` calls one. Registering each driver tool as its own MCP
tool would put ~40 large schemas in every turn; Otto reads the ones it needs.

Every call re-checks that a shim is connected and offers computer use, and
that the user approved computer use in this session for this machine, driver
release and tool set. A reconnect in between fails closed.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from typing import Any

from backend.api.features.local_executor.consent import is_computer_use_approved
from backend.copilot.context import (
    get_current_sandbox,
    get_execution_context,
    is_local_pc_executor,
)
from backend.copilot.sdk.computer_use_policy import (
    LOCAL_PC_COMPUTER_TOOL,
    LOCAL_PC_COMPUTER_TOOLS_TOOL,
    guard_scope,
)
from backend.copilot.tools.local_pc_shim import (
    LocalPCShim,
    ShimComputerUseError,
    ShimConnectionGuard,
)

logger = logging.getLogger(__name__)

_TOOL_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}")
_SUMMARY_CHARS = 280
# A structured-content string this long that a text block already shows is
# sent once, not twice.
_DUPLICATE_STRING_CHARS = 200
_MAX_TIMEOUT_SECONDS = 600

Gate = tuple[LocalPCShim, ShimConnectionGuard]


async def _h_computer_tools(args: dict[str, Any]) -> dict[str, Any]:
    gate = await _require_computer_use()
    if isinstance(gate, dict):
        return gate
    shim, guard = gate
    try:
        catalog = await shim.computer.tools(_guard=guard)
    except ShimComputerUseError as exc:
        return _err(exc.code, str(exc), exc.details)
    requested = args.get("tools") or []
    if not isinstance(requested, list) or not all(
        isinstance(t, str) for t in requested
    ):
        return _err("INVALID_ARGUMENT", "tools must be a list of tool names")
    return _ok(_catalog_view(catalog, requested))


async def _h_computer(args: dict[str, Any]) -> dict[str, Any]:
    tool, arguments, timeout = (
        args.get("tool"),
        args.get("arguments"),
        args.get("timeout_seconds"),
    )
    if not isinstance(tool, str) or _TOOL_NAME.fullmatch(tool) is None:
        return _err(
            "INVALID_ARGUMENT", "tool must be a tool name from local_pc_computer_tools"
        )
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        return _err("INVALID_ARGUMENT", "arguments must be an object")
    if timeout is not None and (
        not isinstance(timeout, int) or not 1 <= timeout <= _MAX_TIMEOUT_SECONDS
    ):
        return _err("INVALID_ARGUMENT", "timeout_seconds must be 1-600")
    gate = await _require_computer_use()
    if isinstance(gate, dict):
        return gate
    shim, guard = gate
    if tool not in guard.computer_use_tools:
        return _err(
            "FEATURE_NOT_SUPPORTED",
            f"`{tool}` isn't offered on this computer. Call "
            f"`{LOCAL_PC_COMPUTER_TOOLS_TOOL}` to see the tools it offers.",
            {"tool": tool},
        )
    try:
        result = await shim.computer.call(
            tool, arguments, timeout_seconds=timeout, _guard=guard
        )
    except ShimComputerUseError as exc:
        return _err(exc.code, str(exc), exc.details)
    return _mcp_result(result)


async def _require_computer_use() -> Gate | dict[str, Any]:
    """The shim and its connection guard, or the MCP error to return."""
    sandbox = get_current_sandbox()
    shim = sandbox if is_local_pc_executor(sandbox) else None
    if not isinstance(shim, LocalPCShim):
        return _err(
            "NO_LOCAL_PC_EXECUTOR",
            "No Local PC executor is connected for this session. Computer use "
            "needs the autogpt-local-executor daemon running on the user's computer.",
        )
    guard = shim.capture_connection_guard()
    scope = guard_scope(guard, shim.capability_set)
    if scope is None:
        return _err(
            "CAPABILITY_NOT_GRANTED",
            "The connected Local PC executor doesn't offer computer use: it wasn't "
            "turned on there, or cua-driver couldn't start. Skip computer use for "
            "this turn.",
            {"capabilities": list(shim.capabilities or [])},
        )
    user_id, session = get_execution_context()
    if (
        not user_id
        or session is None
        or not await is_computer_use_approved(
            session.session_id,
            user_id,
            machine_id=scope.machine_id,
            driver=scope.driver,
            tools=scope.tools,
        )
    ):
        return _err(
            "COMPUTER_USE_CONSENT_REQUIRED",
            "Computer control is not approved for this chat session. Stop and "
            "ask the user to approve Local PC computer use in the AutoGPT UI.",
        )
    if not shim.connection_guard_matches(guard):
        return _err(
            "COMPUTER_USE_CONNECTION_CHANGED",
            "The Local PC executor reconnected while computer access was being "
            "checked. Retry after the user reviews the reconnected computer.",
        )
    return shim, guard


def _catalog_view(catalog: dict[str, Any], requested: list[str]) -> dict[str, Any]:
    tools = [tool for tool in catalog.get("tools") or [] if isinstance(tool, dict)]
    by_name = {tool.get("name"): tool for tool in tools}
    view: dict[str, Any] = {
        "driver": catalog.get("driver"),
        "instructions": catalog.get("instructions") or "",
        "tools": [
            {"name": tool.get("name"), "summary": _summary(tool.get("description"))}
            for tool in tools
        ],
        "how_to_call": (
            f"Call `{LOCAL_PC_COMPUTER_TOOL}` with `tool` and `arguments`. Pass "
            "`tools` here to get the input schemas of the tools you will use."
        ),
    }
    if requested:
        view["schemas"] = {
            name: {
                "description": by_name[name].get("description"),
                "input_schema": by_name[name].get("input_schema"),
            }
            for name in requested
            if name in by_name
        }
        unknown = [name for name in requested if name not in by_name]
        if unknown:
            view["unknown_tools"] = unknown
    return view


def _summary(description: Any) -> str:
    text = str(description or "").strip()
    first = text.split("\n\n", 1)[0].replace("\n", " ")
    return first if len(first) <= _SUMMARY_CHARS else first[: _SUMMARY_CHARS - 1] + "…"


def _mcp_result(result: dict[str, Any]) -> dict[str, Any]:
    """The driver's answer as MCP content: its text and images as they came,
    then its structured content without strings the text already shows."""
    content: list[dict[str, Any]] = []
    texts: list[str] = []
    for block in result.get("content") or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            texts.append(block["text"])
            content.append({"type": "text", "text": block["text"]})
        elif block.get("type") == "image" and isinstance(block.get("data"), str):
            content.append(
                {
                    "type": "image",
                    "data": block["data"],
                    "mimeType": str(block.get("mime_type") or "image/png"),
                }
            )
    structured = result.get("structured_content")
    if isinstance(structured, dict) and structured:
        shown = _without_duplicates(structured, texts)
        content.append({"type": "text", "text": json.dumps(shown, default=str)})
    if not content:
        content.append({"type": "text", "text": "(no output)"})
    return {"content": content, "isError": bool(result.get("is_error"))}


def _without_duplicates(structured: dict[str, Any], texts: list[str]) -> dict[str, Any]:
    return {
        key: (
            "(shown in the text above)"
            if isinstance(value, str)
            and len(value) >= _DUPLICATE_STRING_CHARS
            and any(value.strip() in text for text in texts)
            else value
        )
        for key, value in structured.items()
    }


def _ok(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": json.dumps(payload, default=str)}],
        "isError": False,
    }


def _err(code: str, message: str, details: dict | None = None) -> dict[str, Any]:
    """A structured error: the LLM-friendly message and the code to branch on."""
    body: dict[str, Any] = {"code": code, "error": message}
    if details:
        body["details"] = details
    return {
        "content": [{"type": "text", "text": json.dumps(body, default=str)}],
        "isError": True,
    }


LOCAL_PC_COMPUTER_TOOLS: list[
    tuple[str, str, dict[str, Any], Callable[[dict[str, Any]], Any]]
] = [
    (
        LOCAL_PC_COMPUTER_TOOLS_TOOL,
        "Computer use on the user's Local PC, through cua-driver. Returns the "
        "driver's operating guidance (read it before acting: it works window by "
        "window, in the background, by accessibility element), the tools this "
        "computer offers, and the input schemas of the tools you name in `tools`.",
        {
            "type": "object",
            "properties": {
                "tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tool names whose input schemas to return.",
                },
            },
        },
        _h_computer_tools,
    ),
    (
        LOCAL_PC_COMPUTER_TOOL,
        "Call one cua-driver tool on the user's Local PC, such as list_windows, "
        "get_window_state, run_steps, click or type_text, with the arguments its "
        f"schema from `{LOCAL_PC_COMPUTER_TOOLS_TOOL}` describes. Returns what the "
        "driver said, screenshots included. An error result is the driver's own "
        "verdict: read it and adjust. Never repeat an action that may have "
        "happened without reading the window's state first.",
        {
            "type": "object",
            "properties": {
                "tool": {"type": "string", "description": "The cua-driver tool."},
                "arguments": {
                    "type": "object",
                    "description": "The tool's arguments; leave out `session`.",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": _MAX_TIMEOUT_SECONDS,
                    "description": "How long the call may run (default 120).",
                },
            },
            "required": ["tool"],
        },
        _h_computer,
    ),
]


LOCAL_PC_COMPUTER_TOOL_NAMES: list[str] = [name for name, *_ in LOCAL_PC_COMPUTER_TOOLS]
