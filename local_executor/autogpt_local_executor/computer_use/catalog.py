"""Which cua-driver tools the shim relays, which of their arguments need the
person's approval, and what it refuses.

The list was reviewed against one driver release. The shim states it twice:
as the YAML permission policy the driver enforces on every call, including
the calls `run_steps` and `run_script` make inside the driver, and as its
own check on each call before relaying it. See docs/CUA_DRIVER.md.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

# The cua-driver release this catalog was reviewed against. A driver of
# another minor version may add tools or arguments nobody reviewed, so the
# bridge refuses to run one.
REVIEWED_DRIVER_VERSION = "0.38.0"

# Reading state: windows, apps, the accessibility tree, screenshots.
OBSERVE_TOOLS = frozenset(
    {
        "check_permissions",
        "get_cursor_position",
        "get_desktop_state",
        "get_guide",
        "get_screen_size",
        "get_window_state",
        "list_apps",
        "list_windows",
        "verify_state",
        "zoom",
    }
)

# Acting on apps and windows. `run_steps` and `run_script` batch these; the
# driver lets them call only a fixed subset of input and observation tools.
ACT_TOOLS = frozenset(
    {
        "bring_to_front",
        "click",
        "double_click",
        "drag",
        "hotkey",
        "invoke_menu",
        "launch_app",
        "mouse_button_down",
        "mouse_button_up",
        "mouse_drag",
        "move_cursor",
        "press_key",
        "right_click",
        "run_script",
        "run_steps",
        "scroll",
        "set_value",
        "set_window_frame",
        "type_text",
    }
)

# Typed control of Chromium pages. Without a grant, which the shim never
# gives, the driver reaches only browsers it launched with their own
# profile, never the person's signed-in one. `browser_set_input_files`
# hands files from the disk to a page, so every call needs approval.
BROWSER_TOOLS = frozenset(
    {
        "browser_click",
        "browser_dialog",
        "browser_navigate",
        "browser_pointer",
        "browser_prepare",
        "browser_select",
        "browser_set_input_files",
        "browser_type",
        "get_browser_state",
    }
)

# Offered only with --enable-clipboard.
CLIPBOARD_TOOLS = frozenset({"clipboard_read", "clipboard_write"})

# Everything else the driver has stays off: its configuration, updates,
# extensions and session lifecycle; recording and replay; killing
# processes; downloads, which need cua-driver's own host approval; arbitrary
# page JavaScript; and the agent-cursor controls, so the overlay that shows
# the person where the agent is working can't be hidden.


def offered_tools(*, enable_clipboard: bool) -> frozenset[str]:
    tools = OBSERVE_TOOLS | ACT_TOOLS | BROWSER_TOOLS
    if enable_clipboard:
        tools |= CLIPBOARD_TOOLS
    return tools


def policy_document(tools: Iterable[str]) -> str:
    """The driver's YAML permission policy allowing exactly `tools`. JSON is
    YAML, and tool names need no quoting either way."""
    return json.dumps({"allow": {"tools": sorted(set(tools))}}, indent=2) + "\n"


# Arguments that reach the computer's disk, per tool, with what each does
# there. The shim runs a call that uses one only when the platform says the
# person approved that call (`approved_arguments`); the platform asks on a
# card that shows the path.
APPROVAL_ARGUMENTS: dict[str, dict[str, str]] = {
    "get_window_state": {"screenshot_out_file": "writes the screenshot to this file"},
    "get_desktop_state": {"screenshot_out_file": "writes the screenshot to this file"},
    "clipboard_write": {
        "file_path": "puts this file on the clipboard, to paste into an app",
        "image_path": "puts this image file on the clipboard",
    },
    "browser_set_input_files": {"files": "hands these files to the page, which can upload them"},
}


class ArgumentRefused(Exception):
    def __init__(self, tool: str, argument: str, reason: str) -> None:
        super().__init__(f"{tool}: {argument!r} is not accepted here: {reason}")
        self.tool = tool
        self.argument = argument
        self.reason = reason


class ApprovalRequired(Exception):
    """The call uses an argument the person hasn't approved."""

    def __init__(self, tool: str, argument: str, reason: str) -> None:
        super().__init__(
            f"{tool}: {argument!r} {reason}, so the person must approve this call first"
        )
        self.tool = tool
        self.argument = argument
        self.reason = reason


def approval_arguments(tool: str, arguments: Mapping[str, Any]) -> list[str]:
    """The arguments of this call that need the person's approval."""
    return [
        argument
        for argument in APPROVAL_ARGUMENTS.get(tool, {})
        if arguments.get(argument) not in (None, "", [])
    ]


def check_arguments(tool: str, arguments: Mapping[str, Any], approved: Iterable[str] = ()) -> None:
    """Raise ApprovalRequired for a disk argument the person hasn't approved,
    and ArgumentRefused for one this machine never accepts, looking inside
    `run_steps` too, whose steps the shim can see."""
    approved = set(approved)
    if unknown := approved - set(APPROVAL_ARGUMENTS.get(tool, {})):
        raise ArgumentRefused(
            tool, sorted(unknown)[0], "only an argument that needs approval can be approved"
        )
    for argument in approval_arguments(tool, arguments):
        if argument not in approved:
            raise ApprovalRequired(tool, argument, APPROVAL_ARGUMENTS[tool][argument])
    if tool == "check_permissions" and arguments.get("prompt"):
        raise ArgumentRefused(
            tool,
            "prompt",
            "permission prompts are raised by the person at this machine, "
            "with `autogpt-shim doctor`",
        )
    if tool == "clipboard_write":
        values = [name for name in ("text", "file_path", "image_path") if name in arguments]
        if len(values) != 1 or not isinstance(arguments[values[0]], str):
            raise ArgumentRefused(
                tool, "text", "clipboard_write takes exactly one of text, file_path or image_path"
            )
    if tool == "run_steps":
        for index, step in enumerate(arguments.get("steps") or []):
            for step_tool, step_args in _step_calls(step):
                if used := approval_arguments(step_tool, step_args):
                    raise ArgumentRefused(
                        tool,
                        f"steps[{index}].{used[0]}",
                        "a step can't touch the disk; call the tool on its own so the "
                        "person can approve it",
                    )
                try:
                    check_arguments(step_tool, step_args)
                except ArgumentRefused as exc:
                    raise ArgumentRefused(
                        tool, f"steps[{index}].{exc.argument}", exc.reason
                    ) from exc


# The keys of a run_steps step that aren't a tool name in its `{<tool>: args}`
# form.
_STEP_KEYS = frozenset({"args", "expect", "timeout_ms", "tool", "wait_for"})


def _step_calls(step: Any) -> list[tuple[str, Mapping[str, Any]]]:
    """The tool calls in one run_steps step: `{tool, args}` or `{<tool>: args}`."""
    if not isinstance(step, Mapping):
        return []
    if isinstance(step.get("tool"), str):
        args = step.get("args")
        return [(step["tool"], args if isinstance(args, Mapping) else {})]
    return [
        (key, value)
        for key, value in step.items()
        if key not in _STEP_KEYS and isinstance(key, str) and isinstance(value, Mapping)
    ]


def takes_session(schema: Mapping[str, Any]) -> bool:
    properties = schema.get("properties")
    return isinstance(properties, Mapping) and "session" in properties


def without_session(schema: Mapping[str, Any]) -> dict[str, Any]:
    """The schema the platform sees: the shim sets `session` itself."""
    copy = json.loads(json.dumps(schema))
    properties = copy.get("properties")
    if isinstance(properties, dict):
        properties.pop("session", None)
    required = copy.get("required")
    if isinstance(required, list) and "session" in required:
        copy["required"] = [name for name in required if name != "session"]
    return copy


def same_minor(version: str, reviewed: str = REVIEWED_DRIVER_VERSION) -> bool:
    return version.split(".")[:2] == reviewed.split(".")[:2]


__all__ = [
    "ACT_TOOLS",
    "APPROVAL_ARGUMENTS",
    "BROWSER_TOOLS",
    "CLIPBOARD_TOOLS",
    "OBSERVE_TOOLS",
    "REVIEWED_DRIVER_VERSION",
    "ApprovalRequired",
    "ArgumentRefused",
    "approval_arguments",
    "check_arguments",
    "offered_tools",
    "policy_document",
    "same_minor",
    "takes_session",
    "without_session",
]
