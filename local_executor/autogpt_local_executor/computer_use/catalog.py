"""Which cua-driver tools the shim relays, which of their arguments need the
person's approval, and what it refuses.

The list was reviewed against one driver release. The shim states it twice:
as the YAML permission policy the driver enforces on every call, including
the calls `run_steps` and `run_script` make inside the driver, and as its
own check on each call before relaying it. See docs/CUA_DRIVER.md.
"""

from __future__ import annotations

import json
import shutil
import sys
import unicodedata
from collections.abc import Callable, Iterable, Mapping
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

# Everything else the reviewed release has stays off: its configuration,
# updates, extensions and session lifecycle; recording and replay; killing
# processes; downloads, which need cua-driver's own host approval; arbitrary
# page JavaScript; and the agent-cursor controls, so the overlay that shows
# the person where the agent is working can't be hidden. docs/CUA_DRIVER.md
# gives the reason for each. tests/test_cua_driver_contract.py checks every
# tool the driver has is offered or here, so a release that adds one fails
# until someone reviews it.
NOT_OFFERED_TOOLS = frozenset(
    {
        "browser_download",
        "check_for_update",
        # Windows only: a diagnostic for cua-driver's developers.
        "debug_window_info",
        "end_session",
        "escalate_session",
        "get_accessibility_tree",
        "get_agent_cursor_state",
        "get_config",
        "get_recording_state",
        "get_session",
        "get_session_state",
        "health_report",
        "install_extension",
        "install_ffmpeg",
        "kill_app",
        "list_sessions",
        "page",
        "parallel_mouse_drag",
        "parse_visual_regions",
        "replay_trajectory",
        "set_agent_cursor_enabled",
        "set_agent_cursor_motion",
        "set_agent_cursor_theme",
        "set_config",
        "start_recording",
        "start_session",
        "stop_recording",
    }
)


def offered_tools(*, enable_clipboard: bool) -> frozenset[str]:
    tools = OBSERVE_TOOLS | ACT_TOOLS | BROWSER_TOOLS
    if enable_clipboard:
        tools |= CLIPBOARD_TOOLS
    return tools


def policy_document(tools: Iterable[str]) -> str:
    """The driver's YAML permission policy allowing exactly `tools`. JSON is
    YAML, and tool names need no quoting either way."""
    return json.dumps({"allow": {"tools": sorted(set(tools))}}, indent=2) + "\n"


# Arguments that can reach the computer's disk, per tool, with what each
# does there. The shim runs a call that uses one only when the platform says
# the person approved that call (`approved_arguments`); the platform asks on
# a card that shows the path, or the script.
APPROVAL_ARGUMENTS: dict[str, dict[str, str]] = {
    "get_window_state": {"screenshot_out_file": "writes the screenshot to this file"},
    "get_desktop_state": {"screenshot_out_file": "writes the screenshot to this file"},
    "clipboard_write": {
        "file_path": "puts this file on the clipboard, to paste into an app",
        "image_path": "puts this image file on the clipboard",
    },
    "browser_set_input_files": {"files": "hands these files to the page, which can upload them"},
    # A script's calls run inside the driver, where the shim can't check them,
    # and one can pass `screenshot_out_file` to get_window_state: cua-driver
    # 0.38's policy can allow a tool but can't forbid one of its arguments.
    "run_script": {
        "script": "runs this script, which drives apps and can save a screenshot to any file"
    },
    # Each runs whatever it names, out of sight: the driver starts programs
    # in the background with no window of their own, and opens links and
    # files with whatever app handles them. A launch that only names an app
    # (`name`, `bundle_id`, `aumid`) needs no approval; check_arguments
    # holds those to an app's name or id.
    "launch_app": {
        "launch_path": "runs this command line",
        "path": "runs this program",
        "additional_arguments": "starts the app with these command-line arguments",
        "urls": "opens these links or files, each in the app that handles it",
        "webkit_inspector_port": "opens a debugging port into the app",
    },
}

# Approval arguments the driver takes on some systems only (sys.platform,
# with "linux" for every Linux); it takes each other one everywhere. The
# contract tests check them against the driver of the system they run on.
APPROVAL_ARGUMENT_PLATFORMS: dict[tuple[str, str], frozenset[str]] = {
    ("launch_app", "launch_path"): frozenset({"linux", "win32"}),
    ("launch_app", "path"): frozenset({"win32"}),
    ("launch_app", "webkit_inspector_port"): frozenset({"darwin", "win32"}),
}

# The launch_app arguments that name an app rather than a program to run.
_APP_IDENTITIES = ("name", "bundle_id", "aumid")
# A path, a URL scheme or a drive letter: what an app's name or id never has.
_NOT_IN_AN_APP_NAME = frozenset("/\\:")


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


def check_arguments(
    tool: str,
    arguments: Mapping[str, Any],
    approved: Iterable[str] = (),
    *,
    platform: str | None = None,
    on_path: Callable[[str], str | None] = shutil.which,
) -> None:
    """Raise ApprovalRequired for a disk argument the person hasn't approved,
    and ArgumentRefused for one this machine never accepts, looking inside
    `run_steps` too, whose steps the shim can see. `platform` and `on_path`
    are the machine's (sys.platform, shutil.which) unless a test says
    otherwise."""
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
    if tool == "launch_app":
        _check_app_identity(arguments, sys.platform if platform is None else platform, on_path)
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
        # The driver runs `observe` as a get_window_state call of its own.
        observe = arguments.get("observe")
        if isinstance(observe, Mapping) and (
            used := approval_arguments("get_window_state", observe)
        ):
            raise ArgumentRefused(
                tool,
                f"observe.{used[0]}",
                "the observation that ends a batch can't touch the disk; call "
                "get_window_state on its own so the person can approve it",
            )


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


def _check_app_identity(
    arguments: Mapping[str, Any], platform: str, on_path: Callable[[str], str | None]
) -> None:
    """A launch that runs without approval names an app, by its name or id.
    cua-driver also takes a path there (an executable, a `.app` bundle) and,
    on Windows, hands the name to ShellExecuteEx, which opens files and URL
    schemes; those go in `path`, `launch_path` or `urls`, which the person
    approves."""
    for argument in _APP_IDENTITIES:
        value = arguments.get(argument)
        if value is None:
            continue
        if (
            not isinstance(value, str)
            or value.startswith("-")
            or any(char in _NOT_IN_AN_APP_NAME for char in value)
            or any(unicodedata.category(char).startswith("C") for char in value)
        ):
            raise ArgumentRefused(
                "launch_app",
                argument,
                "names an app by its name or id; a path or a link goes in launch_path, "
                "path or urls, which the person approves",
            )
    # On Linux the driver runs `name` as a command line first, split at
    # spaces, and matches it against the installed apps only if no program
    # has its first word. A program's own name ("reboot") would run it, in
    # the background. An app's display name, as list_apps gives it ("Google
    # Chrome", "Firefox"), starts with no program's name (PATH lookups are
    # case-sensitive) and so reaches the match.
    # On Windows the driver launches a bundle_id that is an AUMID as a
    # packaged app and hands any other to ShellExecuteEx as a program name,
    # without looking for an installed app first: `logoff` signs the person
    # out. (A launch by `name` is checked against the installed apps, which
    # takes the driver: ComputerUseHandler._check_windows_launch_name.)
    bundle_id = arguments.get("bundle_id")
    if platform == "win32" and isinstance(bundle_id, str) and not is_aumid(bundle_id):
        raise ArgumentRefused(
            "launch_app",
            "bundle_id",
            "on Windows a bundle_id is a packaged app's AUMID (`Package!App`); name the "
            "app in name, as list_apps shows it, or pass its AUMID as aumid",
        )
    name = arguments.get("name")
    words = name.split() if platform.startswith("linux") and isinstance(name, str) else []
    if words and on_path(words[0]):
        rest = " with the rest as its arguments" if len(words) > 1 else ""
        raise ArgumentRefused(
            "launch_app",
            "name",
            f"the driver would run the program {words[0]!r}{rest}; name the app as "
            "list_apps shows it, or pass a command line as launch_path, which the "
            "person approves",
        )


def is_aumid(value: str) -> bool:
    """Whether cua-driver 0.38 takes `value` for a packaged app's AUMID
    (`{PackageFamilyName}!{ApplicationId}`): one `!`, text on both sides."""
    family, separator, app = value.partition("!")
    return bool(separator and family and app) and "!" not in app


def windows_app_name_matches(name: str, app_names: Iterable[str]) -> bool:
    """Whether cua-driver 0.38's `shell:AppsFolder` lookup on Windows would
    find an installed app for a launch_app `name`: the exact name, ignoring
    case and a trailing `.exe`, or else one that starts with it. Without a
    match the driver hands the name to ShellExecuteEx, which runs a program
    of that name from PATH."""
    query = name.strip().lower().removesuffix(".exe")
    if not query:
        return False
    return any(app.strip().lower().startswith(query) for app in app_names)


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
    "APPROVAL_ARGUMENT_PLATFORMS",
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
