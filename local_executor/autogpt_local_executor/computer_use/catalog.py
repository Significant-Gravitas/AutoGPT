"""Which cua-driver tools the shim relays, which of their arguments need the
person's approval, and what it refuses.

The list was reviewed against one driver release. The shim states it twice:
as the YAML permission policy the driver enforces on every call, including
the calls `run_steps` and `run_script` make inside the driver, and as its
own check on each call before relaying it. See docs/CUA_DRIVER.md.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import string
import sys
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, NamedTuple

from ..hidden_characters import has_hidden_character

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

# Acting on apps and windows. `run_steps` and `run_script` call some of
# these inside the driver, which lets them reach only a fixed subset of
# input and observation tools: a batch can't launch an app, a script can.
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
    # On a Mac: a screenshot of the window, with a crosshair at the point.
    "click": {"debug_image_out": "writes a screenshot to this file before it clicks"},
    "clipboard_write": {
        "file_path": "puts this file on the clipboard, to paste into an app",
        "image_path": "puts this image file on the clipboard",
    },
    "browser_set_input_files": {"files": "hands these files to the page, which can upload them"},
    # A script's calls run inside the driver, where the shim can't check them:
    # one can pass `screenshot_out_file` to get_window_state, or start any
    # program with launch_app. cua-driver 0.38's policy can allow a tool but
    # can't forbid one of its arguments.
    "run_script": {
        "script": "runs this script, which drives apps, can start any program and can "
        "save a screenshot to any file"
    },
    # Each runs whatever it names, out of sight: the driver starts programs
    # in the background with no window of their own, and opens links and
    # files with whatever app handles them. A launch that only names an app
    # (`name`, `bundle_id`, `aumid`) needs no approval; check_arguments
    # holds those to an app's name or id. On Linux the shim then sends the
    # driver the matched app's own command as `launch_path` (linux_launch):
    # that one is the shim's, filled in after this check, not the model's.
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
    ("click", "debug_image_out"): frozenset({"darwin"}),
    ("launch_app", "launch_path"): frozenset({"linux", "win32"}),
    ("launch_app", "path"): frozenset({"win32"}),
    ("launch_app", "webkit_inspector_port"): frozenset({"darwin", "win32"}),
}

# Arguments that carry an address the driver's browser loads, per tool. The
# shim relays a call only when each is a web address (is_web_address): a
# browser loads a `file:` address too, and its own pages, and what it loads
# is read back with get_browser_state, outside the shared folders and with
# no approval card. A file or another kind of link is opened with
# launch_app's `urls`, which the person approves.
# tests/test_cua_driver_contract.py checks that every offered tool's
# argument named for an address is here or needs approval.
WEB_ADDRESS_ARGUMENTS: dict[str, tuple[str, ...]] = {"browser_navigate": ("url",)}
# Every such argument. A run_steps step is held to it whatever tool it names,
# as it is to the approval arguments.
_STEP_ADDRESS_ARGUMENTS = tuple(
    dict.fromkeys(
        argument for arguments in WEB_ADDRESS_ARGUMENTS.values() for argument in arguments
    )
)
_NOT_A_WEB_ADDRESS = (
    "only a web address (http:// or https://) or about:blank is opened this way; to open "
    "a file or another kind of link, pass it in launch_app's urls, which the person approves"
)
_ASCII_LOWER = str.maketrans(string.ascii_uppercase, string.ascii_lowercase)

# Every argument that needs approval on some tool. A run_steps step may use
# none of them, whatever tool it names: the driver runs some steps as
# another tool (a `press` with a target, and a `triple_click`, as a click).
_STEP_DISK_ARGUMENTS = tuple(
    dict.fromkeys(argument for arguments in APPROVAL_ARGUMENTS.values() for argument in arguments)
)

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


def is_web_address(value: object) -> bool:
    """Whether `value` is an address the driver's browser may be sent to:
    one that starts with `http://` or `https://`, in any letter case, or
    exactly `about:blank`.

    cua-driver 0.38's browser_navigate hands the string to the browser as
    it is (Page.navigate) once it starts with `http://`, `https://` or
    `about:`, in any ASCII case, with nothing trimmed. A browser does trim:
    it strips spaces and control characters from an address's ends and
    drops tabs and newlines inside it before it reads the scheme. So an
    address that holds a space or a control or other invisible character
    anywhere is refused, not trimmed: what the browser would make of it
    isn't what was checked. `about:` is taken for `about:blank` alone: the
    browser reads any other `about:` address as one of its own pages
    (`about:version` is `chrome://version`), which the driver lets
    through."""
    if not isinstance(value, str):
        return False
    # Every such character, a joiner a script uses included, which a name
    # may hold (hidden_characters): a browser may drop one from a host's
    # name before it looks the host up, and then loads another address than
    # the one checked here. Such a host is written in its `xn--` form.
    if any(char.isspace() or unicodedata.category(char).startswith("C") for char in value):
        return False
    return value == "about:blank" or value[:8].translate(_ASCII_LOWER).startswith(
        ("http://", "https://")
    )


def check_web_addresses(tool: str, arguments: Mapping[str, Any]) -> None:
    """Raise ArgumentRefused for an address the driver's browser would load
    that isn't a web address (is_web_address), in a call's own arguments
    and in each step of a `run_steps`, whatever tool the step names. An
    argument left out is the driver's to refuse."""
    for argument in WEB_ADDRESS_ARGUMENTS.get(tool, ()):
        if argument in arguments and not is_web_address(arguments[argument]):
            raise ArgumentRefused(tool, argument, _NOT_A_WEB_ADDRESS)
    if tool != "run_steps":
        return
    steps = arguments.get("steps")
    for index, step in enumerate(steps if isinstance(steps, list) else []):
        for _step_tool, step_args in _step_calls(step):
            for argument in _STEP_ADDRESS_ARGUMENTS:
                if argument in step_args and not is_web_address(step_args[argument]):
                    raise ArgumentRefused(tool, f"steps[{index}].{argument}", _NOT_A_WEB_ADDRESS)


def approval_arguments(tool: str, arguments: Mapping[str, Any]) -> list[str]:
    """The arguments of this call that need the person's approval."""
    return _given(APPROVAL_ARGUMENTS.get(tool, {}), arguments)


def _given(names: Iterable[str], arguments: Mapping[str, Any]) -> list[str]:
    return [name for name in names if arguments.get(name) not in (None, "", [])]


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
    # Before the approvals: no approval makes an address a web address.
    check_web_addresses(tool, arguments)
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
        # Steps the shim can't look inside are refused: a number stopped this
        # check with a TypeError, which left the call unaudited, and a string
        # or an object was read a character or a key at a time, none a step.
        steps = arguments.get("steps")
        if steps is not None and not isinstance(steps, list):
            raise ArgumentRefused(tool, "steps", "it must be a list of steps, each an object")
        for index, step in enumerate(steps or []):
            if not isinstance(step, Mapping):
                raise ArgumentRefused(
                    tool,
                    f"steps[{index}]",
                    "a step must be an object, {tool, args} or {<tool>: args}",
                )
            for step_tool, step_args in _step_calls(step):
                if used := _given(_STEP_DISK_ARGUMENTS, step_args):
                    raise ArgumentRefused(
                        tool,
                        f"steps[{index}].{used[0]}",
                        "a step can't touch the disk; call the tool on its own so the "
                        "person can approve it",
                    )
                # A launch's name is checked against the installed apps
                # (ComputerUseHandler._check_launch_name) only when
                # launch_app is called on its own. cua-driver 0.38 batches
                # no launch either; this doesn't rely on it. In any case, as
                # the driver reads some step names (`triple_click`).
                if step_tool.lower() == "launch_app":
                    raise ArgumentRefused(
                        tool,
                        f"steps[{index}]",
                        "a launch can't be a step; call launch_app on its own, where its "
                        "name is checked against the installed apps",
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
            # As a folder's name in the folder dialog: no control, bidi
            # control or zero-width character, except a joiner where the
            # name's script uses one (a Persian name, an emoji sequence).
            or has_hidden_character(value)
        ):
            raise ArgumentRefused(
                "launch_app",
                argument,
                "names an app by its name or id; a path or a link goes in launch_path, "
                "path or urls, which the person approves",
            )
    # On Linux the driver runs a `name` as a command line first, split at
    # spaces, and matches it against the installed apps only if no program
    # has its first word: a program's own name ("reboot") would run it, in
    # the background. The shim sends the driver no name to run (a launch by
    # name goes as the matched app's own command, ComputerUseHandler.
    # _linux_launch_by_name), and refuses such a name all the same: it is a
    # command line, not an app's name, and nothing then depends on it never
    # reaching the driver. An app's display name, as list_apps gives it
    # ("Google Chrome", "Firefox"), starts with no program's name (PATH
    # lookups are case-sensitive).
    # On Windows cua-driver 0.38 takes what to launch in this order
    # (platform-windows, launch_app): a `launch_path`, then a `path`, each
    # handed to ShellExecuteEx as it is; then an `aumid`, any string at all,
    # activated as a packaged app with no look at the name or the installed
    # apps (one that names no packaged app fails the call: it isn't run as a
    # program, and the name isn't tried instead); then a `bundle_id` that is
    # an AUMID, the same way; then the `name`, looked up among the installed
    # apps and, when none matches, handed to ShellExecuteEx as a program
    # name, as a `bundle_id` that isn't an AUMID is when there is no name:
    # `logoff` signs the person out.
    # So a bundle_id has to be an AUMID, and so does an aumid: nothing the
    # driver takes without a look at the installed apps goes to it
    # unchecked, and its own handling of a string that is no AUMID isn't
    # what stands between a call and a program. (A launch by `name` is
    # checked against the installed apps, which takes the driver:
    # ComputerUseHandler._check_launch_name.)
    if platform == "win32":
        for argument, instead in (
            ("bundle_id", "or pass its AUMID as aumid"),
            ("aumid", "or leave aumid out"),
        ):
            value = arguments.get(argument)
            if isinstance(value, str) and value and not is_aumid(value):
                raise ArgumentRefused(
                    "launch_app",
                    argument,
                    f"on Windows {'a' if argument == 'bundle_id' else 'an'} {argument} is a "
                    "packaged app's AUMID (`Package!App`); name the app in name, as "
                    f"list_apps shows it, {instead}",
                )
    name = arguments.get("name")
    words = name.split() if platform.startswith("linux") and isinstance(name, str) else []
    if words and on_path(words[0]):
        rest = " with the rest as its arguments" if len(words) > 1 else ""
        raise ArgumentRefused(
            "launch_app",
            "name",
            f"{words[0]!r} is a program, which cua-driver runs as a command line{rest}; "
            "name the app as list_apps shows it, or pass a command line as launch_path, "
            "which the person approves",
        )


def is_aumid(value: str) -> bool:
    """Whether cua-driver 0.38 takes `value` for a packaged app's AUMID
    (`{PackageFamilyName}!{ApplicationId}`): one `!`, text on both sides."""
    family, separator, app = value.partition("!")
    return bool(separator and family and app) and "!" not in app


def windows_launch_uses_name(arguments: Mapping[str, Any]) -> bool:
    """Whether cua-driver 0.38 on Windows may act on a launch_app's `name`.
    It reads each argument only when it is a string, and takes a
    `launch_path`, a `path`, an `aumid` and a `bundle_id` that is an AUMID
    before a name, in that order, never falling back to the name once it
    has one of them. An `aumid` counts here only when it is an AUMID: any
    other is refused before the call is sent (check_arguments), and were
    one sent, the name would still be checked."""
    if not isinstance(arguments.get("name"), str):
        return False
    for argument in ("launch_path", "path"):
        value = arguments.get(argument)
        if isinstance(value, str) and value:
            return False
    for argument in ("aumid", "bundle_id"):
        value = arguments.get(argument)
        if isinstance(value, str) and is_aumid(value):
            return False
    return True


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


class InstalledApp(NamedTuple):
    """An installed app as cua-driver 0.38's list_apps gives it on Windows
    and Linux."""

    name: str
    # On Linux, its desktop-file id.
    bundle_id: str | None
    # On Linux, its launcher's command: the .desktop file's Exec, field
    # codes left out.
    launch_path: str


class LinuxLaunch(NamedTuple):
    """What the shim sends cua-driver to start an installed app on Linux."""

    # The app's launch command, its program an absolute path, and after a
    # program alone a space (linux_launch).
    launch_path: str
    # The command's words from the first that holds `://` on.
    additional_arguments: tuple[str, ...] = ()

    @property
    def command(self) -> str:
        """The whole command line the driver runs, a space between its words."""
        return " ".join([self.launch_path.rstrip(), *self.additional_arguments])


# What cua-driver splits a command line at (Rust's split_whitespace, the
# Unicode White_Space characters). Python's str.split also splits at U+001C
# to U+001F, which the driver keeps inside a word.
_DRIVER_WHITESPACE = re.compile(
    "[\t\n\x0b\x0c\r \x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+"
)


def _driver_words(command: str) -> list[str]:
    return [word for word in _DRIVER_WHITESPACE.split(command) if word]


def _launcher_program_name(launch_path: str) -> str:
    """cua-driver 0.38's exec_basename: the program a launcher's command
    runs, without its folder and in ASCII lowercase, past a leading `env`
    and `NAME=value` words."""
    for word in _driver_words(launch_path):
        if word == "env" or ("=" in word and not word.startswith(("/", "-"))):
            continue
        cleaned = word.strip("\"'")
        parts = [part for part in cleaned.split("/") if part and part != "."]
        # As Rust's Path::file_name: none for a path that ends in `..`.
        return (parts[-1] if parts and parts[-1] != ".." else cleaned).translate(_ASCII_LOWER)
    return ""


def linux_apps(listed: Iterable[InstalledApp]) -> list[InstalledApp]:
    """The installed apps of a list_apps listing in the order cua-driver
    0.38 looks through them (list_installed_apps): by display name,
    lowercased, each once. list_apps itself puts the running apps first,
    each once for every process of theirs. The driver's order among apps of
    one name is no fixed one; here it is the listing's."""
    return sorted(dict.fromkeys(listed), key=lambda app: app.name.lower())


def linux_app_matching(name: str, apps: Sequence[InstalledApp]) -> InstalledApp | None:
    """The installed app cua-driver 0.38 on Linux starts for a launch_app
    `name` that no program has (match_installed_app): the first of `apps`,
    in the driver's order (linux_apps), whose display name, desktop-file id
    or launcher's program is the name, ignoring ASCII case; else the only
    one whose display name contains it."""
    query = name.translate(_ASCII_LOWER)
    if not query:
        return None
    for app in apps:
        if query in (
            app.name.translate(_ASCII_LOWER),
            (app.bundle_id or "").translate(_ASCII_LOWER),
            _launcher_program_name(app.launch_path),
        ):
            return app
    containing = [app for app in apps if query in app.name.translate(_ASCII_LOWER)]
    return containing[0] if len(containing) == 1 else None


def linux_launch(
    launch_path: str,
    *,
    on_path: Callable[[str], str | None] = shutil.which,
    exists: Callable[[str], bool] = os.path.lexists,
) -> LinuxLaunch | None:
    """What to send cua-driver 0.38 to start an installed app whose
    launcher's command is `launch_path`: the same words, the first of them,
    the program, as its absolute path. None when the program can't be named
    that way for certain. `on_path` and `exists` are the machine's
    (shutil.which, os.path.lexists) unless a test says otherwise; both look
    at the disk.

    The driver runs a launch_path split at whitespace, with any
    additional_arguments after it. Only if the program doesn't start does it
    go on: to an installed app the whole launch_path matches, and then to
    xdg-open, when the launch_path holds `://` or names an existing path,
    relative ones from its working folder, the person's home folder. So
    what is sent is never that:

    - The program is an absolute path, so the string names nothing in the
      working folder. It is the path the app's command already gives, or
      the one a bare name has on PATH, which is where the driver would look
      it up. A relative path with a folder, a program found through a
      relative PATH entry, and a path the driver would split at a space are
      not sent.
    - The words from the first that holds `://` on go as
      additional_arguments, which the driver runs but doesn't look at again.
    - A program alone is sent with a space after it. The driver's split
      drops the space, so it runs the same program, and it looks the string
      up as it is, space and all. Without the space the string would be the
      program's own path, which the driver hands to xdg-open when the file
      is there and yet doesn't start: a script whose interpreter is gone,
      or one with no `#!` line, which the driver doesn't run through a
      shell. tests/test_cua_driver_contract.py holds the driver to this.
    - A string that names an existing path all the same is not sent."""
    words = _driver_words(launch_path)
    if not words or "\0" in launch_path:
        return None
    program = words[0]
    if "/" in program and not program.startswith("/"):
        return None
    found = on_path(program)
    if not found or not found.startswith("/") or "://" in found or _driver_words(found) != [found]:
        return None
    arguments = words[1:]
    links = next((at for at, word in enumerate(arguments) if "://" in word), len(arguments))
    command = " ".join([found, *arguments[:links]])
    if not links:
        command += " "
    if exists(command):
        return None
    return LinuxLaunch(command, tuple(arguments[links:]))


def linux_launch_uses_name(arguments: Mapping[str, Any]) -> bool:
    """Whether cua-driver 0.38 on Linux acts on a launch_app's `name`: it
    opens `urls` instead when any of them is a string, and runs a
    `launch_path` that is a string before a name."""
    urls = arguments.get("urls")
    if isinstance(urls, list) and any(isinstance(url, str) for url in urls):
        return False
    return not isinstance(arguments.get("launch_path"), str)


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
