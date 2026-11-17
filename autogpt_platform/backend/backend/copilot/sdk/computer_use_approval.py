"""Computer-use calls that reach the user's disk wait for their approval.

The shim names each tool's approval arguments in its tool list
(``approval_arguments``): ``screenshot_out_file``, clipboard ``file_path`` and
``image_path``, ``browser_set_input_files.files``, ``run_script.script``,
whose calls inside the driver can save a screenshot to a file, and what
``launch_app`` runs out of sight (a command line, a program's path, extra
arguments, links or files to open, an inspector port). A call that uses one
is its own gate subject, an outward act, so it asks in Ask First and in
Auto, on a card that shows the path, the script or what runs. The handler sends the
argument to the shim as approved only when the gate decided on that subject
or the user approved the exact call (``mcp_seam.approved_for``), and the shim
refuses it otherwise.
"""

from typing import Any

from backend.blocks._preview import ChangePreview
from backend.copilot.constants import AUTOPILOT_NAME
from backend.copilot.gate.policy import Effect
from backend.copilot.gate.subject import Subject

_KEY_PREFIX = "local_pc_computer"
# What the card asks, per tool and argument; another approval argument gets a
# general ask and the shim's own words in the reason. A call that uses more
# than one is asked about by the first of them listed here.
_ASKS: dict[tuple[str, str], str] = {
    ("get_window_state", "screenshot_out_file"): "Save a screenshot on your computer",
    ("get_desktop_state", "screenshot_out_file"): "Save a screenshot on your computer",
    ("clipboard_write", "file_path"): "Copy a file on your computer to its clipboard",
    ("clipboard_write", "image_path"): (
        "Copy an image on your computer to its clipboard"
    ),
    ("browser_set_input_files", "files"): "Give a web page files from your computer",
    ("run_script", "script"): "Run a script on your computer",
    # The shim starts these out of sight, with no window of their own.
    ("launch_app", "launch_path"): "Run a program on your computer",
    ("launch_app", "path"): "Run a program on your computer",
    ("launch_app", "additional_arguments"): (
        "Start an app on your computer with command-line arguments"
    ),
    ("launch_app", "urls"): "Open links or files on your computer",
    ("launch_app", "webkit_inspector_port"): (
        "Open a debugging port into an app on your computer"
    ),
}
# The arguments a launch names its app with, shown before what it runs.
_LAUNCHED_APP = ("name", "bundle_id", "aumid")
# Arguments the card's headline shows on one line: a script's line breaks.
_ONE_LINE = frozenset({"script"})
# Arguments that write a file, which can replace one already there.
_WRITES = frozenset({"screenshot_out_file"})
_MAX_PATHS_CHARS = 300

# The tool list could not be read, so nobody can tell whether the call
# reaches the disk: it asks, as an unreadable run does.
UNREADABLE = Subject(
    key=f"{_KEY_PREFIX}:unreadable",
    name="Use your computer",
    effect=Effect.EXTERNAL,
    reason=(
        f"{AUTOPILOT_NAME} could not read which of this computer's tools reach "
        "its disk, so he asks."
    ),
)


def computer_subject(
    tool: str, arguments: dict[str, Any], spec: dict[str, Any] | None
) -> Subject:
    """The subject of one computer-use call: its own when it reaches the disk,
    else the tool's, so the card names the driver tool either way."""
    used = used_approval_arguments(spec, arguments)
    if not used:
        return Subject(
            key=_KEY_PREFIX,
            name="Use your computer",
            effect=Effect.PLATFORM,
            preview=ChangePreview(ask="Use your computer", object=tool),
        )
    ranked = [argument for (named, argument) in _ASKS if named == tool]
    ordered = sorted(
        used, key=lambda a: ranked.index(a) if a in ranked else len(ranked)
    )
    lead = ordered[0]
    ask = _ASKS.get((tool, lead), "Use a file on your computer")
    why = approval_arguments(spec)[lead]
    return Subject(
        key=disk_subject_key(tool, used),
        name=ask,
        effect=Effect.EXTERNAL,
        reason=(
            f"`{tool}` {why}. That reaches past the folders you shared, so "
            f"{AUTOPILOT_NAME} asks first."
        ),
        irreversible=any(argument in _WRITES for argument in used),
        preview=ChangePreview(ask=ask, object=_shown(tool, arguments, ordered)),
    )


def disk_subject_key(tool: str, used: list[str]) -> str:
    return f"{_KEY_PREFIX}:{tool}:{','.join(sorted(used))}"


def approval_arguments(spec: dict[str, Any] | None) -> dict[str, str]:
    """The tool's arguments that need the user's approval, as the shim named
    them, each with what it does on the disk."""
    named = (spec or {}).get("approval_arguments")
    if not isinstance(named, dict):
        return {}
    return {name: str(why) for name, why in named.items() if isinstance(name, str)}


def used_approval_arguments(
    spec: dict[str, Any] | None, arguments: dict[str, Any]
) -> list[str]:
    return sorted(
        name
        for name in approval_arguments(spec)
        if arguments.get(name) not in (None, "", [])
    )


def tool_spec(catalog: dict[str, Any], tool: str) -> dict[str, Any] | None:
    return next(
        (
            spec
            for spec in catalog.get("tools") or []
            if isinstance(spec, dict) and spec.get("name") == tool
        ),
        None,
    )


def _shown(tool: str, arguments: dict[str, Any], used: list[str]) -> str:
    """The paths, the script, or the app and what it runs, that the card's
    headline names."""
    values: list[str] = []
    if tool == "launch_app":
        values += [
            str(arguments[name]) for name in _LAUNCHED_APP if arguments.get(name)
        ][:1]
    for name in used:
        value = arguments.get(name)
        items = (
            [str(item) for item in value] if isinstance(value, list) else [str(value)]
        )
        values += [
            " ".join(item.split()) if name in _ONE_LINE else item for item in items
        ]
    shown = ", ".join(values)
    if len(shown) <= _MAX_PATHS_CHARS:
        return shown
    return shown[: _MAX_PATHS_CHARS - 1] + "…"
