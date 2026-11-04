"""Computer-use calls that reach the user's disk wait for their approval.

The shim names each tool's approval arguments in its tool list
(``approval_arguments``): ``screenshot_out_file``, clipboard ``file_path`` and
``image_path``, ``browser_set_input_files.files``. A call that uses one is its
own gate subject, an outward act, so it asks in Ask First and in Auto, on a
card that shows the path. The handler sends the argument to the shim as
approved only when the gate decided on that subject or the user approved the
exact call (``mcp_seam.approved_for``), and the shim refuses it otherwise.
"""

from typing import Any

from backend.blocks._preview import ChangePreview
from backend.copilot.constants import AUTOPILOT_NAME
from backend.copilot.gate.policy import Effect
from backend.copilot.gate.subject import Subject

_KEY_PREFIX = "local_pc_computer"
# What the card asks, per tool and argument; another approval argument gets a
# general ask and the shim's own words in the reason.
_ASKS: dict[tuple[str, str], str] = {
    ("get_window_state", "screenshot_out_file"): "Save a screenshot on your computer",
    ("get_desktop_state", "screenshot_out_file"): "Save a screenshot on your computer",
    ("clipboard_write", "file_path"): "Copy a file on your computer to its clipboard",
    ("clipboard_write", "image_path"): (
        "Copy an image on your computer to its clipboard"
    ),
    ("browser_set_input_files", "files"): "Give a web page files from your computer",
}
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
    ask = _ASKS.get((tool, used[0]), "Use a file on your computer")
    why = approval_arguments(spec)[used[0]]
    return Subject(
        key=disk_subject_key(tool, used),
        name=ask,
        effect=Effect.EXTERNAL,
        reason=(
            f"`{tool}` {why}. That reaches your computer's disk outside the "
            f"folders you shared, so {AUTOPILOT_NAME} asks first."
        ),
        irreversible=any(argument in _WRITES for argument in used),
        preview=ChangePreview(ask=ask, object=_paths(arguments, used)),
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


def _paths(arguments: dict[str, Any], used: list[str]) -> str:
    values: list[str] = []
    for name in used:
        value = arguments.get(name)
        values += (
            [str(item) for item in value] if isinstance(value, list) else [str(value)]
        )
    shown = ", ".join(values)
    if len(shown) <= _MAX_PATHS_CHARS:
        return shown
    return shown[: _MAX_PATHS_CHARS - 1] + "…"
