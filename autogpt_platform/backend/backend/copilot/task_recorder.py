"""Write what a task's chat does into the task's record, as it happens.

Runs after every tool call (``BaseTool.execute``). A chat that is working on a
task gets one plain line per call ("Searched the web"), an artifact for each
file it writes and verified evidence for each call that changed something
outside. The lines are the harness's own labels: a tool's output is never
copied in, so text planted in a web page or an email cannot reach the record.

Best-effort throughout: the record matters less than the tool call.
"""

import json
import logging
import time
from typing import Any

from prisma.enums import TaskArtifactKind

from backend.copilot.model import ChatSession
from backend.copilot.response_model import StreamToolOutputAvailable
from backend.data.db_accessors import task_record_db
from backend.data.task_record import ArtifactInput, ToolCallRecord

logger = logging.getLogger(__name__)

# Tools that write the record themselves, or say nothing about the work.
SILENT_TOOLS = frozenset(
    {
        "report_progress",
        "add_evidence",
        "finish_task",
        "start_task",
        "create_task",
        "list_tasks",
        "get_task",
        "update_task",
        "archive_task",
        "ask_question",
        "find_capability",
        "describe_capability",
        "resume_capability",
        "get_sub_session_result",
        "TodoWrite",
        "get_platform_info",
    }
)

# One line per call, and the folded line for several in a row.
_LABELS: dict[str, tuple[str, str]] = {
    "web_search": ("Searched the web", "Searched the web {count} times"),
    "web_fetch": ("Read a web page", "Read {count} web pages"),
    "browser_navigate": ("Opened a page in the browser", "Opened {count} pages"),
    "browser_act": ("Worked in the browser", "Took {count} steps in the browser"),
    "browser_screenshot": ("Took a screenshot", "Took {count} screenshots"),
    "read_workspace_file": ("Read a file", "Read {count} files"),
    "list_workspace_files": ("Looked through the files", "Looked through the files"),
    "delete_workspace_file": ("Removed a file", "Removed {count} files"),
    "memory_search": ("Checked what it remembers", "Checked what it remembers"),
    "memory_store": ("Saved a note to memory", "Saved {count} notes to memory"),
    "bash_exec": ("Ran a command", "Ran {count} commands"),
    "run_agent": ("Ran a workflow", "Ran {count} workflows"),
    "run_sub_session": ("Handed part of the work on", "Handed {count} parts on"),
    "delegate_to_expert": ("Handed part of the work to a teammate", ""),
    "consult_teammate": ("Asked a teammate", "Asked teammates {count} times"),
    "post_to_chat_platform": ("Posted a message", "Posted {count} messages"),
    "schedule_followup": ("Scheduled a follow-up", "Scheduled {count} follow-ups"),
    "schedule_routine": ("Set up a routine", "Set up {count} routines"),
    "search_docs": ("Searched the docs", "Searched the docs {count} times"),
    "find_agent": ("Looked for a workflow", "Looked for workflows"),
    "find_library_agent": ("Looked in the library", "Looked in the library"),
    "create_agent": ("Built a workflow", "Built {count} workflows"),
    "edit_agent": ("Edited a workflow", "Edited workflows"),
    "connect_integration": ("Asked to connect an account", ""),
}

# Platform tools whose success changes something outside the task itself, so
# the call is its own proof.
_ACTING_TOOLS: dict[str, str] = {
    "run_agent": "Ran a workflow",
    "post_to_chat_platform": "Posted a message",
    "schedule_followup": "Scheduled a follow-up",
    "schedule_routine": "Set up a routine",
    "create_agent": "Built a workflow",
}
# A capability named with one of these does something, rather than reads.
_ACTING_WORDS = (
    "send",
    "create",
    "post",
    "publish",
    "reply",
    "update",
    "delete",
    "remove",
    "book",
    "pay",
    "invite",
    "schedule",
    "upload",
    "write",
    "add",
)

# Reads whose page can later be given as proof of what it says.
_PAGE_READS = frozenset({"web_fetch", "browser_navigate"})

_NO_TASK_TTL_SECONDS = 10.0
_no_task_until: dict[str, float] = {}


def forget(session_id: str) -> None:
    """A task was just given to this chat: start asking again."""
    _no_task_until.pop(session_id, None)


def describe(
    tool_name: str, args: dict[str, Any], output: str | None, success: bool
) -> ToolCallRecord | None:
    """What one tool call adds to the record, or None for a call that adds
    nothing: a failed one, a dry run, or plumbing."""
    result = _as_dict(output)
    # In a chat a tool that fails still returns normally, with an error
    # response: a file that was refused must not read as written.
    if not success or result.get("type") == "error":
        return None
    name, inner_args, capability = _unwrap(tool_name, args)
    if name is None or name in SILENT_TOOLS:
        return None
    label, label_many = _LABELS.get(name, (_fallback_label(name, capability), ""))
    record = ToolCallRecord(
        tool=name, tool_call_id="", label=label, label_many=label_many or label
    )
    if name == "write_workspace_file":
        title = str(
            result.get("name")
            or inner_args.get("filename")
            or inner_args.get("path")
            or ""
        ).strip()
        file_id = _text(result.get("file_id"))
        if title and file_id:
            record.label = record.label_many = f"Wrote {title}"
            record.artifact = ArtifactInput(
                title=title, kind=TaskArtifactKind.FILE, workspace_file_id=file_id
            )
    elif name in _PAGE_READS:
        record.url = _text(inner_args.get("url"))
    elif name == "delete_workspace_file":
        record.deleted_file_id = _text(
            result.get("file_id") or inner_args.get("file_id")
        )
    if name in _ACTING_TOOLS:
        record.evidence_claim = _ACTING_TOOLS[name]
    elif capability and _acts(capability):
        record.evidence_claim = label
    return record


def _unwrap(
    tool_name: str, args: dict[str, Any]
) -> tuple[str | None, dict[str, Any], str | None]:
    """The tool behind a ``run_capability`` call: its name, its own input, and
    the capability id when it is not a platform tool."""
    if tool_name != "run_capability":
        return tool_name, args, None
    if args.get("validate_only"):
        return None, {}, None
    capability = str(args.get("id") or "")
    inner = args.get("input")
    inner_args = inner if isinstance(inner, dict) else {}
    if capability.startswith("tool:"):
        return capability.removeprefix("tool:"), inner_args, None
    return capability or None, inner_args, capability or None


def _fallback_label(name: str, capability: str | None) -> str:
    if capability:
        kind, _, rest = capability.partition(":")
        words = rest.rsplit("/", 1)[-1].replace("_", " ").replace("-", " ").strip()
        return f"Used {words}" if words else f"Used a {kind or 'tool'}"
    return f"Used {name.replace('_', ' ')}"


def _acts(capability: str) -> bool:
    words = capability.lower().replace("-", "_").replace(":", "_").replace("/", "_")
    return any(part in _ACTING_WORDS for part in words.split("_"))


def _as_dict(output: str | None) -> dict[str, Any]:
    if not output:
        return {}
    try:
        parsed = json.loads(output)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


async def record(
    session: ChatSession,
    tool_name: str,
    tool_call_id: str,
    args: dict[str, Any],
    output: StreamToolOutputAvailable,
) -> None:
    """Record one finished tool call on the task this chat is working on."""
    user_id = session.user_id
    if not user_id or _no_task_until.get(session.session_id, 0) > time.monotonic():
        return
    try:
        call = describe(
            tool_name,
            args,
            output.output if isinstance(output.output, str) else None,
            output.success,
        )
        if call is None:
            return
        call.tool_call_id = tool_call_id
        has_task = await task_record_db().record_tool_call(
            session.session_id, user_id, call, session.expert_id
        )
        if not has_task:
            _no_task_until[session.session_id] = time.monotonic() + _NO_TASK_TTL_SECONDS
    except Exception as e:
        logger.warning(
            "Could not record %s on the task of %s: %s",
            tool_name,
            session.session_id,
            e,
        )
