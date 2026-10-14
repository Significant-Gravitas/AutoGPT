"""Reassigning a board card: the brief the new assignee reads first.

The new session opens with one hidden user row carrying the old
conversation, so the new assignee can say where the task stands.
"""

import re
from typing import Any

from .board import REASSIGN_BRIEF_KIND
from .expert_kickoff import is_hidden_chat_message
from .model import ChatMessage
from .service import strip_injected_context_for_display

TRANSCRIPT_CHAR_BUDGET = 24_000
MAX_BRIEF_FILES = 50

_OMITTED = "[earlier messages omitted]"
# The brief's own wrapper tags: text inside must not open or close them.
_WRAPPER_TAG = re.compile(r"<(/?)(previous_conversation|files)", re.IGNORECASE)
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f]+")

_BRIEF = """\
The user reassigned this task from {from_name} to you. Below is the conversation so far, for context only.
Reply in at most four short lines: where the task stands, then one concrete next step you suggest, then ask whether to go ahead.
Open with "You re-assigned this task to me." Do not restate the transcript.

<previous_conversation>
{transcript}
</previous_conversation>
{files_block}"""


def reassign_brief_metadata(expert_id: str | None) -> dict[str, Any]:
    """Keeps the brief off user surfaces; it still counts as the start."""
    return {"hidden": True, "kind": REASSIGN_BRIEF_KIND, "expert_id": expert_id}


def build_reassign_brief(
    *,
    from_name: str,
    to_name: str,
    messages: list[ChatMessage],
    files: list[tuple[str, str]],
) -> str:
    """The new assignee's first message. *files* are ``(name, file_id)``."""
    return _BRIEF.format(
        from_name=_neutralise(from_name),
        transcript=_transcript(_turns(messages, from_name)),
        files_block=_files_block(files),
    )


def _turns(messages: list[ChatMessage], from_name: str) -> list[tuple[str, str]]:
    """``(speaker, text)`` pairs, oldest first."""
    turns = []
    for message in messages:
        if message.role not in ("user", "assistant") or not message.content:
            continue
        if is_hidden_chat_message(message.metadata):
            continue
        if message.role == "user":
            content = strip_injected_context_for_display(message.content)
            speaker = "User"
        else:
            content = message.content
            speaker = from_name
        if content.strip():
            turns.append((_neutralise(speaker), _neutralise(content)))
    return turns


def _transcript(turns: list[tuple[str, str]]) -> str:
    """Newest last; the oldest whole turns go first when it runs long."""
    lines = [f"{speaker}: {text}" for speaker, text in turns]
    joined = "\n".join(lines)
    if len(joined) <= TRANSCRIPT_CHAR_BUDGET:
        return joined
    budget = TRANSCRIPT_CHAR_BUDGET - len(_OMITTED) - 1
    kept: list[str] = []
    size = 0
    for line in reversed(lines):
        cost = len(line) + (1 if kept else 0)
        if size + cost > budget:
            break
        kept.append(line)
        size += cost
    if not kept:
        speaker, text = turns[-1]
        label = f"{speaker}: "
        room = max(budget - len(label), 0)
        kept = [label + text[len(text) - room :]]
    return "\n".join([_OMITTED, *reversed(kept)])


def _files_block(files: list[tuple[str, str]]) -> str:
    if not files:
        return ""
    lines = "".join(
        f"- {_file_name(name)} (file_id: {file_id})\n"
        for name, file_id in files[:MAX_BRIEF_FILES]
    )
    if len(files) > MAX_BRIEF_FILES:
        lines += f"- … and {len(files) - MAX_BRIEF_FILES} more\n"
    return (
        f"\n<files>\n{lines}</files>\n"
        "These files are readable from this conversation."
    )


def _file_name(name: str) -> str:
    return _neutralise(_CONTROL_CHARS.sub(" ", name).strip())


def _neutralise(text: str) -> str:
    return _WRAPPER_TAG.sub(r"&lt;\1\2", text)
