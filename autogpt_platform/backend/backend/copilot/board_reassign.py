"""Reassigning a board card: the brief the new assignee reads first.

The new session opens with one hidden user row carrying the old
conversation, so the new assignee can say where the task stands.
"""

from typing import Any

from .expert_kickoff import is_hidden_chat_message
from .model import ChatMessage
from .service import strip_injected_context_for_display

REASSIGN_BRIEF_KIND = "reassign_brief"
TRANSCRIPT_CHAR_BUDGET = 24_000

_OMITTED = "[earlier messages omitted]"

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
        from_name=from_name,
        transcript=_transcript(_turns(messages, from_name)),
        files_block=_files_block(files),
    )


def _turns(messages: list[ChatMessage], from_name: str) -> list[str]:
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
            turns.append(f"{speaker}: {content}")
    return turns


def _transcript(turns: list[str]) -> str:
    """Newest last; the oldest whole turns go first when it runs long."""
    joined = "\n".join(turns)
    if len(joined) <= TRANSCRIPT_CHAR_BUDGET:
        return joined
    budget = TRANSCRIPT_CHAR_BUDGET - len(_OMITTED) - 1
    kept: list[str] = []
    size = 0
    for turn in reversed(turns):
        cost = len(turn) + (1 if kept else 0)
        if size + cost > budget:
            break
        kept.append(turn)
        size += cost
    if not kept:
        kept = [turns[-1][-budget:]]
    return "\n".join([_OMITTED, *reversed(kept)])


def _files_block(files: list[tuple[str, str]]) -> str:
    if not files:
        return ""
    lines = "".join(f"- {name} (file_id: {file_id})\n" for name, file_id in files)
    return (
        f"\n<files>\n{lines}</files>\n"
        "These files are readable from this conversation."
    )
