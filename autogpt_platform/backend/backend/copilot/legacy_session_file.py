"""Restore a CLI session file an older session uploaded, without the
first-turn memory block the old code stored in it.

A file uploaded before warm context became query-only recorded every query
the engine sent, and the stored block (``legacy_first_turn_memory.py``)
reached it in three shapes:

- the first turn's query, in the file's first user entry: the stored first
  message as sent, after the engine's query-only blocks. The block is
  stripped there when the matcher proves it;
- the same query with pending messages the user sent meanwhile folded in
  front of the stored message (``combine_pending_with_current`` joins them
  with blank lines), so the block opens a later paragraph of that entry;
- a query rebuilt from the database (a turn without ``--resume``, whose CLI
  session starts from it), whose ``<conversation_history>`` opens with a
  ``User:`` line holding the stored first message: the block, or the part of
  it the old history compression kept when it cut that message short.

The last two cannot be stripped without editing text around them that a user
wrote, so the file is not resumed from at all: ``restore_session_file``
returns no content, the turn rebuilds its history from the database, where
``without_stored_first_turn_memory`` reads the first message without the
block, and the turn's upload replaces the file. That gives up, once, what
only the file held (tool calls in full, the CLI's own compaction).

Only those positions count: the platform wrote them, and a user's own words
never reach them with a ``<memory_context>`` tag intact, since the inbound
sanitizer removes it. The same text anywhere else, typed or pasted, is the
user's: it is neither stripped nor a reason to drop the file. Dropping a file
edits nothing, so the block's opening (or, in a history line, either end of
the block) is enough there; a strip needs the whole block proved.

Every other line, and the whole file when nothing is stripped, comes back
byte for byte; a rewritten line keeps its own line ending.
"""

import json

from pydantic import BaseModel

from .cli_session_entry import is_user_entry, rewrite_user_entry, user_entry_texts
from .legacy_first_turn_memory import (
    MEMORY_OPEN,
    after_query_blocks_end,
    memory_block_start,
    strip_first_turn_memory,
)

# Either tag of the block: a history the old compression cut short can keep
# only its closing one.
_PROBE = b"memory_context>"
_MEMORY_END = "\n</temporal_context>\n</memory_context>"
# The history opens a query rebuilt from the database, and its first line is
# ``User: `` and the session's first message.
_HISTORY_OPEN = "<conversation_history>\n"
# Where that first message ends: at the reply to it, or at the end of the
# history. Not at the next ``User:``, which a recalled episode inside the
# block can hold.
_FIRST_MESSAGE_ENDS = ("\nYou responded: ", "\n</conversation_history>")


class SessionFileRestore(BaseModel):
    """What a restore may do with a CLI session file."""

    content: bytes | None
    """The file to resume from; None when it must not be resumed from."""
    stripped: bool = False
    """The first user entry lost the block."""
    reason: str = ""
    """Why the file must not be resumed from."""


def restore_session_file(content: bytes) -> SessionFileRestore:
    """``content``, a CLI session file, as a restore may use it (see the
    module docstring)."""
    if _PROBE not in content:
        return SessionFileRestore(content=content)
    lines = content.splitlines(keepends=True)
    first = _first_user_entry(lines)
    if first is None:
        return SessionFileRestore(content=content)
    reason = _unstrippable_copy(lines, first)
    if reason:
        return SessionFileRestore(content=None, reason=reason)
    rewritten = rewrite_user_entry(_parse(lines[first]), _strip_query_text)
    if rewritten is None:
        return SessionFileRestore(content=content)
    line = lines[first]
    ending = line[len(line.rstrip(b"\r\n")) :]
    lines[first] = json.dumps(rewritten, ensure_ascii=False).encode() + ending
    return SessionFileRestore(content=b"".join(lines), stripped=True)


def _first_user_entry(lines: list[bytes]) -> int | None:
    """The index of the first ``"type": "user"`` line, whatever its shape:
    lines before it are skipped, and nothing after it is ever taken for the
    first turn's query."""
    return next(
        (index for index, line in enumerate(lines) if is_user_entry(_parse(line))),
        None,
    )


def _unstrippable_copy(lines: list[bytes], first: int) -> str:
    """Why the file holds a copy of the block a restore cannot strip, or
    ``""`` when it holds none."""
    for line in lines[first:]:
        if _PROBE not in line:
            continue
        if any(_history_copy(text) for text in user_entry_texts(_parse(line))):
            return (
                "a query rebuilt from the database copied the first-turn memory"
                " block into its history"
            )
    if any(_folded_copy(text) for text in user_entry_texts(_parse(lines[first]))):
        return (
            "its first query holds the first-turn memory block behind pending"
            " messages folded in front of it"
        )
    return ""


def _folded_copy(text: str) -> bool:
    """Whether the block opens a paragraph of the first query other than the
    one the platform put it at: a pending message was folded in front."""
    simple = memory_block_start(text, after_query_blocks=True)
    position = text.find(MEMORY_OPEN)
    while position != -1:
        if position != simple and text.endswith("\n\n", 0, position):
            return True
        position = text.find(MEMORY_OPEN, position + 1)
    return False


def _history_copy(text: str) -> bool:
    """Whether ``text`` is a query rebuilt from the database whose history's
    first line, the stored first message, holds the block or either end of
    it."""
    start = after_query_blocks_end(text)
    if not text.startswith(_HISTORY_OPEN, start):
        return False
    line_start = start + len(_HISTORY_OPEN)
    ends = [text.find(mark, line_start) for mark in _FIRST_MESSAGE_ENDS]
    line_end = min((end for end in ends if end != -1), default=len(text))
    first_message = text[line_start:line_end]
    return MEMORY_OPEN in first_message or _MEMORY_END in first_message


def _strip_query_text(text: str) -> str:
    stripped = strip_first_turn_memory(text, after_query_blocks=True)
    return text if stripped is None else stripped


def _parse(line: bytes) -> object:
    """A line's JSON value, or None when it is not JSON (or not UTF-8)."""
    try:
        return json.loads(line)
    except ValueError:
        return None
