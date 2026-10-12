"""Task board: per-user columns and the pure rules behind them."""

import re
import uuid
from datetime import timedelta

from pydantic import BaseModel

BACKLOG = "backlog"
IN_PROGRESS = "in_progress"
DONE = "done"
BUILTIN_COLUMN_KEYS = (BACKLOG, IN_PROGRESS, DONE)
MAX_COLUMNS = 6
MAX_COLUMN_NAME = 40
UNPLACED_WINDOW = timedelta(days=7)
LAST_LINE_MAX = 140
# A reassigned session's first, hidden row: the old chat, briefed.
REASSIGN_BRIEF_KIND = "reassign_brief"

_CUSTOM_KEY = re.compile(
    r"col_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)
# Horizontal whitespace only: ``\s`` would span lines and go quadratic.
_LIST_MARKER = re.compile(r"(?m)^[ \t]*(?:\d+[.)]|[-*+])[ \t]+")
# The state line only needs the start of a reply.
_SCAN_MAX = 2000
_MARKDOWN_NOISE = re.compile(r"[#*_`>~]+")
_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")


class BoardColumn(BaseModel):
    key: str
    name: str
    builtin: bool = False


DEFAULT_COLUMNS = [
    BoardColumn(key=BACKLOG, name="Backlog", builtin=True),
    BoardColumn(key=IN_PROGRESS, name="In progress", builtin=True),
    BoardColumn(key=DONE, name="Done", builtin=True),
]


class BoardColumnsError(ValueError):
    """A column list the board cannot store."""


def new_column_key() -> str:
    return f"col_{uuid.uuid4()}"


def validate_columns(columns: list[BoardColumn]) -> list[BoardColumn]:
    if len(columns) > MAX_COLUMNS:
        raise BoardColumnsError(f"A board has at most {MAX_COLUMNS} columns")
    keys = [c.key for c in columns]
    if len(set(keys)) != len(keys):
        raise BoardColumnsError("Column keys must be unique")
    missing = [k for k in BUILTIN_COLUMN_KEYS if k not in keys]
    if missing:
        raise BoardColumnsError(f"Built-in columns cannot be removed: {missing}")
    result = []
    for column in columns:
        name = column.name.strip()
        if not name or len(name) > MAX_COLUMN_NAME:
            raise BoardColumnsError(
                f"Column names must be 1 to {MAX_COLUMN_NAME} characters"
            )
        builtin = column.key in BUILTIN_COLUMN_KEYS
        if not builtin and not _CUSTOM_KEY.fullmatch(column.key):
            raise BoardColumnsError(f"Invalid column key: {column.key!r}")
        result.append(BoardColumn(key=column.key, name=name, builtin=builtin))
    return result


def first_sentence(text: str | None) -> str | None:
    """The card's state line: the first sentence of a reply, one line, clipped."""
    if not text:
        return None
    head = text.lstrip()[:_SCAN_MAX]
    flat = " ".join(_MARKDOWN_NOISE.sub("", _LIST_MARKER.sub("", head)).split())
    if not flat:
        return None
    end = _SENTENCE_END.search(flat)
    sentence = flat[: end.end()] if end else flat
    if len(sentence) <= LAST_LINE_MAX:
        return sentence
    return sentence[: LAST_LINE_MAX - 1].rstrip() + "…"
