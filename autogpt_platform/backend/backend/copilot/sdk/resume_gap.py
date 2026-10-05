"""Which DB rows a resumed CLI session has not seen.

A resumed turn loads the CLI session file, which covers every DB row below the
transcript watermark (the next uncovered DB sequence; see
``backend.copilot.transcript.next_uncovered_sequence``).  The rows at or above
it are re-sent as a flat history prefix on the turn's message.  Everything
here works on sequences, never on positions in a filtered list: reasoning rows
take sequences but never reach the CLI, and the loaded window of a long
session does not start at sequence 0.
"""

from backend.copilot.constants import (
    COPILOT_ERROR_PREFIX,
    COPILOT_RETRYABLE_ERROR_PREFIX,
    COPILOT_SYSTEM_PREFIX,
)
from backend.copilot.model import ChatMessage
from backend.copilot.sdk.compaction import filter_compaction_messages

_MARKER_PREFIXES = (
    COPILOT_ERROR_PREFIX,
    COPILOT_RETRYABLE_ERROR_PREFIX,
    COPILOT_SYSTEM_PREFIX,
)


def row_sequence(index: int, message: ChatMessage) -> int:
    """The row's DB sequence, or its list position when it has none.

    Every stored row has a sequence.  Rows built in memory without one (test
    fixtures, a save that never back-filled) fall back to their position in
    the unfiltered list, which equals the sequence for an uncapped history.
    """
    return message.sequence if message.sequence is not None else index


def uncovered_rows(prior: list[ChatMessage], watermark: int) -> list[ChatMessage]:
    """Rows of *prior* at or above *watermark*, without reasoning rows.

    *prior* is the loaded history before the current message, reasoning rows
    included, in sequence order.
    """
    return [
        message
        for index, message in enumerate(prior)
        if message.role != "reasoning" and row_sequence(index, message) >= watermark
    ]


def missing_range(prior: list[ChatMessage], watermark: int) -> tuple[int, int] | None:
    """The sequences ``[watermark, first loaded)`` that are not in memory.

    Only a window that starts above the watermark has such a hole (the
    eager-load cap on a long session); its rows must come from the DB.
    """
    if not prior:
        return None
    first_loaded = min(row_sequence(index, m) for index, m in enumerate(prior))
    if first_loaded <= watermark:
        return None
    return watermark, first_loaded


def is_marker_row(message: ChatMessage) -> bool:
    """True for an error or system marker: UI state the model never said."""
    if message.role != "assistant" or not message.content:
        return False
    return message.content.startswith(_MARKER_PREFIXES)


def sendable_gap(rows: list[ChatMessage]) -> list[ChatMessage]:
    """The uncovered rows that may be shown to the model as its history."""
    return filter_compaction_messages([m for m in rows if not is_marker_row(m)])
