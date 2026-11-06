"""Which DB rows a resumed CLI session has not seen.

A resumed turn loads the CLI session file, which covers every DB row below the
transcript watermark (the next uncovered DB sequence; see
``backend.copilot.transcript.next_uncovered_sequence``).  The rows at or above
it are re-sent as a flat history prefix on the turn's message.  Everything
here works on sequences, never on positions in a filtered list: reasoning rows
take sequences but never reach the CLI, and the loaded window of a long
session does not start at sequence 0.
"""

import logging

from backend.copilot.constants import COPILOT_SYSTEM_PREFIX
from backend.copilot.markers import is_error_marker
from backend.copilot.model import ChatMessage
from backend.copilot.sdk.compaction import filter_compaction_messages
from backend.data.db_accessors import chat_db

logger = logging.getLogger(__name__)


def row_sequences(prior: list[ChatMessage]) -> list[int]:
    """The DB sequence of each row of *prior*.

    Every stored row has a sequence.  A row built in memory without one (test
    fixtures, a save that never back-filled) takes the sequence after the row
    before it, or its list position when no sequenced row precedes it.
    """
    sequences: list[int] = []
    for index, message in enumerate(prior):
        if message.sequence is not None:
            sequences.append(message.sequence)
            continue
        sequences.append(sequences[-1] + 1 if sequences else index)
    return sequences


def uncovered_rows(prior: list[ChatMessage], watermark: int) -> list[ChatMessage]:
    """Rows of *prior* at or above *watermark*, without reasoning rows.

    *prior* is the loaded history before the current message, reasoning rows
    included, in sequence order.
    """
    return [
        message
        for message, sequence in zip(prior, row_sequences(prior))
        if message.role != "reasoning" and sequence >= watermark
    ]


def missing_range(prior: list[ChatMessage], watermark: int) -> tuple[int, int] | None:
    """The sequences ``[watermark, first loaded)`` that are not in memory.

    Only a window that starts above the watermark has such a hole (the
    eager-load cap on a long session); its rows must come from the DB.  Only
    stored sequences say where the window starts.
    """
    stored = [m.sequence for m in prior if m.sequence is not None]
    if not stored:
        return None
    first_loaded = min(stored)
    if first_loaded <= watermark:
        return None
    return watermark, first_loaded


def is_marker_row(message: ChatMessage) -> bool:
    """True for an error marker or a system notice (stopped, incomplete,
    errored): UI state the model never said."""
    return is_error_marker(message) or bool(
        message.role == "assistant"
        and message.content
        and message.content.startswith(COPILOT_SYSTEM_PREFIX)
    )


def sendable_gap(rows: list[ChatMessage]) -> list[ChatMessage]:
    """The uncovered rows that may be shown to the model as its history."""
    return filter_compaction_messages([m for m in rows if not is_marker_row(m)])


async def fetch_missing_rows(
    session_id: str, missing: tuple[int, int], log_prefix: str
) -> list[ChatMessage] | None:
    """The rows in ``[start, end)`` from the DB, or ``None`` if the fetch failed."""
    start, end = missing
    try:
        page = await chat_db().get_chat_messages_paginated(
            session_id,
            limit=end - start,
            after_sequence=start,
            before_sequence=end,
        )
    except Exception as e:
        logger.warning(
            "%s [%s] hole-fill DB fetch failed (range=[%d,%d)):"
            " %s — sending gap without hole",
            log_prefix,
            session_id[:8],
            start,
            end,
            e,
        )
        return None
    if not page:
        return []
    return [m for m in page.messages if m.role != "reasoning"]
