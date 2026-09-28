"""Strip the first turn's warm-context block from the chat messages that hold it.

Until warm context became query-only (``graphiti/context_marker.py``), the SDK
engine wrote a session's first-turn Graphiti warm context into the session's
first user message (``inject_user_context(warm_ctx=...)``, since #12790),
right after the skill index::

    <memory_context>
    <temporal_context>
    <FACTS>
      - ...
    </FACTS>

    <RECENT_EPISODES>
      - ...
    </RECENT_EPISODES>
    </temporal_context>
    </memory_context>

followed by a blank line and the rest of the stored message. Every later turn
read it back: from the CLI session file on ``--resume`` (not reached here, see
below), and from this row wherever history is rebuilt from the database (a
turn without ``--resume``, a context-overflow retry, a baseline turn, the
transcript seeded from it), as did the dream's recent-session bodies. So a
fact the user forgot mid-session stayed in front of the model. This script
removes that block from the rows that still hold it.

It strips exactly what the platform wrote, where it wrote it:

- only a session's first message, and only a ``user`` row
  (``inject_user_context`` wrote no other row);
- only a block at the very start of that message, or right after the
  platform's ``<available_skills>`` block;
- only the exact structure ``graphiti/context.py`` rendered: both wrapping
  tags on lines of their own, one or both sections in that order, each item a
  ``  - `` line, and the blank line after the block;
- only when nothing after the block holds a ``<memory_context>`` or
  ``</memory_context>`` tag. ``inject_user_context`` stripped every such tag
  from the user's words before writing the row, so one there means the row is
  not the platform's own (a first turn that never reached injection keeps the
  user's raw text), or that stored memory forged an early close. Either way
  the row is counted ``left`` and not touched.

A ``<memory_context>`` tag anywhere else is left alone. The chat view has
always hidden a leading ``<memory_context>`` block on any user message
(``strip_injected_context_for_display``), and still does.

A row is written only while its session is idle, and only if it still holds
what was read. The session's cached copy in Redis (``copilot/model.py``) is
evicted right after the write and again at the end of the run: the next turn
would otherwise load the old first message from the cache, and a session in
use keeps refreshing that copy. A session with a turn queued or running, or a
row that changed since it was read, is skipped with nothing written and
counted busy; a write that raises, or an eviction that still fails at the end
of the run, is counted failed. Either makes the script exit 1: run it again.
Every step is idempotent, so a re-run only picks up what is left.

What it does not reach:

- CLI session files already uploaded for ``--resume``. An old session's file
  still holds the block in its first user entry, and the next upload does not
  rewrite it: the upload scrub removes only blocks carrying the current
  process's mark. An old session resumed on the SDK engine keeps reading the
  block until the CLI compacts that part of its history away or the file is
  replaced.
- Text derived from the block before it was removed: compaction summaries,
  the assistant's own replies and tool results in the session, Langfuse
  traces.
- A request that loaded an idle session before its row was written and saves
  it after the run's final eviction: it caches the old first message again.

Dry run by default: it only counts. Pass ``--apply`` to write.

Usage:

    poetry run python -m backend.copilot.first_turn_memory_backfill \\
        [--apply] [--batch-size N] [--session ID]

``--session`` handles one session (a canary).
"""

import argparse
import asyncio
import logging
import re
import sys

from pydantic import BaseModel

from backend.copilot.graphiti.context import CONTEXT_TAG_NAME
from backend.copilot.model import CHAT_SESSION_CACHE_PREFIX
from backend.copilot.service import MEMORY_CONTEXT_TAG, SKILLS_CONTEXT_TAG
from backend.data.db import (
    connect,
    disconnect,
    execute_raw_with_schema,
    query_raw_with_schema,
)
from backend.data.redis_client import get_redis_async

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 200

# One or more ``  - `` items, as ``context._format_context`` renders a
# section. An item's text may span lines.
_ITEMS = r"  - .*?"
_FACTS = rf"<FACTS>\n{_ITEMS}\n</FACTS>"
_EPISODES = rf"<RECENT_EPISODES>\n{_ITEMS}\n</RECENT_EPISODES>"
_PLATFORM_BLOCK_RE = re.compile(
    rf"(?P<skills><{SKILLS_CONTEXT_TAG}>\n.*?\n</{SKILLS_CONTEXT_TAG}>\n\n)?"
    rf"<{MEMORY_CONTEXT_TAG}>\n<{CONTEXT_TAG_NAME}>\n"
    rf"(?:{_FACTS}(?:\n\n{_EPISODES})?|{_EPISODES})"
    rf"\n</{CONTEXT_TAG_NAME}>\n</{MEMORY_CONTEXT_TAG}>\n\n",
    re.DOTALL,
)
# The tags the inbound sanitizer removes from a user's words
# (``service.strip_server_injected_tags``).
_MEMORY_TAG_RE = re.compile(rf"</?{MEMORY_CONTEXT_TAG}>", re.IGNORECASE)


def strip_first_turn_memory(content: str) -> str | None:
    """``content`` without the platform's first-turn memory block, or ``None``
    when it does not hold exactly that block where the platform wrote it (see
    the module docstring)."""
    match = _PLATFORM_BLOCK_RE.match(content)
    if match is None:
        return None
    rest = content[match.end() :]
    if _MEMORY_TAG_RE.search(rest):
        return None
    return (match["skills"] or "") + rest


class BackfillCounts(BaseModel):
    """What a run found. ``scanned`` counts the first messages holding a
    ``<memory_context>`` tag, ``stripped`` those whose block was removed (in a
    dry run, would be), ``left`` those that do not hold exactly the platform's
    block; ``busy`` and ``failed`` are what a re-run must pick up."""

    scanned: int = 0
    stripped: int = 0
    left: int = 0
    busy: int = 0
    failed: int = 0


class _FirstMessage(BaseModel):
    id: str
    session_id: str
    content: str
    chat_status: str


async def backfill(
    *,
    apply: bool,
    batch_size: int = DEFAULT_BATCH_SIZE,
    session_id: str | None = None,
) -> BackfillCounts:
    """Strip the block from every first message holding it, or from one
    session's, ``batch_size`` rows at a time; with ``apply`` False, count."""
    counts = BackfillCounts()
    changed: list[str] = []
    after = ""
    while batch := await _first_messages(after, batch_size, session_id):
        after = batch[-1].id
        for row in batch:
            await _handle(row, apply=apply, counts=counts, changed=changed)
    counts.failed += await _evict_all(changed)
    return counts


async def _handle(
    row: _FirstMessage, *, apply: bool, counts: BackfillCounts, changed: list[str]
) -> None:
    counts.scanned += 1
    stripped = strip_first_turn_memory(row.content)
    if stripped is None:
        counts.left += 1
        logger.info(f"Left message {row.id}: not the platform's first-turn block")
        return
    if not apply:
        counts.stripped += 1
        return
    try:
        written = await _write(row, stripped)
    except Exception:
        logger.warning(f"Could not strip message {row.id}", exc_info=True)
        counts.failed += 1
        return
    if not written:
        logger.info(
            f"Skipped message {row.id}: its session is not idle, or the message "
            "changed since it was read"
        )
        counts.busy += 1
        return
    counts.stripped += 1
    changed.append(row.session_id)
    try:
        await _evict(row.session_id)
    except Exception:
        # The end of the run evicts every changed session again.
        logger.warning(f"Could not evict session {row.session_id}", exc_info=True)


async def _first_messages(
    after: str, limit: int, session_id: str | None
) -> list[_FirstMessage]:
    """The next ``limit`` first messages, by id after ``after``, that hold a
    ``<memory_context>`` tag."""
    rows = await query_raw_with_schema(
        _FIRST_MESSAGES_QUERY, after, limit, f"<{MEMORY_CONTEXT_TAG}>", session_id
    )
    return [_FirstMessage.model_validate(row) for row in rows]


async def _write(row: _FirstMessage, stripped: str) -> bool:
    """Write ``stripped`` if the session is idle and the row still holds what
    was read; whether it was written."""
    if row.chat_status != "idle":
        return False
    written = await execute_raw_with_schema(_STRIP_QUERY, row.id, row.content, stripped)
    return written == 1


async def _evict_all(session_ids: list[str]) -> int:
    """Evict each changed session's cached copy once more, catching a copy a
    request cached again from what it had loaded before the write; how many
    evictions failed."""
    failed = 0
    for session_id in session_ids:
        try:
            await _evict(session_id)
        except Exception:
            logger.warning(f"Could not evict session {session_id}", exc_info=True)
            failed += 1
    return failed


async def _evict(session_id: str) -> None:
    redis = await get_redis_async()
    await redis.delete(f"{CHAT_SESSION_CACHE_PREFIX}{session_id}")


# The first message of each session, by the lowest sequence: the row
# ``inject_user_context`` rewrote. ``$4`` narrows the scan to one session.
_FIRST_MESSAGES_QUERY = """
SELECT m.id, m."sessionId" AS session_id, m.content,
       s."chatStatus"::text AS chat_status
FROM {schema_prefix}"ChatMessage" AS m
JOIN {schema_prefix}"ChatSession" AS s ON s.id = m."sessionId"
WHERE m.id > $1
  AND m.role = 'user'
  AND strpos(m.content, $3) > 0
  AND ($4::text IS NULL OR m."sessionId" = $4)
  AND m.sequence = (
      SELECT min(f.sequence)
      FROM {schema_prefix}"ChatMessage" AS f
      WHERE f."sessionId" = m."sessionId"
  )
ORDER BY m.id
LIMIT $2
"""

# Compare-and-set, only while no turn is queued or running in the session.
_STRIP_QUERY = """
UPDATE {schema_prefix}"ChatMessage" AS m
SET content = $3
FROM {schema_prefix}"ChatSession" AS s
WHERE m.id = $1
  AND m.content = $2
  AND s.id = m."sessionId"
  AND s."chatStatus" = 'idle'
"""


async def main(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    await connect()
    try:
        counts = await backfill(
            apply=args.apply, batch_size=args.batch_size, session_id=args.session
        )
    finally:
        await disconnect()
    verb = "stripped" if args.apply else "would strip (dry run)"
    print(
        f"{verb} the first-turn memory block from {counts.stripped} of "
        f"{counts.scanned} first messages holding <{MEMORY_CONTEXT_TAG}>; "
        f"left {counts.left} that do not hold exactly the platform's block"
    )
    if not (counts.busy or counts.failed):
        return 0
    print(f"skipped {counts.busy} busy and {counts.failed} failed: run again")
    return 1


def _batch_size(value: str) -> int:
    size = int(value)
    if size < 1:
        raise argparse.ArgumentTypeError("the batch size must be at least 1")
    return size


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the changes; without it, only count (dry run).",
    )
    parser.add_argument(
        "--batch-size",
        type=_batch_size,
        default=DEFAULT_BATCH_SIZE,
        help=f"Rows read per query (default {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument("--session", help="Handle one session instead of all.")
    sys.exit(asyncio.run(main(parser.parse_args())))
