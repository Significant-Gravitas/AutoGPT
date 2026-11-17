"""The first-turn memory block older sessions still hold: how to prove the
platform wrote it, and how to read a stored message without it.

Until warm context became query-only (``graphiti/context_marker.py``), the SDK
engine wrote a session's first-turn Graphiti warm context into the session's
first user message (``inject_user_context(warm_ctx=...)``, from #12790),
wrapped in ``<memory_context>`` and followed by a blank line, after the skill
index when there was one::

    <memory_context>
    <temporal_context>
    <FACTS>
      - Alice works on Atlas (2025-06-01 00:00:00+00:00 — present)
    </FACTS>

    <RECENT_EPISODES>
      - [2025-06-01 00:00:00+00:00] what is Alice working on
    </RECENT_EPISODES>
    </temporal_context>
    </memory_context>

``strip_first_turn_memory`` removes the block only where the platform put it,
and only when a renderer could have written all of it:

- at the start of the text or right after the platform's ``<available_skills>``
  block; in a CLI session entry also after the query-only blocks the engine
  put in front of the stored message (``<skills_update>``,
  ``<builder_context>``, ``<budget_status>``), which a stored message never
  holds;
- the wrapper exactly as written, around a body production's renderer or
  this stack's could have written: every stamp and time, every episode's
  length (``legacy_first_turn_memory_body.py``);
- nothing after the block holding a ``<memory_context>`` tag: the inbound
  sanitizer removes those from a user's words, so one there means the text is
  not the platform's own, or that stored memory forged an early close.

Text that fails any of these is left as it is, however much it looks like the
block: a raw first message that never reached the sanitizer, or an imported
row, can hold a ``<memory_context>`` block a user wrote. Text alone cannot say
who wrote it, though: a byte-for-byte copy of a block a renderer could have
written, typed or imported at that position, is removed too.

It reads the text once and never backtracks: a leading block ends at the first
closing tag of its name, the memory block at the first ``</temporal_context>``
and ``</memory_context>`` pair. The sanitizer leaves ``<budget_status>`` in a
user's words, and a pattern that tried every way to split such text into
blocks took time exponential in their number.

``without_stored_first_turn_memory`` applies it wherever a stored first
message becomes model input or a tool's output. The tag names are literals:
they are the ones the old code wrote, and must not follow a later rename.
"""

import re

from .legacy_first_turn_memory_body import is_rendered
from .model import ChatMessage

MEMORY_OPEN = "<memory_context>\n<temporal_context>\n"
MEMORY_CLOSE = "\n</temporal_context>\n</memory_context>\n\n"
_QUERY_TAGS = ("skills_update", "builder_context", "budget_status")
_SKILLS_TAG = "available_skills"
# The tags the inbound sanitizer removes from a user's words
# (``service.strip_server_injected_tags``).
_MEMORY_TAG_RE = re.compile(r"</?memory_context>", re.IGNORECASE)


def strip_first_turn_memory(
    content: str, *, after_query_blocks: bool = False
) -> str | None:
    """``content`` without the platform's first-turn memory block, or ``None``
    when it does not hold a block a renderer could have written, where the
    platform put it (see the module docstring).

    ``after_query_blocks`` is for a CLI session entry, which may open with the
    engine's query-only blocks; a stored message never does, so the backfill
    and the history readers leave it False.
    """
    block_start = memory_block_start(content, after_query_blocks=after_query_blocks)
    if not content.startswith(MEMORY_OPEN, block_start):
        return None
    body_start = block_start + len(MEMORY_OPEN)
    body_end = content.find(MEMORY_CLOSE, body_start)
    if body_end == -1 or not is_rendered(content[body_start:body_end]):
        return None
    rest = content[body_end + len(MEMORY_CLOSE) :]
    if _MEMORY_TAG_RE.search(rest):
        return None
    return content[:block_start] + rest


def without_stored_first_turn_memory(
    messages: list[ChatMessage],
) -> list[ChatMessage]:
    """``messages`` as a model may read them: the session's first message
    (sequence 0, a user row) without the block the platform stored in it when
    it provably holds one, every other message as it is. The list itself comes
    back when nothing changes.

    Model input never reads the old block, whatever the storage holds: a row
    the backfill has not reached, or a stale cached session.
    """
    for index, message in enumerate(messages):
        if message.sequence != 0:
            continue
        stripped = (
            strip_first_turn_memory(message.content)
            if message.role == "user" and message.content
            else None
        )
        if stripped is None:
            return messages
        readable = list(messages)
        readable[index] = message.model_copy(update={"content": stripped})
        return readable
    return messages


def memory_block_start(content: str, *, after_query_blocks: bool = False) -> int:
    """Where the platform put the block in ``content``: past the query-only
    blocks (with ``after_query_blocks``) and past the ``<available_skills>``
    block."""
    start = after_query_blocks_end(content) if after_query_blocks else 0
    skills_end = block_end(content, start, _SKILLS_TAG)
    return start if skills_end is None else skills_end


def after_query_blocks_end(content: str) -> int:
    """Where ``content`` goes on past the query-only blocks it opens with."""
    position = 0
    while (end := _query_block_end(content, position)) is not None:
        position = end
    return position


def block_end(content: str, position: int, tag: str) -> int | None:
    """Where the ``<tag>`` block opening at ``position`` ends, its blank line
    included, or None when none opens there. It ends at the first closing
    tag of its name."""
    opening, closing = f"<{tag}>\n", f"\n</{tag}>\n\n"
    if not content.startswith(opening, position):
        return None
    end = content.find(closing, position + len(opening))
    return None if end == -1 else end + len(closing)


def _query_block_end(content: str, position: int) -> int | None:
    ends = (block_end(content, position, tag) for tag in _QUERY_TAGS)
    return next((end for end in ends if end is not None), None)
