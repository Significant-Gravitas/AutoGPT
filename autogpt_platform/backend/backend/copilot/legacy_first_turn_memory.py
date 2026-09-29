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

The renderer that wrote it, ``graphiti/context._format_context`` with the
helpers of ``graphiti/_format.py``, did not change on master from #12720 to
this change: ``  - {fact} ({valid_from} — {valid_to})`` for a fact and
``  - [{created_at}] {body}`` for an episode, the times ``str()`` of a
datetime (``unknown`` and ``present`` when unset), the sections in that order
with at least one line each. This stack's recall policy
(``graphiti/recall_render.py``) writes ``(valid: {from} — {to})`` or how and
when the fact was retired instead; a block it wrote is matched as well.

``strip_first_turn_memory`` removes the block only where the platform put it,
and only when every part of it is what that renderer writes:

- at the start of the text or right after the platform's ``<available_skills>``
  block; in a CLI session entry also after the query-only blocks the engine
  put in front of the stored message (``<skills_update>``,
  ``<builder_context>``, ``<budget_status>``), which a stored message never
  holds;
- the wrapper and the sections exactly as rendered, every fact line ending in
  a validity or retirement stamp, every episode line starting with a
  bracketed ``created_at``;
- nothing after the block holding a ``<memory_context>`` tag: the inbound
  sanitizer removes those from a user's words, so one there means the text is
  not the platform's own, or that stored memory forged an early close.

Text that fails any of these is left as it is, however much it looks like the
block: a raw first message that never reached the sanitizer, or an imported
row, can hold a ``<memory_context>`` block a user wrote.

It reads the text once and never backtracks: a leading block ends at the first
closing tag of its name, the memory block at the first ``</temporal_context>``
and ``</memory_context>`` pair. The sanitizer leaves ``<budget_status>`` in a
user's words, and a pattern that tried every way to split such text into
blocks took time exponential in their number.

``without_stored_first_turn_memory`` applies it wherever a stored first
message becomes model input. The tag names are literals: they are the ones
the old code wrote, and must not follow a later rename.
"""

import re

from .model import ChatMessage

MEMORY_OPEN = "<memory_context>\n<temporal_context>\n"
MEMORY_CLOSE = "\n</temporal_context>\n</memory_context>\n\n"
_QUERY_TAGS = ("skills_update", "builder_context", "budget_status")
_SKILLS_TAG = "available_skills"
_FACTS = ("<FACTS>\n", "\n</FACTS>")
_EPISODES = ("<RECENT_EPISODES>\n", "\n</RECENT_EPISODES>")
_BETWEEN_SECTIONS = f"{_FACTS[1]}\n\n{_EPISODES[0]}"
_ITEM = "  - "
# ``str()`` of a datetime: microseconds when set, the offset when aware.
_DATETIME = (
    r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d{6})?"
    r"(?:[+-]\d{2}:\d{2}(?::\d{2}(?:\.\d{6})?)?)?"
)
# What closes a fact line: when it holds (``valid: `` since the recall
# policy), or how and when it was retired.
_FACT_STAMP_RE = re.compile(
    rf"(?:valid: )?(?:{_DATETIME}|unknown) — (?:{_DATETIME}|present)"
    rf"|(?:superseded|contradicted|retracted|expired)"
    rf" (?:{_DATETIME}|at an unknown time)"
)
_EPISODE_STAMP_RE = re.compile(rf"\[{_DATETIME}\] ")
# The tags the inbound sanitizer removes from a user's words
# (``service.strip_server_injected_tags``).
_MEMORY_TAG_RE = re.compile(r"</?memory_context>", re.IGNORECASE)


def strip_first_turn_memory(
    content: str, *, after_query_blocks: bool = False
) -> str | None:
    """``content`` without the platform's first-turn memory block, or ``None``
    when it does not hold a block the renderer provably wrote, where the
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
    if body_end == -1 or not _is_rendered(content[body_start:body_end]):
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


def _is_rendered(body: str) -> bool:
    """Whether ``body`` is ``<FACTS>``, ``<RECENT_EPISODES>`` or both, in that
    order, as the renderer wrote them."""
    if body.startswith(_EPISODES[0]):
        return _are_episodes(_lines(body, _EPISODES))
    if not body.startswith(_FACTS[0]):
        return False
    if body.endswith(_FACTS[1]) and _are_facts(_lines(body, _FACTS)):
        return True
    between = body.find(_BETWEEN_SECTIONS)
    if between == -1:
        return False
    facts = body[: between + len(_FACTS[1])]
    episodes = body[between + len(_BETWEEN_SECTIONS) - len(_EPISODES[0]) :]
    return _are_facts(_lines(facts, _FACTS)) and _are_episodes(
        _lines(episodes, _EPISODES)
    )


def _lines(section: str, tags: tuple[str, str]) -> str | None:
    """What is between a section's tags, or None when it is not wrapped in
    them."""
    opening, closing = tags
    if len(section) < len(opening) + len(closing):
        return None
    if not (section.startswith(opening) and section.endswith(closing)):
        return None
    return section[len(opening) : -len(closing)]


def _are_facts(lines: str | None) -> bool:
    items = _items(lines)
    return bool(items) and all(_ends_with_fact_stamp(item) for item in items)


def _are_episodes(lines: str | None) -> bool:
    items = _items(lines)
    return bool(items) and all(_EPISODE_STAMP_RE.match(item) for item in items)


def _items(lines: str | None) -> list[str]:
    """A section's ``  - `` items without the marker; none when the section
    does not open on one."""
    if lines is None or not lines.startswith(_ITEM):
        return []
    return lines[len(_ITEM) :].split(f"\n{_ITEM}")


def _ends_with_fact_stamp(item: str) -> bool:
    """Whether ``item`` ends with `` (stamp)``: when the fact holds, or how
    it was retired."""
    opening = item.rfind(" (")
    return (
        opening != -1
        and item.endswith(")")
        and _FACT_STAMP_RE.fullmatch(item, opening + 2, len(item) - 1) is not None
    )
