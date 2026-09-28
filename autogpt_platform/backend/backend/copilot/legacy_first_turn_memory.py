"""The first-turn memory block older sessions still hold, and how to find it.

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

followed by a blank line and the rest of the message. Two copies of it are
still read back:

- the stored first message, which history rebuilt from the database and the
  dream read: ``first_turn_memory_backfill.py`` strips it there, once;
- the first user entry of the CLI session file uploaded for ``--resume``,
  which recorded the first turn's query as sent: ``download_transcript``
  strips it there on every restore (``strip_first_turn_memory_from_session``),
  for both engines, so the session's next upload no longer carries it.

``strip_first_turn_memory`` is the one matcher for both. It removes exactly
the block the platform wrote, where it wrote it, and nothing else:

- only at the start of the stored message, or right after the platform's
  ``<available_skills>`` block; in a CLI session entry, also after the
  query-only blocks the engine put in front of the stored message
  (``<skills_update>``, ``<builder_context>``, ``<budget_status>``), which a
  stored message never holds;
- only the exact structure ``graphiti/context.py`` rendered: both wrapping
  tags on lines of their own, ``<FACTS>`` and/or ``<RECENT_EPISODES>`` in that
  order, each opening on a ``  - `` item, and the blank line after the block;
- only when nothing after the block holds a ``<memory_context>`` or
  ``</memory_context>`` tag. ``inject_user_context`` stripped every such tag
  from the user's words before it wrote the message, so one there means the
  text is not the platform's own (a first turn that never reached injection
  keeps the user's raw text), or that stored memory forged an early close.

It reads the text once, front to back, and never backtracks: each leading
block ends at the first closing tag of its own name, and the memory block at
the first ``</temporal_context>`` / ``</memory_context>`` pair. A user can
type ``<budget_status>`` and the section tags, and a pattern that tried every
way to split such text would take exponential time on a few kilobytes of it;
this takes linear time, and leaves text that is not unambiguously the
platform's alone.

In a CLI session file only the first user entry is looked at, the first
turn's query: a copy the user pasted into a later message is theirs.

The tag names are literals: they are the ones the old code wrote, and must not
follow a later rename.
"""

import json
import re

from .cli_session_entry import is_user_entry, rewrite_user_entry

_QUERY_TAGS = ("skills_update", "builder_context", "budget_status")
_SKILLS_TAG = "available_skills"
_MEMORY_OPEN = "<memory_context>\n<temporal_context>\n"
_MEMORY_CLOSE = "\n</temporal_context>\n</memory_context>\n\n"
_FACTS_OPEN = "<FACTS>\n  - "
_FACTS_CLOSE = "\n</FACTS>"
_EPISODES_OPEN = "<RECENT_EPISODES>\n  - "
_EPISODES_CLOSE = "\n</RECENT_EPISODES>"
_BETWEEN_SECTIONS = f"{_FACTS_CLOSE}\n\n{_EPISODES_OPEN}"
# The tags the inbound sanitizer removes from a user's words
# (``service.strip_server_injected_tags``).
_MEMORY_TAG_RE = re.compile(r"</?memory_context>", re.IGNORECASE)
_PROBE = b"<memory_context>"


def strip_first_turn_memory(
    content: str, *, after_query_blocks: bool = False
) -> str | None:
    """``content`` without the platform's first-turn memory block, or ``None``
    when it does not hold exactly that block where the platform wrote it (see
    the module docstring).

    ``after_query_blocks`` is for a CLI session entry, which may open with the
    engine's query-only blocks; a stored message never does, so the backfill
    leaves it False.
    """
    start = _after_query_blocks(content) if after_query_blocks else 0
    skills_end = _block_end(content, start, _SKILLS_TAG)
    block_start = start if skills_end is None else skills_end
    if not content.startswith(_MEMORY_OPEN, block_start):
        return None
    body_start = block_start + len(_MEMORY_OPEN)
    body_end = content.find(_MEMORY_CLOSE, body_start)
    if body_end == -1 or not _is_rendered(content[body_start:body_end]):
        return None
    rest = content[body_end + len(_MEMORY_CLOSE) :]
    if _MEMORY_TAG_RE.search(rest):
        return None
    return content[:block_start] + rest


def strip_first_turn_memory_from_session(content: bytes) -> bytes:
    """``content``, a CLI session file, without the first-turn block in its
    first user entry.

    Every other line, and the whole file when that entry does not hold
    exactly the block, comes back byte for byte. Lines are split on bytes, as
    the CLI wrote them: a string split would also break a line at a U+2028
    inside a JSON string, and could take a later entry for the first.
    """
    if _PROBE not in content:
        return content
    lines = content.splitlines(keepends=True)
    for index, line in enumerate(lines):
        entry = _parse(line)
        if not is_user_entry(entry):
            continue
        rewritten = rewrite_user_entry(entry, _strip_query_text)
        if rewritten is None:
            return content
        end = b"\n" if line.endswith(b"\n") else b""
        lines[index] = json.dumps(rewritten, ensure_ascii=False).encode() + end
        return b"".join(lines)
    return content


def _after_query_blocks(content: str) -> int:
    """Where ``content`` goes on past the query-only blocks it opens with."""
    position = 0
    while (end := _query_block_end(content, position)) is not None:
        position = end
    return position


def _query_block_end(content: str, position: int) -> int | None:
    ends = (_block_end(content, position, tag) for tag in _QUERY_TAGS)
    return next((end for end in ends if end is not None), None)


def _block_end(content: str, position: int, tag: str) -> int | None:
    """Where the ``<tag>`` block opening at ``position`` ends, its blank line
    included, or None when none opens there. It ends at the first closing
    tag of its name."""
    opening, closing = f"<{tag}>\n", f"\n</{tag}>\n\n"
    if not content.startswith(opening, position):
        return None
    end = content.find(closing, position + len(opening))
    return None if end == -1 else end + len(closing)


def _is_rendered(body: str) -> bool:
    """Whether ``body`` is ``<FACTS>``, ``<RECENT_EPISODES>`` or both, in that
    order, each opening on a ``  - `` item, as ``graphiti/context.py``
    rendered them."""
    if body.startswith(_EPISODES_OPEN):
        return body.endswith(_EPISODES_CLOSE)
    if not body.startswith(_FACTS_OPEN):
        return False
    if body.endswith(_FACTS_CLOSE):
        return True
    between = body.find(_BETWEEN_SECTIONS, len(_FACTS_OPEN))
    return between != -1 and body.endswith(_EPISODES_CLOSE)


def _strip_query_text(text: str) -> str:
    stripped = strip_first_turn_memory(text, after_query_blocks=True)
    return text if stripped is None else stripped


def _parse(line: bytes) -> object:
    """A line's JSON value, or None when it is not JSON (or not UTF-8)."""
    try:
        return json.loads(line)
    except ValueError:
        return None
