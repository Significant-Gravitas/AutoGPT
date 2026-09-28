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
  order, each item a ``  - `` line, and the blank line after the block;
- only when nothing after the block holds a ``<memory_context>`` or
  ``</memory_context>`` tag. ``inject_user_context`` stripped every such tag
  from the user's words before it wrote the message, so one there means the
  text is not the platform's own (a first turn that never reached injection
  keeps the user's raw text), or that stored memory forged an early close.

In a CLI session file only the first user entry is looked at, the first
turn's query: a copy the user pasted into a later message is theirs.

The tag names are literals: they are the ones the old code wrote, and must not
follow a later rename.
"""

import json
import re

from .cli_session_entry import is_user_entry, rewrite_user_entry

# One or more ``  - `` items, as ``graphiti/context.py`` rendered a section.
# An item's text may span lines.
_ITEMS = r"  - .*?"
_FACTS = rf"<FACTS>\n{_ITEMS}\n</FACTS>"
_EPISODES = rf"<RECENT_EPISODES>\n{_ITEMS}\n</RECENT_EPISODES>"
_FIRST_TURN_BLOCK_RE = re.compile(
    r"(?P<query>(?:<(?P<query_tag>skills_update|builder_context|budget_status)>\n"
    r".*?\n</(?P=query_tag)>\n\n)*)"
    r"(?P<skills><available_skills>\n.*?\n</available_skills>\n\n)?"
    r"<memory_context>\n<temporal_context>\n"
    rf"(?:{_FACTS}(?:\n\n{_EPISODES})?|{_EPISODES})"
    r"\n</temporal_context>\n</memory_context>\n\n",
    re.DOTALL,
)
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
    match = _FIRST_TURN_BLOCK_RE.match(content)
    if match is None or (match["query"] and not after_query_blocks):
        return None
    rest = content[match.end() :]
    if _MEMORY_TAG_RE.search(rest):
        return None
    return match["query"] + (match["skills"] or "") + rest


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


def _strip_query_text(text: str) -> str:
    stripped = strip_first_turn_memory(text, after_query_blocks=True)
    return text if stripped is None else stripped


def _parse(line: bytes) -> object:
    """A line's JSON value, or None when it is not JSON (or not UTF-8)."""
    try:
        return json.loads(line)
    except ValueError:
        return None
