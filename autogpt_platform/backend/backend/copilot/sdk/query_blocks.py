"""Read the platform's context blocks off the front of a turn's query string.

The SDK path puts tagged blocks (``<user_context>``, ``<conversation_history>``,
``<budget_status>`` and so on) in front of the user's words before handing the
query to the CLI. :func:`scan_context_blocks` reports which blocks are there and
how big each one is, for the turn-summary line. Names and sizes only: it never
returns block content.
"""

import re

from backend.copilot.service import SERVER_INJECTED_BLOCK_TAGS

CONTEXT_BLOCK_TAGS = frozenset(SERVER_INJECTED_BLOCK_TAGS)
# What ``_build_query_message`` writes between the history block and the turn.
HISTORY_LEAD_OUT = "Now, the user says:\n"
_OPEN_TAG_RE = re.compile(r"<([A-Za-z_][A-Za-z0-9_]*)>")


def scan_context_blocks(query: str) -> tuple[list[tuple[str, int]], int]:
    """The platform blocks leading *query*, as ``(tag, chars)``, and the prefix size.

    Walks known block tags from the start of the string, stepping over the
    whitespace between them and the history lead-out, and stops at the first
    thing that is not one: the user's own words. The prefix size runs from the
    start of the query to the end of the last block or lead-out found.
    """
    blocks: list[tuple[str, int]] = []
    pos = 0
    while True:
        start = _skip_space(query, pos)
        if query.startswith(HISTORY_LEAD_OUT, start):
            pos = start + len(HISTORY_LEAD_OUT)
            continue
        match = _OPEN_TAG_RE.match(query, start)
        if match is None or match.group(1) not in CONTEXT_BLOCK_TAGS:
            return blocks, pos
        close = f"</{match.group(1)}>"
        end = query.find(close, match.end())
        if end == -1:
            return blocks, pos
        pos = end + len(close)
        blocks.append((match.group(1), pos - start))


def _skip_space(text: str, pos: int) -> int:
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos
