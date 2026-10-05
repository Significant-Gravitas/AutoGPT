"""Fit the context re-sent after a CLI compaction under the hook-output cap.

The ``SessionStart`` hook (matcher ``compact``) hands the model back the
first-turn blocks the compaction summary dropped.  CLI 2.1.284 caps every
hook callback's ``additionalContext`` at 10,000 characters, counted as
JavaScript string length (UTF-16 code units): one character more and the
model gets a 2,000-character preview plus a file path instead, and that
preview is what the session file keeps and every ``--resume`` replays.  The
threshold is not configurable.

Each callback has its own allowance, and the CLI joins every callback's
context with a newline into one ``SessionStart hook additional context``
reminder, in registration order.  So the rebuilt context goes out as up to
:data:`AFTER_COMPACTION_MAX_PARTS` parts, one per callback, split on block
boundaries.  When that is still not enough, the lowest-priority blocks are
dropped; a block too big for a part on its own is shortened on its own item
boundaries.  Measured with a stub API against the real CLI 2.1.284.
"""

from collections.abc import Sequence

from pydantic import BaseModel

from backend.copilot.service import (
    ENV_CONTEXT_TAG,
    SESSION_CONTEXT_TAG,
    SKILLS_CONTEXT_TAG,
    USER_CONTEXT_TAG,
)

# The CLI cap is 10,000 per callback, measured on the context alone (the
# reminder wrapper is added outside it); the margin covers a CLI that starts
# counting a label or separator against it.
AFTER_COMPACTION_PART_BUDGET = 9_500
# One SessionStart callback per part.  Prod first turns re-send at most
# 22,762 characters (median 5,490, p90 10,328; n=180), which four parts hold
# with room for what packing whole blocks leaves unused.
AFTER_COMPACTION_MAX_PARTS = 4

# Most important first; blocks are dropped from the end of this list.  A tag
# not listed here ranks below all of them.
_PRIORITY = (
    # The session's id and pending follow-ups: tiny, and nothing else says it.
    SESSION_CONTEXT_TAG,
    # The working directory every file tool resolves paths against.
    ENV_CONTEXT_TAG,
    # Who the user is; the summary paraphrases it at best.
    USER_CONTEXT_TAG,
    # What the model matches requests against before acting; shortened by
    # whole entries rather than dropped.
    SKILLS_CONTEXT_TAG,
    # An expert can only run the workflows listed here.
    "expert_workflows",
    # The only place an expert hears about its own machine.
    "expert_computer",
    # Standing work the expert offers; `tool:list_routines` lists it again.
    "routines",
    # The roster; `tool:list_team` lists it again.
    "team_context",
    # General guidance with no per-user data in it.
    "standing_work",
)

# Blocks that can be shortened item by item, with the line that tells the
# model how to get the items left out.
_REST_OF = {
    SKILLS_CONTEXT_TAG: (
        "{n} more skills are not listed here: call `tool:list_skills` for "
        "the full index."
    ),
    "team_context": (
        "{n} more are not listed here: call `tool:list_team` for the full roster."
    ),
    "routines": (
        "{n} more are not listed here: call `tool:list_routines` for all of them."
    ),
    "expert_workflows": (
        "{n} more are not listed here: call `tool:list_expert_workflows` for "
        "all of them."
    ),
}

_ITEM_PREFIX = "- "


class ContextBlock(BaseModel):
    """One top-level ``<tag>…</tag>`` block, with its trailing newlines.

    ``tag`` is ``""`` for text that does not parse as a block.
    """

    tag: str
    text: str

    @property
    def label(self) -> str:
        return self.tag or "untagged text"


class FittedContext(BaseModel):
    """The parts to hand out, one per callback, and what was cut to fit."""

    parts: list[str]
    rebuilt_chars: int
    dropped: list[str] = []
    shortened: list[str] = []

    @property
    def delivered_chars(self) -> int:
        return sum(len(part) for part in self.parts)


def fit_after_compaction(
    lead: str,
    prefix: str,
    *,
    budget: int = AFTER_COMPACTION_PART_BUDGET,
    max_parts: int = AFTER_COMPACTION_MAX_PARTS,
) -> FittedContext:
    """Split ``lead + prefix`` into parts that each stay within ``budget``.

    Parts break only between blocks and keep the blocks' order; ``lead``
    opens the first part.  An empty ``prefix`` gives no parts.
    """
    if not prefix:
        return FittedContext(parts=[], rebuilt_chars=0)
    block_budget = budget - utf16_len(lead)
    blocks: list[ContextBlock] = []
    dropped: list[str] = []
    shortened: list[str] = []
    for block in split_blocks(prefix):
        if utf16_len(block.text) <= block_budget:
            blocks.append(block)
            continue
        short = shorten_block(block, block_budget)
        if short is None:
            dropped.append(block.label)
            continue
        blocks.append(short)
        shortened.append(block.label)
    parts = _pack(lead, blocks, budget)
    while len(parts) > max_parts:
        dropped.append(blocks.pop(_lowest_priority(blocks)).label)
        parts = _pack(lead, blocks, budget)
    return FittedContext(
        parts=parts,
        rebuilt_chars=len(lead) + len(prefix),
        dropped=dropped,
        shortened=shortened,
    )


def utf16_len(text: str) -> int:
    """Length as the CLI counts it: JavaScript string length."""
    return len(text.encode("utf-16-le")) // 2


def split_blocks(prefix: str) -> list[ContextBlock]:
    """Split the rendered prefix into its top-level blocks, in order.

    Every first-turn builder renders ``<tag>\\n…\\n</tag>`` followed by
    blank lines; anything from the first spot that does not parse that way
    is kept whole as one untagged block.
    """
    blocks: list[ContextBlock] = []
    pos = 0
    while pos < len(prefix):
        end, tag = _block_end(prefix, pos)
        if end < 0:
            blocks.append(ContextBlock(tag="", text=prefix[pos:]))
            break
        blocks.append(ContextBlock(tag=tag, text=prefix[pos:end]))
        pos = end
    return blocks


def shorten_block(block: ContextBlock, budget: int) -> ContextBlock | None:
    """Keep the block's leading items that fit, plus one line saying how to
    get the rest.  ``None`` when the block has no items to drop or even its
    frame does not fit."""
    rest_of = _REST_OF.get(block.tag)
    if rest_of is None:
        return None
    opening = f"<{block.tag}>\n"
    closing_at = block.text.rfind(f"\n</{block.tag}>")
    head, items, tail = _items(block.text[len(opening) : closing_at])
    after = block.text[closing_at:]
    for keep in range(len(items) - 1, -1, -1):
        lines = [*head, *items[:keep], rest_of.format(n=len(items) - keep), *tail]
        text = opening + "\n".join(lines) + after
        if utf16_len(text) <= budget:
            return ContextBlock(tag=block.tag, text=text)
    return None


def _block_end(prefix: str, pos: int) -> tuple[int, str]:
    """End of the block that starts at ``pos`` (past its trailing newlines)
    and its tag, or ``(-1, "")`` when no block starts there."""
    if not prefix.startswith("<", pos):
        return -1, ""
    tag_end = prefix.find(">\n", pos)
    tag = prefix[pos + 1 : tag_end] if tag_end > 0 else ""
    if not tag or not tag.replace("_", "").isalnum():
        return -1, ""
    closing = f"\n</{tag}>"
    close_at = prefix.find(closing, tag_end)
    if close_at < 0:
        return -1, ""
    end = close_at + len(closing)
    while prefix.startswith("\n", end):
        end += 1
    return end, tag


def _items(body: str) -> tuple[list[str], list[str], list[str]]:
    """``(head, items, tail)`` lines of a block body.

    An item starts at a line beginning ``- `` and takes the lines after it up
    to the next item; lines before the first item are the head, lines after
    the last item line are the tail.
    """
    lines = body.split("\n")
    starts = [i for i, line in enumerate(lines) if line.startswith(_ITEM_PREFIX)]
    if not starts:
        return lines, [], []
    bounds = [*starts[1:], starts[-1] + 1]
    items = ["\n".join(lines[s:e]) for s, e in zip(starts, bounds)]
    return lines[: starts[0]], items, lines[starts[-1] + 1 :]


def _pack(lead: str, blocks: Sequence[ContextBlock], budget: int) -> list[str]:
    parts = [lead]
    for block in blocks:
        if utf16_len(parts[-1]) + utf16_len(block.text) <= budget:
            parts[-1] += block.text
        else:
            parts.append(block.text)
    return parts if blocks else []


def _lowest_priority(blocks: Sequence[ContextBlock]) -> int:
    """Index of the block to drop next; the last of equal rank goes first."""

    def rank(index: int) -> tuple[int, int]:
        tag = blocks[index].tag
        return (_PRIORITY.index(tag) if tag in _PRIORITY else len(_PRIORITY), index)

    return max(range(len(blocks)), key=rank)
