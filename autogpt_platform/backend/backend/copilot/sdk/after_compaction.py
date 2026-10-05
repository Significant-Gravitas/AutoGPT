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
reminder, in the order the callbacks answer (``after_compaction_handout``
makes that the part order).  So the rebuilt context goes out as up to
:data:`AFTER_COMPACTION_MAX_PARTS` parts, one per callback, split on block
boundaries.  When that is still not enough, the lowest-priority blocks are
dropped; a block too big for a part on its own is shortened on its own item
boundaries.  Measured with a stub API against the real CLI 2.1.284.
"""

from collections.abc import Collection, Sequence

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


class _ItemList(BaseModel):
    """A block that lists one item per ``- `` line and can lose items."""

    # The tool that lists the items again; without it in the turn's tool set
    # the block is not shortened.
    tool: str
    # The line that replaces the items left out (``{n}`` of them).
    rest: str
    # How the line after the last item starts when the renderer adds one.
    # Lines after the last item that do not start this way belong to that
    # item (a description with a line break in it).
    closing_starts: tuple[str, ...] = ()
    # Shortened only when the body opens with one of these; any body when
    # empty.
    opens_with: tuple[str, ...] = ()


_ITEM_LISTS = {
    SKILLS_CONTEXT_TAG: _ItemList(
        tool="list_skills",
        rest=(
            "{n} more skills are not listed here: call `tool:list_skills` for "
            "the full index."
        ),
    ),
    "team_context": _ItemList(
        tool="list_team",
        rest="{n} more are not listed here: call `tool:list_team` for the full roster.",
        closing_starts=("When a request clearly matches", "These are your teammates."),
        # Not the hiring form shown to a user with nobody hired: its items
        # are templates, which `list_team` does not list.
        opens_with=(
            "The user has hired these experts:",
            "Your teammates on this user's team:",
        ),
    ),
    "routines": _ItemList(
        tool="list_routines",
        rest="{n} more are not listed here: call `tool:list_routines` for all of them.",
        closing_starts=("The routines marked (proposal)",),
    ),
    "expert_workflows": _ItemList(
        tool="list_expert_workflows",
        rest=(
            "{n} more are not listed here: call `tool:list_expert_workflows` for "
            "all of them."
        ),
        closing_starts=("If the user skips or declines",),
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
    hidden_tools: Collection[str] = (),
) -> FittedContext:
    """Split ``lead + prefix`` into parts that each stay within ``budget``.

    Parts break only between blocks and keep the blocks' order; ``lead``
    opens the first part.  An empty ``prefix`` gives no parts.
    ``hidden_tools`` are the tool names this turn does not offer the model.
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
        short = shorten_block(block, block_budget, hidden_tools=hidden_tools)
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


def shorten_block(
    block: ContextBlock, budget: int, *, hidden_tools: Collection[str] = ()
) -> ContextBlock | None:
    """Keep the block's leading items that fit, plus one line saying how to
    get the rest.  ``None`` when the block has no items to drop, the model
    has no tool to list them with, or even its frame does not fit."""
    item_list = _ITEM_LISTS.get(block.tag)
    if item_list is None or item_list.tool in hidden_tools:
        return None
    opening = f"<{block.tag}>\n"
    closing_at = block.text.rfind(f"\n</{block.tag}>")
    body = block.text[len(opening) : closing_at]
    if item_list.opens_with and not body.startswith(item_list.opens_with):
        return None
    head, items, tail = _items(body, item_list.closing_starts)
    after = block.text[closing_at:]
    for keep in range(len(items) - 1, -1, -1):
        rest = item_list.rest.format(n=len(items) - keep)
        text = opening + "\n".join([*head, *items[:keep], rest, *tail]) + after
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


def _items(
    body: str, closing_starts: tuple[str, ...]
) -> tuple[list[str], list[str], list[str]]:
    """``(head, items, tail)`` lines of a block body.

    An item starts at a line beginning ``- `` and takes the lines after it up
    to the next item; lines before the first item are the head.  The tail
    starts at the first line after the last item that begins with one of
    ``closing_starts``; without such a line there is no tail.
    """
    lines = body.split("\n")
    starts = [i for i, line in enumerate(lines) if line.startswith(_ITEM_PREFIX)]
    if not starts:
        return lines, [], []
    tail_at = next(
        (
            i
            for i in range(starts[-1] + 1, len(lines))
            if lines[i].startswith(closing_starts)
        ),
        len(lines),
    )
    bounds = [*starts[1:], tail_at]
    items = ["\n".join(lines[s:e]) for s, e in zip(starts, bounds)]
    return lines[: starts[0]], items, lines[tail_at:]


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
