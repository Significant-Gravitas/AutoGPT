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
:data:`AFTER_COMPACTION_MAX_PARTS` parts, one per callback, split between
blocks.  When that is still not enough, the lowest-priority blocks are
dropped; a block too big for a part on its own is shortened by whole items.
The blocks come from the first-turn builders as ``ContextBlock``s, items kept
apart from the text around them, so nothing here parses rendered text.
Measured with a stub API against the real CLI 2.1.284.
"""

from collections.abc import Collection, Sequence

from pydantic import BaseModel

from backend.copilot.context_blocks import ContextBlock
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
    """How a block that leaves items out points at the rest."""

    # The tool that lists the items again; without it in the turn's tool set
    # the block is not shortened.
    tool: str
    # The line that replaces the items left out (``{n}`` of them).
    rest: str


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
    ),
    "routines": _ItemList(
        tool="list_routines",
        rest="{n} more are not listed here: call `tool:list_routines` for all of them.",
    ),
    "expert_workflows": _ItemList(
        tool="list_expert_workflows",
        rest=(
            "{n} more are not listed here: call `tool:list_expert_workflows` for "
            "all of them."
        ),
    ),
}


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
    blocks: Sequence[ContextBlock],
    *,
    budget: int = AFTER_COMPACTION_PART_BUDGET,
    max_parts: int = AFTER_COMPACTION_MAX_PARTS,
    hidden_tools: Collection[str] = (),
) -> FittedContext:
    """Split ``lead`` and *blocks* into parts that each stay within ``budget``.

    Parts break only between blocks and keep the blocks' order; ``lead``
    opens the first part.  No blocks gives no parts.
    ``hidden_tools`` are the tool names this turn does not offer the model.
    """
    if not blocks:
        return FittedContext(parts=[], rebuilt_chars=0)
    block_budget = budget - utf16_len(lead)
    kept: list[ContextBlock] = []
    dropped: list[str] = []
    shortened: list[str] = []
    for block in blocks:
        if utf16_len(block.text) <= block_budget:
            kept.append(block)
            continue
        short = shorten_block(block, block_budget, hidden_tools=hidden_tools)
        if short is None:
            dropped.append(block.tag)
            continue
        kept.append(short)
        shortened.append(block.tag)
    parts = _pack(lead, kept, budget)
    while len(parts) > max_parts:
        dropped.append(kept.pop(_lowest_priority(kept)).tag)
        parts = _pack(lead, kept, budget)
    return FittedContext(
        parts=parts,
        rebuilt_chars=len(lead) + sum(len(block.text) for block in blocks),
        dropped=dropped,
        shortened=shortened,
    )


def utf16_len(text: str) -> int:
    """Length as the CLI counts it: JavaScript string length."""
    return len(text.encode("utf-16-le")) // 2


def shorten_block(
    block: ContextBlock, budget: int, *, hidden_tools: Collection[str] = ()
) -> ContextBlock | None:
    """Keep the block's leading items that fit, plus one line saying how to
    get the rest.  ``None`` when the block has no items to drop, the model
    has no tool to list them with, or even its frame does not fit."""
    item_list = _ITEM_LISTS.get(block.tag)
    if item_list is None or item_list.tool in hidden_tools:
        return None
    items = block.items
    for keep in range(len(items) - 1, -1, -1):
        rest = item_list.rest.format(n=len(items) - keep)
        # The pointer to the rest stands where the items left out stood.
        short = block.model_copy(update={"items": (*items[:keep], rest)})
        if utf16_len(short.text) <= budget:
            return short
    return None


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
