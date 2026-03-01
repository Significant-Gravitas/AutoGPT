"""Block management and recommendation for agent generation.

Provides cached access to block metadata and keyword-based block recommendation
for selecting relevant blocks given a user's goal. No LLM calls are used;
recommendation is done via fast keyword matching.
"""

import logging
import re
from typing import Any, Type

from backend.blocks import get_blocks as get_block_classes
from backend.blocks._base import Block, BlockCategory

logger = logging.getLogger(__name__)

# Re-export BlockCategory for convenience
__all__ = ["BlockCategory", "_reset_caches"]

# ---------------------------------------------------------------------------
# Module-level caches
# ---------------------------------------------------------------------------
_blocks_cache: list[dict[str, Any]] | None = None
_summaries_cache: list[dict[str, str]] | None = None


def _reset_caches() -> None:
    """Reset all module-level caches (useful for testing)."""
    global _blocks_cache, _summaries_cache
    _blocks_cache = None
    _summaries_cache = None


# ---------------------------------------------------------------------------
# 1. get_blocks_as_dicts
# ---------------------------------------------------------------------------


def get_blocks_as_dicts() -> list[dict[str, Any]]:
    """Get all available blocks as dicts (cached after first call).

    Each dict contains the keys returned by ``Block.get_info().model_dump()``:
    id, name, description, inputSchema, outputSchema, categories,
    staticOutput, costs, contributors, uiType.

    Returns:
        List of block info dicts.
    """
    global _blocks_cache
    if _blocks_cache is not None:
        return _blocks_cache

    block_classes: dict[str, Type[Block]] = get_block_classes()  # type: ignore[assignment]
    blocks: list[dict[str, Any]] = []
    for block_cls in block_classes.values():
        try:
            instance = block_cls()
            info = instance.get_info().model_dump()
            blocks.append(info)
        except Exception:
            logger.warning(
                "Failed to load block info for %s, skipping",
                getattr(block_cls, "__name__", "unknown"),
                exc_info=True,
            )

    _blocks_cache = blocks
    logger.info("Cached %d block dicts", len(blocks))
    return _blocks_cache


# ---------------------------------------------------------------------------
# 2. get_block_summaries
# ---------------------------------------------------------------------------

_MAX_DESCRIPTION_LENGTH = 150


def get_block_summaries() -> list[dict[str, str]]:
    """Get concise block summaries suitable for LLM context (cached).

    Each summary dict has:
        - id: block UUID
        - name: block name
        - description: truncated to 150 chars
        - categories: list of category names (e.g. ["BASIC", "AI"])
        - input_fields: list of top-level input field names
        - output_fields: list of top-level output field names

    Returns:
        List of summary dicts.
    """
    global _summaries_cache
    if _summaries_cache is not None:
        return _summaries_cache

    all_blocks = get_blocks_as_dicts()
    summaries: list[dict[str, Any]] = []

    for block in all_blocks:
        description = block.get("description", "") or ""
        if len(description) > _MAX_DESCRIPTION_LENGTH:
            description = description[: _MAX_DESCRIPTION_LENGTH - 3] + "..."

        categories = [cat.get("category", "") for cat in block.get("categories", [])]

        input_fields = list(block.get("inputSchema", {}).get("properties", {}).keys())
        output_fields = list(block.get("outputSchema", {}).get("properties", {}).keys())

        summaries.append(
            {
                "id": block["id"],
                "name": block["name"],
                "description": description,
                "categories": categories,
                "input_fields": input_fields,
                "output_fields": output_fields,
            }
        )

    _summaries_cache = summaries
    logger.info("Cached %d block summaries", len(summaries))
    return _summaries_cache


# ---------------------------------------------------------------------------
# 3. recommend_blocks_for_goal
# ---------------------------------------------------------------------------

# Words that are too common to be useful for scoring
_STOP_WORDS: set[str] = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "shall",
    "can",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "as",
    "into",
    "about",
    "that",
    "this",
    "it",
    "its",
    "and",
    "or",
    "but",
    "not",
    "if",
    "then",
    "than",
    "so",
    "no",
    "up",
    "out",
    "i",
    "my",
    "me",
    "we",
    "our",
    "you",
    "your",
}


def _tokenize(text: str) -> set[str]:
    """Lowercase and split text into word tokens, removing stop words."""
    words = set(re.findall(r"[a-z0-9]+", text.lower()))
    return words - _STOP_WORDS


def _score_block(
    block: dict[str, Any],
    goal_tokens: set[str],
) -> float:
    """Score a block's relevance to the goal based on keyword overlap.

    Name matches are weighted more heavily than description matches.
    """
    name = block.get("name", "")
    description = block.get("description", "") or ""

    name_tokens = _tokenize(name)
    desc_tokens = _tokenize(description)

    # Name overlap is worth 3x description overlap
    name_overlap = len(goal_tokens & name_tokens)
    desc_overlap = len(goal_tokens & desc_tokens)

    return name_overlap * 3.0 + desc_overlap * 1.0


def recommend_blocks_for_goal(
    goal: str,
    max_blocks: int = 25,
) -> list[dict[str, Any]]:
    """Recommend relevant blocks for a goal using keyword matching.

    Scoring approach:
    - Tokenise the goal and each block's name + description.
    - Score = 3 * (goal-token overlap with name) + 1 * (overlap with description).
    - All blocks in the BASIC category are always included.
    - Return the top ``max_blocks`` blocks sorted by relevance_score (desc).

    Each returned dict is a *full* block dict (with inputSchema, outputSchema,
    etc.) plus an additional ``relevance_score`` field.

    Args:
        goal: Natural-language description of what the user wants to build.
        max_blocks: Maximum number of blocks to return (default 25).

    Returns:
        List of block dicts with ``relevance_score`` added, sorted descending.
    """
    all_blocks = get_blocks_as_dicts()
    goal_tokens = _tokenize(goal)

    scored: list[tuple[float, dict[str, Any]]] = []
    basic_blocks: list[dict[str, Any]] = []

    for block in all_blocks:
        categories = [cat.get("category", "") for cat in block.get("categories", [])]
        is_basic = BlockCategory.BASIC.name in categories

        score = _score_block(block, goal_tokens)

        if is_basic:
            basic_blocks.append({**block, "relevance_score": score})
        else:
            scored.append((score, block))

    # Sort non-basic blocks by score descending
    scored.sort(key=lambda pair: pair[0], reverse=True)

    # Build result: basic blocks first (capped), then top scored blocks
    if len(basic_blocks) >= max_blocks:
        return basic_blocks[:max_blocks]

    basic_ids = {b["id"] for b in basic_blocks}
    result = list(basic_blocks)

    for score, block in scored:
        if block["id"] in basic_ids:
            continue
        if len(result) >= max_blocks:
            break
        result.append({**block, "relevance_score": score})

    return result


# ---------------------------------------------------------------------------
# 4. get_block_by_id
# ---------------------------------------------------------------------------


def get_block_by_id(block_id: str) -> dict[str, Any] | None:
    """Look up a single block by its UUID.

    Args:
        block_id: The block's unique identifier.

    Returns:
        Block info dict or ``None`` if not found.
    """
    for block in get_blocks_as_dicts():
        if block["id"] == block_id:
            return block
    return None
