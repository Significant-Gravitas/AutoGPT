"""Scheduler job to generate LLM-optimized block descriptions.

Runs periodically to rewrite block descriptions into concise, actionable
summaries that help the copilot LLM pick the right blocks during agent
generation.
"""

import asyncio
import logging

from backend.blocks import get_blocks
from backend.util.clients import get_database_manager_client, get_openai_client

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a technical writer for an automation platform. "
    "Rewrite the following block description to be concise (under 50 words), "
    "informative, and actionable. Focus on what the block does and when to "
    "use it. Output ONLY the rewritten description, nothing else. "
    "Do not use markdown formatting."
)

# Rate-limit delay between sequential LLM calls (seconds)
_RATE_LIMIT_DELAY = 0.5
# Maximum tokens for optimized description generation
_MAX_DESCRIPTION_TOKENS = 150
# Model for generating optimized descriptions (fast, cheap)
_MODEL = "gpt-4o-mini"


async def _optimize_descriptions(blocks: list[dict[str, str]]) -> dict[str, str]:
    """Call the shared OpenAI client to rewrite each block description."""
    client = get_openai_client()
    if client is None:
        logger.error(
            "No OpenAI client configured, skipping block description optimization"
        )
        return {}

    results: dict[str, str] = {}
    for block in blocks:
        block_id = block["id"]
        block_name = block["name"]
        description = block["description"]

        try:
            response = await client.chat.completions.create(
                model=_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Block name: {block_name}\nDescription: {description}",
                    },
                ],
                max_tokens=_MAX_DESCRIPTION_TOKENS,
            )
            optimized = (response.choices[0].message.content or "").strip()
            if optimized:
                results[block_id] = optimized
                logger.debug("Optimized description for %s", block_name)
            else:
                logger.warning("Empty response for block %s", block_name)
        except Exception:
            logger.warning(
                "Failed to optimize description for %s", block_name, exc_info=True
            )

        await asyncio.sleep(_RATE_LIMIT_DELAY)

    return results


def optimize_block_descriptions() -> dict[str, int]:
    """Generate optimized descriptions for blocks that don't have one yet.

    Uses the shared OpenAI client to rewrite block descriptions into concise
    summaries suitable for agent generation prompts.

    Returns:
        Dict with counts: processed, success, failed, skipped.
    """
    db_client = get_database_manager_client()

    blocks = db_client.get_blocks_needing_optimization()
    if not blocks:
        logger.info("All blocks already have optimized descriptions")
        return {"processed": 0, "success": 0, "failed": 0, "skipped": 0}

    logger.info("Found %d blocks needing optimized descriptions", len(blocks))

    non_empty = [b for b in blocks if b.get("description", "").strip()]
    skipped = len(blocks) - len(non_empty)

    new_descriptions = asyncio.run(_optimize_descriptions(non_empty))

    stats = {
        "processed": len(non_empty),
        "success": len(new_descriptions),
        "failed": len(non_empty) - len(new_descriptions),
        "skipped": skipped,
    }

    logger.info(
        "Block description optimization complete: "
        "%d/%d succeeded, %d failed, %d skipped",
        stats["success"],
        stats["processed"],
        stats["failed"],
        stats["skipped"],
    )

    if new_descriptions:
        for block_id, optimized in new_descriptions.items():
            db_client.update_block_optimized_description(block_id, optimized)

        # Update in-memory descriptions first so the cache rebuilds with fresh data.
        try:
            block_classes = get_blocks()
            for block_id, optimized in new_descriptions.items():
                if block_id in block_classes:
                    block_classes[block_id]._optimized_description = optimized
            logger.info(
                "Updated %d in-memory block descriptions", len(new_descriptions)
            )
        except Exception:
            logger.warning(
                "Could not update in-memory block descriptions", exc_info=True
            )

        from backend.copilot.tools.agent_generator.blocks import (
            reset_block_caches,  # local to avoid circular import
        )

        reset_block_caches()

    return stats
