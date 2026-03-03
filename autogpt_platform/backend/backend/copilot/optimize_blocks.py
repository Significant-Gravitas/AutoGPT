"""Scheduler job to generate LLM-optimized block descriptions.

Runs periodically to rewrite block descriptions into concise, actionable
summaries that help the copilot LLM pick the right blocks during agent
generation.
"""

import logging
import time

import openai

from backend.copilot.config import ChatConfig
from backend.copilot.tools.agent_generator.blocks import _reset_caches
from backend.util.clients import get_database_manager_client

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a technical writer for an automation platform. "
    "Rewrite the following block description to be concise (under 50 words), "
    "informative, and actionable. Focus on what the block does and when to "
    "use it. Output ONLY the rewritten description, nothing else. "
    "Do not use markdown formatting."
)


def optimize_block_descriptions() -> dict[str, int]:
    """Generate optimized descriptions for blocks that don't have one yet.

    Uses the copilot LLM (via OpenRouter) to rewrite block descriptions
    into concise summaries suitable for agent generation prompts.

    Returns:
        Dict with counts: processed, success, failed, skipped.
    """
    config = ChatConfig()
    db_client = get_database_manager_client()

    # Get blocks that need optimization
    blocks = db_client.get_blocks_needing_optimization()
    if not blocks:
        logger.info("All blocks already have optimized descriptions")
        return {"processed": 0, "success": 0, "failed": 0, "skipped": 0}

    logger.info(f"Found {len(blocks)} blocks needing optimized descriptions")

    # Create sync OpenAI client with copilot config
    client = openai.OpenAI(api_key=config.api_key, base_url=config.base_url)

    stats = {"processed": 0, "success": 0, "failed": 0, "skipped": 0}

    for block in blocks:
        block_id = block["id"]
        block_name = block["name"]
        description = block["description"]

        if not description or not description.strip():
            stats["skipped"] += 1
            continue

        stats["processed"] += 1

        try:
            response = client.chat.completions.create(
                model=config.title_model,  # Use fast/cheap model
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Block name: {block_name}\n" f"Description: {description}"
                        ),
                    },
                ],
                max_tokens=150,
            )

            optimized = (response.choices[0].message.content or "").strip()
            if not optimized:
                logger.warning(f"Empty response for block {block_name}")
                stats["failed"] += 1
                continue

            db_client.update_block_optimized_description(block_id, optimized)
            stats["success"] += 1
            logger.debug(f"Optimized description for {block_name}")

            # Small delay between API calls to avoid rate limits
            time.sleep(0.5)

        except Exception:
            logger.warning(
                f"Failed to optimize description for {block_name}",
                exc_info=True,
            )
            stats["failed"] += 1

    logger.info(
        f"Block description optimization complete: "
        f"{stats['success']}/{stats['processed']} succeeded, "
        f"{stats['failed']} failed, {stats['skipped']} skipped"
    )

    # Invalidate the block cache so the next agent-gen call picks up
    # the new optimized descriptions without requiring a process restart.
    if stats["success"] > 0:
        _reset_caches()
        logger.info(
            "Block cache invalidated — fresh descriptions will load on next use"
        )

    return stats
