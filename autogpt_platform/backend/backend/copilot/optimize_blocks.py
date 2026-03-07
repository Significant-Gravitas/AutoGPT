"""Scheduler job to generate LLM-optimized block descriptions.

Runs periodically to rewrite block descriptions into concise, actionable
summaries that help the copilot LLM pick the right blocks during agent
generation.
"""

import logging
import time

import openai

from backend.copilot.config import ChatConfig
from backend.util.clients import get_database_manager_client

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
# Maximum retries on rate-limit (429) errors
_MAX_RETRIES = 3
# Initial backoff delay for retries (seconds), doubles each retry
_INITIAL_BACKOFF = 2.0


def _call_llm_with_retry(
    client: openai.OpenAI,
    model: str,
    block_name: str,
    description: str,
) -> str | None:
    """Call the LLM with exponential backoff on rate-limit errors."""
    backoff = _INITIAL_BACKOFF
    for attempt in range(_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Block name: {block_name}\n" f"Description: {description}"
                        ),
                    },
                ],
                max_tokens=_MAX_DESCRIPTION_TOKENS,
            )
            return (response.choices[0].message.content or "").strip()
        except openai.RateLimitError:
            if attempt < _MAX_RETRIES - 1:
                logger.warning(
                    "Rate limited on %s, retrying in %.1fs (attempt %d/%d)",
                    block_name,
                    backoff,
                    attempt + 1,
                    _MAX_RETRIES,
                )
                time.sleep(backoff)
                backoff *= 2
            else:
                raise
    return None  # unreachable, but satisfies type checker


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
    # Track newly optimized descriptions to update in-memory block classes
    new_descriptions: dict[str, str] = {}

    for block in blocks:
        block_id = block["id"]
        block_name = block["name"]
        description = block["description"]

        if not description or not description.strip():
            stats["skipped"] += 1
            continue

        stats["processed"] += 1

        try:
            optimized = _call_llm_with_retry(
                client, config.title_model, block_name, description
            )
            if not optimized:
                logger.warning(f"Empty response for block {block_name}")
                stats["failed"] += 1
                continue

            db_client.update_block_optimized_description(block_id, optimized)
            new_descriptions[block_id] = optimized
            stats["success"] += 1
            logger.debug(f"Optimized description for {block_name}")

            # Small delay between API calls to avoid rate limits
            time.sleep(_RATE_LIMIT_DELAY)

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

    # Update in-memory block descriptions first, then invalidate the cache.
    # Updating first ensures that when the cache is rebuilt after reset,
    # it uses the new descriptions instead of stale ones.
    if stats["success"] > 0:
        try:
            from backend.blocks import get_blocks

            block_classes = get_blocks()
            for block_id, optimized in new_descriptions.items():
                if block_id in block_classes:
                    block_classes[block_id]._optimized_description = optimized
            logger.info(
                "Updated %d in-memory block descriptions", len(new_descriptions)
            )
        except Exception:
            logger.debug("Could not update in-memory block descriptions", exc_info=True)

        # Invalidate cache after descriptions are updated so the next
        # agent-gen call rebuilds it with the fresh descriptions.
        from backend.copilot.tools.agent_generator.blocks import _reset_caches

        _reset_caches()

    return stats
