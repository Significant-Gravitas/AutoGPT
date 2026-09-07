"""Assigns a canonical category to listings whose own categories map to none.

Used by the backfill command, so it answers per listing and never raises: an
unclassifiable listing is left uncategorised rather than mislabelled.
"""

import asyncio
import logging
import os

from backend.util.clients import get_openai_client

from .categories import CATEGORY_DESCRIPTIONS, StoreCategory

logger = logging.getLogger(__name__)

# Picking one label off a fixed list of eight is a matching task, not a writing
# one, so it runs on the fast model — same choice as the brain-dump recommender.
_MODEL = os.environ.get("STORE_CATEGORY_MODEL", "anthropic/claude-haiku-4-5")
_TIMEOUT_SECONDS = 30
_MAX_DESCRIPTION_CHARS = 2000

_PROMPT = """Assign the single best category to this marketplace listing for an \
AI agent. Answer with ONLY the category id, lowercase, nothing else.

Categories:
{categories}

If none of them fits, answer exactly: none

Listing name: {name}
Tagline: {sub_heading}
Description: {description}
"""


async def classify_category(
    name: str, sub_heading: str, description: str
) -> StoreCategory | None:
    """The model's category for one listing, or None if it declines or fails."""
    client = get_openai_client(prefer_openrouter=True)
    if client is None:
        logger.warning("Store category classifier: no LLM client configured")
        return None

    prompt = _PROMPT.format(
        categories="\n".join(
            f"- {category.value}: {description}"
            for category, description in CATEGORY_DESCRIPTIONS.items()
        ),
        name=name.strip() or "(none)",
        sub_heading=sub_heading.strip() or "(none)",
        description=description.strip()[:_MAX_DESCRIPTION_CHARS] or "(none)",
    )
    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=16,
            ),
            timeout=_TIMEOUT_SECONDS,
        )
    except Exception as e:
        logger.warning("Store category classification failed for %r: %s", name, e)
        return None

    return _parse_category(response.choices[0].message.content or "")


def _parse_category(content: str) -> StoreCategory | None:
    answer = content.strip().strip(".\"'` ").lower()
    try:
        return StoreCategory(answer)
    except ValueError:
        return None
