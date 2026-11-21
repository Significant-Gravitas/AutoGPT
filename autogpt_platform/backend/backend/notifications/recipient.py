"""Best-effort personalization must never delay an account notification."""

import asyncio
import logging

from backend.util.clients import get_database_manager_async_client

logger = logging.getLogger(__name__)


async def greeting_name(user_id: str) -> str:
    try:
        user = await asyncio.wait_for(
            get_database_manager_async_client(should_retry=False).get_user_by_id(
                user_id
            ),
            timeout=2,
        )
    except Exception:
        logger.debug(
            "Recipient name unavailable; using a neutral greeting", exc_info=True
        )
        return "there"
    return (first_name(user.name) if user else None) or "there"


def first_name(name: str | None) -> str | None:
    """The first word of the user's name, or None when they have none."""
    words = (name or "").split()
    return words[0] if words else None
