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
    return (first_name(user.name, user.email) if user else None) or "there"


def first_name(name: str | None, email: str) -> str | None:
    """The first word of a name the user gave, or None.

    Email sign-up stores the address's local part as the name, so a name equal
    to it was never given.
    """
    words = (name or "").split()
    if not words or " ".join(words).lower() == email.split("@")[0].lower():
        return None
    return words[0]
