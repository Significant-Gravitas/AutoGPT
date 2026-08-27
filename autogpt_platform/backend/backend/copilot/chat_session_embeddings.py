"""Cleanup for legacy chat-session title embeddings."""

import logging

from prisma.enums import ContentType

logger = logging.getLogger(__name__)


async def _delete_content_embedding(session_id: str, user_id: str) -> None:
    from backend.api.features.search.embeddings import delete_content_embedding

    await delete_content_embedding(
        ContentType.CHAT_SESSION, session_id, user_id=user_id
    )


async def delete_chat_session_embedding(session_id: str, user_id: str) -> None:
    try:
        await _delete_content_embedding(session_id, user_id)
    except Exception as error:
        logger.warning(
            "Failed to delete chat session embedding for %s: %s", session_id, error
        )
