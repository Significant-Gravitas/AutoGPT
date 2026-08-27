"""Cleanup for legacy workspace-file embeddings."""

import logging

from prisma.enums import ContentType

logger = logging.getLogger(__name__)


async def _delete_content_embedding(file_id: str, user_id: str) -> None:
    from backend.api.features.search.embeddings import delete_content_embedding

    await delete_content_embedding(ContentType.WORKSPACE_FILE, file_id, user_id=user_id)


async def delete_workspace_file_embedding(file_id: str, user_id: str) -> None:
    try:
        await _delete_content_embedding(file_id, user_id)
    except Exception as error:
        logger.warning(
            "Failed to delete workspace file embedding for %s: %s", file_id, error
        )
