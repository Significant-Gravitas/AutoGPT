from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.chat_session_embeddings import delete_chat_session_embedding


@pytest.mark.asyncio
async def test_delete_chat_session_embedding_cleans_legacy_row():
    with patch(
        "backend.copilot.chat_session_embeddings._delete_content_embedding",
        new_callable=AsyncMock,
    ) as delete:
        await delete_chat_session_embedding("session-1", "user-1")

    delete.assert_awaited_once_with("session-1", "user-1")


@pytest.mark.asyncio
async def test_delete_chat_session_embedding_is_best_effort():
    with patch(
        "backend.copilot.chat_session_embeddings._delete_content_embedding",
        new_callable=AsyncMock,
        side_effect=RuntimeError("offline"),
    ):
        await delete_chat_session_embedding("session-1", "user-1")
