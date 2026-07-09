"""Background embedding generation for ``ChatSession`` titles.

Mirrors ``backend.api.features.library.embeddings``: fire-and-forget so
title updates stay fast. Only the (user-set or auto-generated) title is
embedded — message bodies are intentionally out of scope here.
"""

from __future__ import annotations

import asyncio
import logging

from prisma.enums import ContentType

from backend.api.features.search.embeddings import (
    delete_content_embedding,
    ensure_content_embedding,
    get_content_embedding,
)

logger = logging.getLogger(__name__)

# See ``library/embeddings.py`` for why we hold a strong ref to the task.
_background_tasks: set[asyncio.Task[None]] = set()


async def _run_embedding(session_id: str, user_id: str, title: str) -> None:
    try:
        searchable_text = (title or "").strip()
        if not searchable_text:
            # Nothing to search on — drop any stale row so renaming back
            # to "untitled" doesn't keep the old title indexed.
            await delete_content_embedding(
                ContentType.CHAT_SESSION, session_id, user_id=user_id
            )
            return
        existing = await get_content_embedding(
            ContentType.CHAT_SESSION, session_id, user_id
        )
        if existing and existing.get("searchableText") == searchable_text:
            return
        await ensure_content_embedding(
            content_type=ContentType.CHAT_SESSION,
            content_id=session_id,
            searchable_text=searchable_text,
            metadata={"title": searchable_text},
            user_id=user_id,
            force=True,
        )
    except Exception as e:
        logger.warning(
            "Failed to ensure chat session embedding for %s: %s", session_id, e
        )


def schedule_chat_session_embedding(
    session_id: str, user_id: str, title: str | None
) -> asyncio.Task[None]:
    """Schedule a fire-and-forget (re-)embed of a chat session title."""
    task = asyncio.create_task(_run_embedding(session_id, user_id, title or ""))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


async def delete_chat_session_embedding(session_id: str, user_id: str) -> None:
    """Best-effort embedding cleanup when a chat session is deleted."""
    try:
        await delete_content_embedding(
            ContentType.CHAT_SESSION, session_id, user_id=user_id
        )
    except Exception as e:
        logger.warning(
            "Failed to delete chat session embedding for %s: %s", session_id, e
        )
