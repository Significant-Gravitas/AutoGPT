"""Background embedding generation for ``UserWorkspaceFile`` rows.

Mirrors ``backend.api.features.library.embeddings``: fire-and-forget
scheduling so user-facing file writes don't pay the OpenAI embedding
latency. Failures are logged, not raised — a missing embedding only
degrades search quality, it never breaks correctness.

Only the user-visible ``name`` (and the path stem, when it diverges) is
embedded — file *contents* are intentionally out of scope.
"""

from __future__ import annotations

import asyncio
import logging

from prisma.enums import ContentType

from backend.api.features.search.content_handlers import build_workspace_file_text
from backend.api.features.search.embeddings import (
    delete_content_embedding,
    ensure_content_embedding,
    get_content_embedding,
)

logger = logging.getLogger(__name__)

# See ``library/embeddings.py`` for why we hold a strong ref to the task.
_background_tasks: set[asyncio.Task[None]] = set()


async def _run_embedding(file_id: str, user_id: str, name: str, path: str) -> None:
    try:
        searchable_text = build_workspace_file_text(name, path)
        if not searchable_text:
            logger.debug(
                "Skipping workspace file embedding for %s: empty searchable text",
                file_id,
            )
            return
        existing = await get_content_embedding(
            ContentType.WORKSPACE_FILE, file_id, user_id
        )
        if existing and existing.get("searchableText") == searchable_text:
            return
        await ensure_content_embedding(
            content_type=ContentType.WORKSPACE_FILE,
            content_id=file_id,
            searchable_text=searchable_text,
            metadata={"name": name, "path": path},
            user_id=user_id,
            force=True,
        )
    except Exception as e:
        logger.warning(
            "Failed to ensure workspace file embedding for %s: %s", file_id, e
        )


def schedule_workspace_file_embedding(
    file_id: str, user_id: str, name: str, path: str
) -> asyncio.Task[None]:
    """Schedule a fire-and-forget (re-)embed of a workspace file."""
    task = asyncio.create_task(_run_embedding(file_id, user_id, name, path))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


async def delete_workspace_file_embedding(file_id: str, user_id: str) -> None:
    """Best-effort embedding cleanup when a workspace file is deleted."""
    try:
        await delete_content_embedding(
            ContentType.WORKSPACE_FILE, file_id, user_id=user_id
        )
    except Exception as e:
        logger.warning(
            "Failed to delete workspace file embedding for %s: %s", file_id, e
        )
