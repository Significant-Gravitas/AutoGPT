"""Background embedding generation for LibraryAgent rows.

LibraryAgent embeddings power the "similar agents in your library" check
that runs before CoPilot creates a new agent. Generation is fire-and-forget
so user-facing latency on create/update is unaffected; failures are logged
and swallowed because a missing embedding only degrades search quality, it
never breaks correctness.
"""

from __future__ import annotations

import asyncio
import logging

from prisma.enums import ContentType

from backend.api.features.search.embeddings import (
    ensure_content_embedding,
    get_content_embedding,
)
from backend.data import graph as graph_db

logger = logging.getLogger(__name__)

# ``asyncio.create_task`` only registers the task in a *weak* set on the
# event loop, so a fire-and-forget task with no other strong reference can
# be garbage-collected mid-run. The asyncio docs explicitly warn about
# this and recommend keeping the task object alive elsewhere — see the
# "Important" note at
# https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task.
# Our callers in ``library/db.py`` don't await the task (intentional
# fire-and-forget), so we hold it here and let it remove itself via
# ``add_done_callback`` once it finishes.
_background_tasks: set[asyncio.Task[None]] = set()


def _build_searchable_text(graph: graph_db.GraphModel) -> str:
    parts = [
        graph.name or "",
        graph.description or "",
        graph.instructions or "",
    ]
    return " ".join(part for part in parts if part).strip()


async def _run_embedding(
    library_agent_id: str, user_id: str, graph: graph_db.GraphModel
) -> None:
    try:
        searchable_text = _build_searchable_text(graph)
        if not searchable_text:
            logger.debug(
                "Skipping library agent embedding for %s: empty searchable text",
                library_agent_id,
            )
            return
        # Skip the OpenAI call when the existing embedding's text is byte-
        # identical — e.g. settings-only updates where the version bumped
        # but name/description/instructions did not change.
        existing = await get_content_embedding(
            ContentType.LIBRARY_AGENT, library_agent_id, user_id
        )
        if existing and existing.get("searchableText") == searchable_text:
            logger.debug(
                "Library agent %s embedding text unchanged; skipping refresh",
                library_agent_id,
            )
            return
        await ensure_content_embedding(
            content_type=ContentType.LIBRARY_AGENT,
            content_id=library_agent_id,
            searchable_text=searchable_text,
            metadata={"name": graph.name or ""},
            user_id=user_id,
            force=True,
        )
    except Exception as e:
        logger.warning(
            "Failed to ensure library agent embedding for %s: %s",
            library_agent_id,
            e,
        )


def schedule_library_agent_embedding(
    library_agent_id: str, user_id: str, graph: graph_db.GraphModel
) -> asyncio.Task[None]:
    """Schedule a background (re-)embed. No-ops cheaply when the existing
    embedding's searchableText is unchanged. Failures are logged, not raised."""
    task = asyncio.create_task(_run_embedding(library_agent_id, user_id, graph))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task
