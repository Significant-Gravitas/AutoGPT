"""Background embedding generation for LibraryAgent rows.

LibraryAgent embeddings power the "similar agents in your library" check
that runs before CoPilot creates a new agent. Generation is fire-and-forget
so user-facing latency on create/update is unaffected; failures are logged
and swallowed because a missing embedding only degrades search quality, it
never breaks correctness.
"""

from __future__ import annotations

import logging

from backend.api.features.search.embeddings import ensure_live_library_content_embedding
from backend.data import graph as graph_db
from backend.util.background import spawn_background_task

logger = logging.getLogger(__name__)


def _build_searchable_text(graph: graph_db.GraphModel) -> str:
    parts = [
        graph.name or "",
        graph.description or "",
        graph.instructions or "",
    ]
    return " ".join(part for part in parts if part).strip()


async def _run_embedding(
    library_agent_id: str,
    user_id: str,
    graph: graph_db.GraphModel,
    organization_id: str | None,
    team_id: str | None,
) -> None:
    try:
        searchable_text = _build_searchable_text(graph)
        if not searchable_text:
            logger.debug(
                "Skipping library agent embedding for %s: empty searchable text",
                library_agent_id,
            )
            return
        await ensure_live_library_content_embedding(
            content_id=library_agent_id,
            user_id=user_id,
            organization_id=organization_id,
            team_id=team_id,
            source_graph_id=graph.id,
            source_graph_version=graph.version,
            searchable_text=searchable_text,
            metadata={"name": graph.name or ""},
        )
    except Exception as e:
        logger.warning(
            "Failed to ensure library agent embedding for %s: %s",
            library_agent_id,
            e,
        )


def schedule_library_agent_embedding(
    library_agent_id: str,
    user_id: str,
    graph: graph_db.GraphModel,
    organization_id: str | None,
    team_id: str | None,
):
    """Schedule a background (re-)embed. No-ops cheaply when the existing
    embedding's searchableText is unchanged. Failures are logged, not raised."""
    return spawn_background_task(
        _run_embedding(
            library_agent_id,
            user_id,
            graph,
            organization_id,
            team_id,
        ),
        name=f"library-agent-embedding:{library_agent_id}",
    )
