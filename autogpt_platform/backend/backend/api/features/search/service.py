"""Service layer for the unified ``/api/search/global`` endpoint.

For a non-empty query, ``global_search`` fan-outs to
``unified_hybrid_search`` three times in parallel (once per bucket) so a
slow embedding/lexical call for one content type doesn't stall the
others.

For an empty/whitespace query, ``global_search`` returns the
most-recently-updated items per bucket. That branch uses three plain DB
queries — no embedding needed since it's purely
``ORDER BY updatedAt DESC LIMIT N`` — and the result is cached per-user
with a short TTL because the UI surfaces this listing on every
page-load.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from prisma.enums import ContentType

from backend.api.features.search.model import GlobalSearchResponse, SearchResultItem
from backend.data.db_accessors import library_db, search
from backend.util.cache import cached

logger = logging.getLogger(__name__)


# ----- helpers ---------------------------------------------------------------


def _hybrid_row_to_item(row: dict[str, Any]) -> SearchResultItem | None:
    """Convert one ``unified_hybrid_search`` row into a ``SearchResultItem``.

    Returns None when the row's content type is one we don't surface
    through /search/global (e.g. BLOCK / DOCUMENTATION rows that might
    sneak in if a caller widens the type filter).
    """
    raw_type = row.get("content_type")
    # Both Prisma enum members and bare strings can show up depending on
    # whether the search ran via the in-process accessor or the RPC shim.
    type_str = getattr(raw_type, "value", raw_type)
    metadata = row.get("metadata") or {}

    if type_str == ContentType.LIBRARY_AGENT.value:
        title = metadata.get("name") or row.get("searchable_text") or ""
        item_type: Any = "library_agent"
        subtitle = None
    elif type_str == ContentType.STORE_AGENT.value:
        title = metadata.get("name") or row.get("searchable_text") or ""
        item_type = "store_agent"
        cats = metadata.get("categories") or []
        subtitle = ", ".join(cats[:2]) if cats else None
    elif type_str == ContentType.WORKSPACE_FILE.value:
        title = metadata.get("name") or row.get("searchable_text") or ""
        item_type = "workspace_file"
        subtitle = metadata.get("mime_type") or metadata.get("path")
    elif type_str == ContentType.CHAT_SESSION.value:
        title = metadata.get("title") or row.get("searchable_text") or ""
        item_type = "chat_session"
        subtitle = None
    else:
        return None

    score = row.get("relevance")
    if score is None:
        score = row.get("combined_score")

    return SearchResultItem(
        id=row.get("content_id") or "",
        type=item_type,
        title=title,
        subtitle=subtitle,
        metadata=metadata if isinstance(metadata, dict) else {},
        score=float(score) if score is not None else None,
        updated_at=row.get("updated_at"),
    )


async def _search_bucket(
    query: str,
    user_id: str,
    content_types: list[ContentType],
    limit: int,
) -> list[SearchResultItem]:
    """Run unified_hybrid_search for one bucket; swallow failures so a
    broken bucket doesn't 500 the whole response."""
    try:
        rows, _total = await search().unified_hybrid_search(
            query=query,
            content_types=content_types,
            user_id=user_id,
            page=1,
            page_size=limit,
            # The unified default (0.15) is calibrated for relevance pages;
            # use a slightly more permissive floor for the global top-N so
            # we still return *something* on short queries.
            min_score=0.05,
        )
    except Exception as e:
        logger.warning(
            "Hybrid search failed for bucket %s: %s",
            [ct.value for ct in content_types],
            e,
        )
        return []

    items: list[SearchResultItem] = []
    for row in rows:
        item = _hybrid_row_to_item(row)
        if item is not None:
            items.append(item)
        if len(items) >= limit:
            break
    return items


# ----- recent (empty-query) buckets ------------------------------------------


async def _recent_agents(user_id: str, limit: int) -> list[SearchResultItem]:
    """Most-recently-updated library agents for the user."""
    # Local import — library.model pulls in a heavy graph chain that we
    # don't want to load at search-module import time.
    from backend.api.features.library import model as library_model

    try:
        resp = await library_db().list_library_agents(
            user_id=user_id,
            page=1,
            page_size=limit,
            sort_by=library_model.LibraryAgentSort.UPDATED_AT,
        )
    except Exception as e:
        logger.warning("Failed to list recent library agents for %s: %s", user_id, e)
        return []

    items: list[SearchResultItem] = []
    for agent in resp.agents[:limit]:
        items.append(
            SearchResultItem(
                id=agent.id,
                type="library_agent",
                title=agent.name,
                subtitle=agent.description or None,
                metadata={
                    "graph_id": agent.graph_id,
                    "image_url": agent.image_url,
                    "is_favorite": agent.is_favorite,
                },
                updated_at=agent.updated_at,
            )
        )
    return items


async def _recent_files(user_id: str, limit: int) -> list[SearchResultItem]:
    """Most-recently-updated workspace files across all sessions."""
    from backend.data.workspace import get_workspace
    from backend.util.workspace import WorkspaceManager

    try:
        workspace = await get_workspace(user_id)
        if workspace is None:
            # User has no workspace yet — nothing to list, no need to
            # create one just for the recents view.
            return []
        manager = WorkspaceManager(user_id, workspace.id, session_id=None)
        files = await manager.list_files(limit=limit, include_all_sessions=True)
    except Exception as e:
        logger.warning("Failed to list recent workspace files for %s: %s", user_id, e)
        return []

    items: list[SearchResultItem] = []
    for file in files[:limit]:
        items.append(
            SearchResultItem(
                id=file.id,
                type="workspace_file",
                title=file.name,
                subtitle=file.mime_type or file.path,
                metadata={
                    "path": file.path,
                    "mime_type": file.mime_type,
                    "size_bytes": file.size_bytes,
                },
                updated_at=file.updated_at,
            )
        )
    return items


async def _recent_chats(user_id: str, limit: int) -> list[SearchResultItem]:
    """Most-recently-updated chat sessions for the user."""
    from backend.copilot.model import get_user_sessions

    try:
        sessions, _total = await get_user_sessions(
            user_id=user_id, limit=limit, offset=0
        )
    except Exception as e:
        logger.warning("Failed to list recent chat sessions for %s: %s", user_id, e)
        return []

    items: list[SearchResultItem] = []
    for session in sessions[:limit]:
        items.append(
            SearchResultItem(
                id=session.session_id,
                type="chat_session",
                title=session.title or "Untitled chat",
                metadata={
                    "chat_status": session.chat_status,
                },
                updated_at=session.updated_at,
            )
        )
    return items


# 60-second TTL: short enough that freshly-created items show up
# promptly, long enough to absorb rapid page-load chatter from the
# sidebar. Redis-backed so the cache survives across worker processes.
@cached(ttl_seconds=60, shared_cache=True, cache_none=False)
async def _cached_recent_buckets(user_id: str, limit: int) -> GlobalSearchResponse:
    agents, files, chats = await asyncio.gather(
        _recent_agents(user_id, limit),
        _recent_files(user_id, limit),
        _recent_chats(user_id, limit),
    )
    return GlobalSearchResponse(agents=agents, files=files, chats=chats)


# ----- /search/global --------------------------------------------------------


async def global_search(
    query: str, user_id: str, per_type_limit: int = 4
) -> GlobalSearchResponse:
    """Bucketed search across agents (library + store), files, chat sessions.

    - Non-empty ``query``: hybrid search (semantic + lexical) per bucket,
      each capped at ``per_type_limit`` items.
    - Empty/whitespace ``query``: most-recently-updated items per bucket,
      cached per-user for 60s.
    """
    query = (query or "").strip()
    limit = max(1, min(per_type_limit, 10))

    if not query:
        return await _cached_recent_buckets(user_id, limit)

    agents, files, chats = await asyncio.gather(
        _search_bucket(
            query,
            user_id,
            [ContentType.LIBRARY_AGENT, ContentType.STORE_AGENT],
            limit,
        ),
        _search_bucket(query, user_id, [ContentType.WORKSPACE_FILE], limit),
        _search_bucket(query, user_id, [ContentType.CHAT_SESSION], limit),
    )
    return GlobalSearchResponse(agents=agents, files=files, chats=chats)
