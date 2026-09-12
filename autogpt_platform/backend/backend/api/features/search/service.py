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

import pydantic
from prisma.enums import ContentType

from backend.api.features.search import hybrid_search
from backend.api.features.search.model import (
    GlobalSearchResponse,
    SearchItemType,
    SearchResultItem,
)
from backend.copilot.model import get_user_sessions
from backend.data.db import query_raw_with_schema
from backend.data.db_accessors import library_db, search
from backend.data.workspace import get_workspace
from backend.util.cache import cached
from backend.util.workspace import WorkspaceManager

logger = logging.getLogger(__name__)


# Minimum length of the trailing token before we enable
# ``to_tsquery(... 'token:*')`` prefix matching for hybrid search.
# A single-char prefix would match a large fraction of the embedding
# table (the lexical candidate branch in ``unified_hybrid_search`` has
# no LIMIT before scoring), so we wait for the user to type the second
# character before turning prefix-match on.
_PREFIX_MATCH_MIN_TAIL_LEN = 2


def _should_prefix_match(query: str) -> bool:
    """Whether to enable prefix-match for a search-as-you-type query.

    Looks at the trailing token because that's the one the user is
    actively typing — earlier tokens are assumed complete.
    """
    tail = query.rsplit(maxsplit=1)[-1] if query.strip() else ""
    return len(tail) >= _PREFIX_MATCH_MIN_TAIL_LEN


# ----- helpers ---------------------------------------------------------------


# Pydantic coerces both ``ContentType`` enum members (in-process accessor
# path) and bare strings (raw-SQL / RPC shim path) into the enum at the
# boundary, so the dispatch below can rely on a typed value rather than
# duck-typing the input with ``getattr``.
_CONTENT_TYPE_ADAPTER = pydantic.TypeAdapter(ContentType)

# Hybrid-search rows carry their metadata as a JSON column; Prisma
# deserialises it to a dict on the way out. A typed adapter at the
# boundary turns "you probably get a dict" into "you definitely get a
# dict (or this row gets skipped)" so downstream ``.get(...)`` is safe.
_METADATA_ADAPTER = pydantic.TypeAdapter(dict[str, Any])


def _hybrid_row_to_item(row: hybrid_search.HybridSearchRow) -> SearchResultItem | None:
    """Convert one ``unified_hybrid_search`` row into a ``SearchResultItem``.

    Returns None when the row's content type is one we don't surface
    through /search/global (e.g. BLOCK / DOCUMENTATION rows that might
    sneak in if a caller widens the type filter), or when the row's
    ``content_type`` / ``metadata`` columns fail to coerce to their
    expected shapes (defensive — should not happen in practice).
    """
    try:
        content_type = _CONTENT_TYPE_ADAPTER.validate_python(row.get("content_type"))
        metadata = _METADATA_ADAPTER.validate_python(row.get("metadata") or {})
    except pydantic.ValidationError:
        return None

    item_type: SearchItemType
    if content_type is ContentType.LIBRARY_AGENT:
        title = metadata.get("name") or row.get("searchable_text") or ""
        item_type = "library_agent"
        subtitle = None
    elif content_type is ContentType.STORE_AGENT:
        title = metadata.get("name") or row.get("searchable_text") or ""
        item_type = "store_agent"
        cats = metadata.get("categories") or []
        subtitle = ", ".join(cats[:2]) if cats else None
    elif content_type is ContentType.WORKSPACE_FILE:
        title = metadata.get("name") or row.get("searchable_text") or ""
        item_type = "workspace_file"
        subtitle = metadata.get("mime_type") or metadata.get("path")
    elif content_type is ContentType.CHAT_SESSION:
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
        metadata=metadata,
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
            # The command palette must return *nothing* for unrelated
            # queries — cosine similarity between two embeddings is
            # almost never 0, so a low floor lets the semantic UNION
            # surface arbitrary recent rows for queries like
            # "hello walhalla". Match the store-agent threshold (0.20).
            min_score=hybrid_search.DEFAULT_STORE_AGENT_MIN_SCORE,
            # Search-as-you-type: the trailing token is usually a partial
            # word, so prefix-match the lexical signal ("se" -> "se:*").
            # Gated on the trailing token having >= 2 chars: a single
            # letter ("s:*") would match a huge fraction of the embedding
            # table and let the lexical candidate set explode before
            # ts_rank_cd scoring (the lexical branch isn't capped like
            # the semantic branch's LIMIT 200). For 1-char queries we
            # fall back to whole-word plainto_tsquery, which usually
            # yields no matches and lets the semantic signal take over.
            prefix_match=_should_prefix_match(query),
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

    # Read-time enrichment so store-agent rows embedded BEFORE the
    # ``creator``/``slug`` metadata fields were added still link to the
    # marketplace correctly. Newer embeds carry these in the JSON
    # column; older ones need a join back to ``StoreListing`` + Profile
    # to be clickable.
    await _enrich_store_agent_metadata(items)
    return items


async def _enrich_store_agent_metadata(items: list[SearchResultItem]) -> None:
    """Backfill ``creator`` and ``slug`` on any store-agent items that
    are missing them.

    Mutates ``items`` in place. Single batched SQL roundtrip, runs only
    when at least one store-agent item is missing the fields, so the
    common-case (all rows embedded post-fix) pays nothing.
    """
    stale_ids = [
        item.id
        for item in items
        if item.type == "store_agent"
        and not (item.metadata.get("creator") and item.metadata.get("slug"))
    ]
    if not stale_ids:
        return

    try:
        rows = await query_raw_with_schema(
            """
            SELECT slv.id, sl.slug, p.username AS creator
            FROM {schema_prefix}"StoreListingVersion" slv
            JOIN {schema_prefix}"StoreListing" sl
              ON slv."storeListingId" = sl.id
            JOIN {schema_prefix}"Profile" p
              ON p."userId" = sl."owningUserId"
            WHERE slv.id = ANY($1::text[])
              AND sl."isDeleted" = false
            """,
            stale_ids,
        )
    except Exception as e:
        # Enrichment is purely a click-routing concern — never let it
        # 500 the whole bucket. A search result with a missing
        # marketplace URL renders as inert (handled by the FE click
        # guard), which matches current behaviour.
        logger.warning("Failed to enrich store-agent metadata: %s", e)
        return

    by_id = {row["id"]: row for row in rows}
    for item in items:
        if item.type != "store_agent":
            continue
        enrichment = by_id.get(item.id)
        if enrichment is None:
            continue
        item.metadata["creator"] = enrichment["creator"]
        item.metadata["slug"] = enrichment["slug"]


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


# Over-fetch so the in-Python relevance rerank has enough candidates
# to actually re-order. Bounded so a user with thousands of name-
# matching files doesn't blow the bucket cost.
_RELEVANCE_OVERFETCH_CAP = 32


def _title_relevance_score(title: str, query: str) -> int:
    """Three-tier literal-match score, highest = best.

    Exact case-insensitive match (3) > startswith (2) > contains (1).
    Falls back to 0 when nothing matches, which shouldn't happen in
    practice because the DB filter already applied ``contains``.
    """
    title_lower = title.lower()
    query_lower = query.lower()
    if title_lower == query_lower:
        return 3
    if title_lower.startswith(query_lower):
        return 2
    if query_lower in title_lower:
        return 1
    return 0


async def _files_bucket(
    user_id: str, limit: int, query: str | None = None
) -> list[SearchResultItem]:
    """Workspace files, filtered by ``query`` substring when provided.

    Uses a direct ``ILIKE`` filter on ``name`` instead of the embedding
    index so newly-written files are findable immediately. Workspace
    file embeddings only encode the name anyway, so we lose nothing in
    quality and gain freshness.

    When ``query`` is non-empty we over-fetch up to
    ``_RELEVANCE_OVERFETCH_CAP`` candidates and re-rank by literal
    name-match relevance before truncating to ``limit``, so a stale
    exact match still beats four fresh substring matches. The recents
    branch (empty query) keeps the DB-side ``createdAt DESC`` ordering
    untouched.
    """
    try:
        workspace = await get_workspace(user_id)
        if workspace is None:
            # User has no workspace yet — nothing to list, no need to
            # create one just for the recents view.
            return []
        manager = WorkspaceManager(user_id, workspace.id, session_id=None)
        fetch_limit = max(limit, _RELEVANCE_OVERFETCH_CAP) if query else limit
        files = await manager.list_files(
            limit=fetch_limit,
            include_all_sessions=True,
            name_contains=query or None,
        )
    except Exception as e:
        logger.warning("Failed to list workspace files for %s: %s", user_id, e)
        return []

    if query:
        # Sort by relevance desc, then by ``createdAt`` desc as
        # tiebreaker so the freshness ordering is preserved within a
        # relevance bucket. ``list_files`` already returns
        # createdAt-desc, so a stable sort keeps that tiebreaker.
        files = sorted(
            files, key=lambda f: _title_relevance_score(f.name, query), reverse=True
        )

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


async def _chats_bucket(
    user_id: str, limit: int, query: str | None = None
) -> list[SearchResultItem]:
    """Chat sessions, filtered by ``query`` substring when provided.

    Uses a direct ``ILIKE`` filter on ``title`` instead of the embedding
    index so newly-renamed sessions are findable immediately. Chat
    session embeddings only encode the title anyway.

    Non-empty queries over-fetch + relevance-rerank exactly like
    ``_files_bucket`` (see its docstring). Empty queries keep the
    ``updatedAt DESC`` recents ordering.
    """
    try:
        fetch_limit = max(limit, _RELEVANCE_OVERFETCH_CAP) if query else limit
        sessions, _total = await get_user_sessions(
            user_id=user_id,
            limit=fetch_limit,
            offset=0,
            title_contains=query or None,
        )
    except Exception as e:
        logger.warning("Failed to list chat sessions for %s: %s", user_id, e)
        return []

    if query:
        # Untitled sessions still ride the recents path elsewhere; this
        # branch only fires with a query, where ``title_contains``
        # already filters them out. The placeholder fallback below
        # ("Untitled chat") is therefore unreachable here, but stays as
        # a defensive read.
        sessions = sorted(
            sessions,
            key=lambda s: _title_relevance_score(s.title or "", query),
            reverse=True,
        )

    items: list[SearchResultItem] = []
    for session in sessions[:limit]:
        items.append(
            SearchResultItem(
                id=session.session_id,
                type="chat_session",
                title=session.title or "Untitled chat",
                metadata={
                    "chat_status": session.chat_status,
                    # Exposed so the search result UI can swap the generic
                    # chat icon for a platform-specific one (e.g. Discord)
                    # when the session originated from an external chat.
                    "source_platform": session.metadata.source_platform,
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
        _files_bucket(user_id, limit),
        _chats_bucket(user_id, limit),
    )
    return GlobalSearchResponse(agents=agents, files=files, chats=chats)


# ----- /search/global --------------------------------------------------------


async def global_search(
    query: str, user_id: str, per_type_limit: int = 4
) -> GlobalSearchResponse:
    """Bucketed search across agents (library + store), files, chat sessions.

    - Non-empty ``query``:
        - Agents: hybrid search (semantic + lexical) via the embedding
          index — store agents have rich descriptions worth embedding.
        - Files & chats: direct ``ILIKE`` on ``name`` / ``title`` so
          freshly-created rows are findable without waiting on async
          embedding generation. Their embeddings only encode the
          name/title anyway, so we lose nothing in quality.
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
        # Files & chats bypass the embedding index — see _files_bucket /
        # _chats_bucket docstrings. Direct ILIKE keeps freshly-created
        # rows findable without waiting on async embedding generation.
        _files_bucket(user_id, limit, query=query),
        _chats_bucket(user_id, limit, query=query),
    )
    return GlobalSearchResponse(agents=agents, files=files, chats=chats)
