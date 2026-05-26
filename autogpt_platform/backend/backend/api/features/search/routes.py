"""HTTP routes for the unified ``/api/search`` feature.

``GET /api/search/global?q=…&per_type_limit=4`` — bucketed search across
agents (library + store), workspace files, and chat sessions.

- When ``q`` is non-empty: hybrid search (semantic + lexical + BM25),
  ranked independently inside each bucket and capped at
  ``per_type_limit`` items.
- When ``q`` is empty/omitted: most-recently-updated items per bucket
  (cached per-user with a short TTL). Same response shape — the
  frontend renders both with one component.
"""

from __future__ import annotations

import autogpt_libs.auth as autogpt_auth_lib
from fastapi import APIRouter, Query, Security

from . import service
from .model import GlobalSearchResponse

router = APIRouter(
    tags=["search"],
    dependencies=[Security(autogpt_auth_lib.requires_user)],
)


@router.get(
    "/global",
    summary="Global search (hybrid on query, recent on empty)",
    response_model=GlobalSearchResponse,
)
async def global_search_endpoint(
    q: str = Query(
        "",
        description=(
            "Free-form search query (semantic + lexical). When empty or "
            "omitted, the endpoint returns the most-recently-updated "
            "items per bucket instead."
        ),
    ),
    per_type_limit: int = Query(
        4,
        ge=1,
        le=10,
        description=(
            "Maximum items returned per bucket. The response contains "
            "*at most* this many agents, files, and chats — fewer when "
            "the bucket has no/few matches."
        ),
    ),
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> GlobalSearchResponse:
    return await service.global_search(
        query=q, user_id=user_id, per_type_limit=per_type_limit
    )
