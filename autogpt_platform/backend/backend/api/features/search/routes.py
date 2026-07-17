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
from .rate_limit import enforce_global_search_rate_limit

# Bound on ``q`` enforced at the API edge so a 100 KB query string
# cannot reach ``embed_query`` (paid OpenAI call) or the BM25 reranker.
# 512 chars comfortably covers the human-typeable range; UIs that need
# more for some internal flow should hit the dedicated feature search
# endpoints, not the global one.
MAX_QUERY_LENGTH = 512

router = APIRouter(
    tags=["search"],
    dependencies=[Security(autogpt_auth_lib.requires_user)],
)


@router.get(
    "/global",
    summary="Global search",
    response_model=GlobalSearchResponse,
)
async def global_search_endpoint(
    q: str = Query(
        "",
        max_length=MAX_QUERY_LENGTH,
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
    # Skip rate-limit on the empty-query branch — it hits the per-user
    # cache, not OpenAI. Only the non-empty path triggers paid embedding
    # calls, so that's where the QPS cap matters.
    if q.strip():
        await enforce_global_search_rate_limit(user_id)
    return await service.global_search(
        query=q, user_id=user_id, per_type_limit=per_type_limit
    )
