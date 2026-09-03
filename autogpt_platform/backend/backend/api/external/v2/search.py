"""
V2 External API - Search Endpoints

Cross-domain hybrid search across agents, blocks, and documentation.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query, Security
from prisma.enums import ContentType as SearchContentType

from backend.api.external.middleware import require_auth
from backend.api.features.search.hybrid_search import unified_hybrid_search
from backend.data.auth.base import APIAuthorizationInfo

from .models import MarketplaceSearchResult
from .pagination import Page, PageRequest, page_request
from .rate_limit import search_limiter

logger = logging.getLogger(__name__)

search_router = APIRouter(tags=["search"])


@search_router.get(
    path="",
    summary="Search content and capabilities of the platform",
    operation_id="search",
)
async def search(
    query: str = Query(description="Search query"),
    content_types: Optional[list[SearchContentType]] = Query(
        default=None, description="Content types to filter by"
    ),
    category: Optional[str] = Query(default=None, description="Filter by category"),
    page: PageRequest = Depends(page_request),
    auth: APIAuthorizationInfo = Security(require_auth),
) -> Page[MarketplaceSearchResult]:
    """
    Search the platform's content and capabilities (hybrid search: literal + semantic).

    Searches across agents, blocks, and documentation. Results are ranked
    by a combination of keyword matching and semantic similarity.

    **Rate limit:** 30 requests per minute per user.
    """
    await search_limiter.check(auth.user_id)

    results, total_count = await unified_hybrid_search(
        query=query,
        content_types=content_types,
        category=category,
        page=page.page,
        page_size=page.limit,
        user_id=auth.user_id,
    )

    return page.paged(
        [
            MarketplaceSearchResult(
                content_type=r["content_type"],
                content_id=r["content_id"],
                searchable_text=r["searchable_text"],
                metadata=r.get("metadata"),
                updated_at=r.get("updated_at"),
                combined_score=r.get("combined_score"),
            )
            for r in results
        ],
        total_count=total_count,
    )
