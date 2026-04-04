"""
V2 External API - Search Endpoints

Cross-domain hybrid search across agents, blocks, and documentation.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Query, Security
from prisma.enums import ContentType as SearchContentType

from backend.api.external.middleware import require_auth
from backend.api.features.store.hybrid_search import unified_hybrid_search
from backend.data.auth.base import APIAuthorizationInfo

from .common import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
from .models import MarketplaceSearchResponse, MarketplaceSearchResult
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
    page: int = Query(ge=1, default=1),
    page_size: int = Query(ge=1, le=MAX_PAGE_SIZE, default=DEFAULT_PAGE_SIZE),
    auth: APIAuthorizationInfo = Security(require_auth),
) -> MarketplaceSearchResponse:
    """
    Search the platform's content and capabilities (hybrid search: literal + semantic).

    Searches across agents, blocks, and documentation. Results are ranked
    by a combination of keyword matching and semantic similarity.
    """
    search_limiter.check(auth.user_id)

    results, total_count = await unified_hybrid_search(
        query=query,
        content_types=content_types,
        category=category,
        page=page,
        page_size=page_size,
        user_id=auth.user_id,
    )

    total_pages = max(1, (total_count + page_size - 1) // page_size)

    return MarketplaceSearchResponse(
        results=[
            MarketplaceSearchResult(
                content_type=r.get("content_type", ""),
                content_id=r.get("content_id", ""),
                searchable_text=r.get("searchable_text", ""),
                metadata=r.get("metadata"),
                updated_at=r.get("updated_at"),
                combined_score=r.get("combined_score"),
            )
            for r in results
        ],
        page=page,
        page_size=page_size,
        total_count=total_count,
        total_pages=total_pages,
    )
