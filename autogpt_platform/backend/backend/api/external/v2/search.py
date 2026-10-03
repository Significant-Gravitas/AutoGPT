"""
V2 External API - Search Endpoints

Cross-domain hybrid search across agents, blocks, and documentation.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Security
from prisma.enums import APIKeyPermission, ContentType
from starlette import status

from backend.api.features.search.hybrid_search import unified_hybrid_search

from .models import MarketplaceSearchResult, SearchContentType
from .pagination import Page, PageRequest, page_request
from .rate_limit import search_limiter
from .tenancy import TenantContext, require_auth

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
    auth: TenantContext = Security(require_auth),
) -> Page[MarketplaceSearchResult]:
    """
    Search the platform's content and capabilities (hybrid search: literal + semantic).

    Searches public agents, blocks and documentation by default. The caller's own
    library agents and workspace files are searchable too, each requiring the same
    permission that its own endpoints require.

    **Rate limit:** 30 requests per minute per user.
    """
    await search_limiter.check(auth.user_id)

    requested = content_types or PUBLIC_CONTENT_TYPES
    for content_type in requested:
        if (scope := PRIVATE_CONTENT_TYPE_SCOPES.get(content_type)) and (
            scope not in auth.scopes
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Searching {content_type.value} requires the "
                f"{scope.value} permission",
            )

    results, total_count = await unified_hybrid_search(
        query=query,
        content_types=[ContentType(t.value) for t in requested],
        category=category,
        page=page.page,
        page_size=page.limit,
        user_id=auth.user_id,
    )

    return page.paged(
        [
            MarketplaceSearchResult(
                content_type=SearchContentType(r["content_type"]),
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


# Searching these reaches the caller's own rows, so each costs the permission
# that guards the same data elsewhere in v2. Everything else is public content.
PRIVATE_CONTENT_TYPE_SCOPES: dict[SearchContentType, APIKeyPermission] = {
    SearchContentType.LIBRARY_AGENT: APIKeyPermission.READ_LIBRARY,
    SearchContentType.WORKSPACE_FILE: APIKeyPermission.READ_FILES,
}

# Named here rather than left to the internal default, so v2's contract does not
# change when that default does.
PUBLIC_CONTENT_TYPES = [
    t for t in SearchContentType if t not in PRIVATE_CONTENT_TYPE_SCOPES
]
