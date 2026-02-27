"""
V2 External API - Marketplace Endpoints

Provides access to the agent marketplace (store).
"""

import logging
import urllib.parse
from typing import Literal, Optional

from fastapi import APIRouter, File, HTTPException, Path, Query, Security, UploadFile
from prisma.enums import APIKeyPermission
from prisma.enums import ContentType as SearchContentType

from backend.api.external.middleware import require_permission
from backend.api.features.store import cache as store_cache
from backend.api.features.store import db as store_db
from backend.api.features.store import media as store_media
from backend.api.features.store.hybrid_search import unified_hybrid_search
from backend.data.auth.base import APIAuthorizationInfo
from backend.util.virus_scanner import scan_content_safe

from .common import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
from .models import (
    LibraryAgent,
    MarketplaceAgent,
    MarketplaceAgentDetails,
    MarketplaceAgentListResponse,
    MarketplaceAgentSubmission,
    MarketplaceAgentSubmissionCreateRequest,
    MarketplaceAgentSubmissionEditRequest,
    MarketplaceAgentSubmissionsListResponse,
    MarketplaceCreator,
    MarketplaceCreatorDetails,
    MarketplaceCreatorsResponse,
    MarketplaceMediaUploadResponse,
    MarketplaceSearchResponse,
    MarketplaceSearchResult,
    MarketplaceUserProfile,
    MarketplaceUserProfileUpdateRequest,
)
from .rate_limit import media_upload_limiter, search_limiter

logger = logging.getLogger(__name__)

marketplace_router = APIRouter()


# ============================================================================
# Endpoints - Read (authenticated)
# ============================================================================


@marketplace_router.get(
    path="/agents",
    summary="List marketplace agents",
    response_model=MarketplaceAgentListResponse,
)
async def list_agents(
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_STORE)
    ),
    featured: bool = Query(default=False, description="Filter to featured agents only"),
    creator: Optional[str] = Query(
        default=None, description="Filter by creator username"
    ),
    sorted_by: Optional[Literal["rating", "runs", "name", "updated_at"]] = Query(
        default=None, description="Sort field"
    ),
    search_query: Optional[str] = Query(default=None, description="Search query"),
    category: Optional[str] = Query(default=None, description="Filter by category"),
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(
        default=DEFAULT_PAGE_SIZE,
        ge=1,
        le=MAX_PAGE_SIZE,
        description=f"Items per page (max {MAX_PAGE_SIZE})",
    ),
) -> MarketplaceAgentListResponse:
    """
    List agents available in the marketplace.

    Supports filtering by featured status, creator, category, and search query.
    Results can be sorted by rating, runs, name, or update time.
    """
    result = await store_cache._get_cached_store_agents(
        featured=featured,
        creator=creator,
        sorted_by=sorted_by,
        search_query=search_query,
        category=category,
        page=page,
        page_size=page_size,
    )

    return MarketplaceAgentListResponse(
        agents=[MarketplaceAgent.from_internal(a) for a in result.agents],
        page=result.pagination.current_page,
        page_size=result.pagination.page_size,
        total_count=result.pagination.total_items,
        total_pages=result.pagination.total_pages,
    )


@marketplace_router.get(
    path="/agents/by-version/{version_id}",
    summary="Get agent by store listing version ID",
)
async def get_agent_by_version(
    version_id: str = Path(description="Store listing version ID"),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_STORE)
    ),
) -> MarketplaceAgentDetails:
    """
    Get detailed information about a marketplace agent by its store listing
    version ID.
    """
    try:
        agent = await store_db.get_store_agent_by_version_id(version_id)
    except Exception:
        raise HTTPException(
            status_code=404, detail=f"Agent version #{version_id} not found"
        )
    return MarketplaceAgentDetails.from_internal(agent)


@marketplace_router.get(
    path="/agents/{username}/{agent_name}",
    summary="Get agent details",
    response_model=MarketplaceAgentDetails,
)
async def get_agent_details(
    username: str = Path(description="Creator username"),
    agent_name: str = Path(description="Agent slug/name"),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_STORE)
    ),
) -> MarketplaceAgentDetails:
    """
    Get detailed information about a specific marketplace agent.
    """
    username = urllib.parse.unquote(username).lower()
    agent_name = urllib.parse.unquote(agent_name).lower()

    agent = await store_cache._get_cached_agent_details(
        username=username, agent_name=agent_name
    )

    return MarketplaceAgentDetails.from_internal(agent)


@marketplace_router.get(
    path="/creators",
    summary="List marketplace creators",
    response_model=MarketplaceCreatorsResponse,
)
async def list_creators(
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_STORE)
    ),
    featured: bool = Query(
        default=False, description="Filter to featured creators only"
    ),
    search_query: Optional[str] = Query(default=None, description="Search query"),
    sorted_by: Optional[Literal["agent_rating", "agent_runs", "num_agents"]] = Query(
        default=None, description="Sort field"
    ),
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(
        default=DEFAULT_PAGE_SIZE,
        ge=1,
        le=MAX_PAGE_SIZE,
        description=f"Items per page (max {MAX_PAGE_SIZE})",
    ),
) -> MarketplaceCreatorsResponse:
    """
    List creators on the marketplace.

    Supports filtering by featured status and search query.
    Results can be sorted by rating, runs, or number of agents.
    """
    result = await store_cache._get_cached_store_creators(
        featured=featured,
        search_query=search_query,
        sorted_by=sorted_by,
        page=page,
        page_size=page_size,
    )

    return MarketplaceCreatorsResponse(
        creators=[MarketplaceCreator.from_internal(c) for c in result.creators],
        page=result.pagination.current_page,
        page_size=result.pagination.page_size,
        total_count=result.pagination.total_items,
        total_pages=result.pagination.total_pages,
    )


@marketplace_router.get(
    path="/creators/{username}",
    summary="Get creator details",
    response_model=MarketplaceCreatorDetails,
)
async def get_creator_details(
    username: str = Path(description="Creator username"),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_STORE)
    ),
) -> MarketplaceCreatorDetails:
    """
    Get detailed information about a specific marketplace creator.
    """
    username = urllib.parse.unquote(username).lower()

    creator = await store_cache._get_cached_creator_details(username=username)

    return MarketplaceCreatorDetails.from_internal(creator)


@marketplace_router.post(
    path="/agents/{username}/{agent_name}/add-to-library",
    summary="Add a marketplace agent to your library",
    status_code=201,
)
async def add_agent_to_library(
    username: str = Path(description="Creator username"),
    agent_name: str = Path(description="Agent slug/name"),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_STORE, APIKeyPermission.WRITE_LIBRARY)
    ),
) -> LibraryAgent:
    """
    Add a marketplace agent to the authenticated user's library.

    If the agent is already in the library, returns the existing entry.
    """
    from backend.api.features.library import db as library_db

    username = urllib.parse.unquote(username).lower()
    agent_name = urllib.parse.unquote(agent_name).lower()

    agent_details = await store_cache._get_cached_agent_details(
        username=username, agent_name=agent_name
    )

    agent = await library_db.add_store_agent_to_library(
        store_listing_version_id=agent_details.store_listing_version_id,
        user_id=auth.user_id,
    )

    return LibraryAgent.from_internal(agent)


# ============================================================================
# Endpoints - Search
# ============================================================================


@marketplace_router.get(
    path="/search",
    summary="Search the marketplace",
)
async def search_marketplace(
    query: str = Query(description="Search query"),
    content_types: Optional[list[SearchContentType]] = Query(
        default=None, description="Content types to filter by"
    ),
    category: Optional[str] = Query(default=None, description="Filter by category"),
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(
        default=DEFAULT_PAGE_SIZE,
        ge=1,
        le=MAX_PAGE_SIZE,
        description=f"Items per page (max {MAX_PAGE_SIZE})",
    ),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_STORE)
    ),
) -> MarketplaceSearchResponse:
    """
    Search the marketplace for agents, blocks, and documentation.

    Uses hybrid search combining semantic and lexical matching.
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


# ============================================================================
# Endpoints - Profile
# ============================================================================


@marketplace_router.get(
    path="/profile",
    summary="Get my marketplace profile",
)
async def get_profile(
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_STORE)
    ),
) -> MarketplaceUserProfile:
    """
    Get the authenticated user's marketplace profile.
    """
    profile = await store_db.get_user_profile(auth.user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    return MarketplaceUserProfile.from_internal(profile)


@marketplace_router.post(
    path="/profile",
    summary="Update my marketplace profile",
)
async def update_profile(
    request: MarketplaceUserProfileUpdateRequest,
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.WRITE_STORE)
    ),
) -> MarketplaceCreatorDetails:
    """
    Update the authenticated user's marketplace profile.

    Creates a profile if one doesn't exist.
    """
    from backend.api.features.store.model import Profile

    profile = Profile(
        name=request.name,
        username=request.username,
        description=request.description,
        links=request.links,
        avatar_url=request.avatar_url,
    )

    creator = await store_db.update_profile(auth.user_id, profile)
    return MarketplaceCreatorDetails.from_internal(creator)


# ============================================================================
# Endpoints - Submissions (CRUD)
# ============================================================================


@marketplace_router.get(
    path="/submissions",
    summary="List my submissions",
    response_model=MarketplaceAgentSubmissionsListResponse,
)
async def list_submissions(
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_STORE)
    ),
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(
        default=DEFAULT_PAGE_SIZE,
        ge=1,
        le=MAX_PAGE_SIZE,
        description=f"Items per page (max {MAX_PAGE_SIZE})",
    ),
) -> MarketplaceAgentSubmissionsListResponse:
    """
    List your marketplace submissions.

    Returns all submissions you've created, including drafts, pending,
    approved, and rejected submissions.
    """
    result = await store_db.get_store_submissions(
        user_id=auth.user_id,
        page=page,
        page_size=page_size,
    )

    return MarketplaceAgentSubmissionsListResponse(
        submissions=[
            MarketplaceAgentSubmission.from_internal(s) for s in result.submissions
        ],
        page=result.pagination.current_page,
        page_size=result.pagination.page_size,
        total_count=result.pagination.total_items,
        total_pages=result.pagination.total_pages,
    )


@marketplace_router.post(
    path="/submissions",
    summary="Create a submission",
    response_model=MarketplaceAgentSubmission,
)
async def create_submission(
    request: MarketplaceAgentSubmissionCreateRequest,
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.WRITE_STORE)
    ),
) -> MarketplaceAgentSubmission:
    """
    Create a new marketplace submission.

    This submits an agent for review to be published in the marketplace.
    The submission will be in PENDING status until reviewed by the team.
    """
    submission = await store_db.create_store_submission(
        user_id=auth.user_id,
        agent_id=request.graph_id,
        agent_version=request.graph_version,
        slug=request.slug,
        name=request.name,
        sub_heading=request.sub_heading,
        description=request.description,
        image_urls=request.image_urls,
        video_url=request.video_url,
        categories=request.categories,
    )

    return MarketplaceAgentSubmission.from_internal(submission)


@marketplace_router.delete(
    path="/submissions/{submission_id}",
    summary="Delete a submission",
)
async def delete_submission(
    submission_id: str = Path(description="Submission ID"),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.WRITE_STORE)
    ),
) -> None:
    """
    Delete a marketplace submission.

    Only submissions in DRAFT status can be deleted.
    """
    success = await store_db.delete_store_submission(
        user_id=auth.user_id,
        submission_id=submission_id,
    )

    if not success:
        raise HTTPException(
            status_code=404, detail=f"Submission #{submission_id} not found"
        )


@marketplace_router.post(
    path="/submissions/media",
    summary="Upload submission media",
)
async def upload_submission_media(
    file: UploadFile = File(...),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.WRITE_STORE)
    ),
) -> MarketplaceMediaUploadResponse:
    """
    Upload an image or video for a marketplace submission.

    Accepted types: JPEG, PNG, GIF, WebP, MP4, WebM. Max size: 10MB.
    """
    media_upload_limiter.check(auth.user_id)

    max_size = 10 * 1024 * 1024  # 10MB limit for external API

    content = await file.read()
    if len(content) > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File size ({len(content)} bytes) exceeds the 10MB limit",
        )

    # Virus scan
    await scan_content_safe(content, filename=file.filename or "upload")

    # Reset file position for store_media to read
    await file.seek(0)

    url = await store_media.upload_media(
        user_id=auth.user_id,
        file=file,
    )

    return MarketplaceMediaUploadResponse(url=url)


@marketplace_router.put(
    path="/submissions/{version_id}",
    summary="Edit a submission",
)
async def edit_submission(
    request: MarketplaceAgentSubmissionEditRequest,
    version_id: str = Path(description="Store listing version ID"),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.WRITE_STORE)
    ),
) -> MarketplaceAgentSubmission:
    """
    Edit an existing marketplace submission.
    """
    try:
        submission = await store_db.edit_store_submission(
            user_id=auth.user_id,
            store_listing_version_id=version_id,
            name=request.name,
            sub_heading=request.sub_heading,
            description=request.description,
            image_urls=request.image_urls,
            video_url=request.video_url,
            agent_output_demo_url=request.agent_output_demo_url,
            categories=request.categories,
            changes_summary=request.changes_summary,
            recommended_schedule_cron=request.recommended_schedule_cron,
            instructions=request.instructions,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return MarketplaceAgentSubmission.from_internal(submission)
