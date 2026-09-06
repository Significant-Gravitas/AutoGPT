"""
V2 External API - Marketplace Endpoints

Provides access to the agent marketplace (store).
"""

import logging
import urllib.parse
from typing import Literal, Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Path,
    Query,
    Security,
    UploadFile,
)
from prisma.enums import APIKeyPermission
from starlette import status

from backend.api.features.library import db as library_db
from backend.api.features.store import cache as store_cache
from backend.api.features.store import db as store_db
from backend.api.features.store import media as store_media
from backend.api.features.store.db import (
    StoreAgentsSortOptions,
    StoreCreatorsSortOptions,
)
from backend.api.features.store.model import ProfileUpdateRequest
from backend.util.virus_scanner import scan_content_safe

from .models import (
    LibraryAgent,
    MarketplaceAgent,
    MarketplaceAgentDetails,
    MarketplaceAgentSubmission,
    MarketplaceAgentSubmissionCreateRequest,
    MarketplaceAgentSubmissionEditRequest,
    MarketplaceCreatorDetails,
    MarketplaceMediaUploadResponse,
    MarketplaceUserProfile,
    MarketplaceUserProfileUpdateRequest,
)
from .pagination import Page, PageRequest, page_request
from .rate_limit import media_upload_limiter
from .tenancy import TenantContext, require_auth, require_permission

logger = logging.getLogger(__name__)

marketplace_router = APIRouter(tags=["marketplace"])


# ============================================================================
# Agents
# ============================================================================


@marketplace_router.get(
    path="/agents",
    summary="List or search marketplace agents",
    operation_id="listMarketplaceAgents",
)
async def list_agents(
    featured: bool = Query(
        default=False, description="Filter to only show featured agents"
    ),
    creator: Optional[str] = Query(
        default=None, description="Filter by creator username"
    ),
    category: Optional[str] = Query(default=None, description="Filter by category"),
    search_query: Optional[str] = Query(
        default=None, description="Literal + semantic search on names and descriptions"
    ),
    sorted_by: Optional[Literal["rating", "runs", "name", "updated_at"]] = Query(
        default=None,
        description="Property to sort results by. Ignored if search_query is provided.",
    ),
    page: PageRequest = Depends(page_request),
    # This data is public, but we still require auth for access tracking and rate limits
    auth: TenantContext = Security(require_auth),
) -> Page[MarketplaceAgent]:
    """List agents available in the marketplace, with optional filtering and sorting."""
    result = await store_cache._get_cached_store_agents(
        featured=featured,
        creator=creator,
        sorted_by=StoreAgentsSortOptions(sorted_by) if sorted_by else None,
        search_query=search_query,
        category=category,
        page=page.page,
        page_size=page.limit,
    )

    return page.paged(
        [MarketplaceAgent.from_internal(a) for a in result.agents],
        total_count=result.pagination.total_items,
    )


@marketplace_router.get(
    path="/agents/by-version/{version_id}",
    summary="Get marketplace agent by version ID",
    operation_id="getMarketplaceAgentByListingVersion",
)
async def get_agent_by_version(
    version_id: str,
    # This data is public, but we still require auth for access tracking and rate limits
    auth: TenantContext = Security(require_auth),
) -> MarketplaceAgentDetails:
    """Get details of a marketplace agent by its store listing version ID."""
    agent = await store_db.get_store_agent_by_version_id(version_id)
    return MarketplaceAgentDetails.from_internal(agent)


@marketplace_router.get(
    path="/agents/{username}/{agent_name}",
    summary="Get marketplace agent details",
    operation_id="getMarketplaceAgent",
)
async def get_agent_details(
    username: str,
    agent_name: str,
    # This data is public, but we still require auth for access tracking and rate limits
    auth: TenantContext = Security(require_auth),
) -> MarketplaceAgentDetails:
    """Get details of a specific marketplace agent."""
    username = urllib.parse.unquote(username).lower()
    agent_name = urllib.parse.unquote(agent_name).lower()

    agent = await store_cache._get_cached_agent_details(
        username=username, agent_name=agent_name
    )

    return MarketplaceAgentDetails.from_internal(agent)


@marketplace_router.post(
    path="/agents/{username}/{agent_name}/add-to-library",
    summary="Add marketplace agent to library",
    operation_id="addMarketplaceAgentToLibrary",
    status_code=status.HTTP_201_CREATED,
)
async def add_agent_to_library(
    username: str,
    agent_name: str,
    auth: TenantContext = Security(require_permission(APIKeyPermission.WRITE_LIBRARY)),
) -> LibraryAgent:
    """Add a marketplace agent to the authenticated user's library."""
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
# Creators
# ============================================================================


@marketplace_router.get(
    path="/creators",
    summary="List marketplace creators",
    operation_id="listMarketplaceCreators",
)
async def list_creators(
    featured: bool = Query(
        default=False, description="Filter to featured creators only"
    ),
    search_query: Optional[str] = Query(
        default=None, description="Literal + semantic search on names and descriptions"
    ),
    sorted_by: Optional[Literal["agent_rating", "agent_runs", "num_agents"]] = Query(
        default=None, description="Sort field"
    ),
    page: PageRequest = Depends(page_request),
    # This data is public, but we still require auth for access tracking and rate limits
    auth: TenantContext = Security(require_auth),
) -> Page[MarketplaceCreatorDetails]:
    """List or search marketplace creators."""
    result = await store_cache._get_cached_store_creators(
        featured=featured,
        search_query=search_query,
        sorted_by=StoreCreatorsSortOptions(sorted_by) if sorted_by else None,
        page=page.page,
        page_size=page.limit,
    )

    return page.paged(
        [MarketplaceCreatorDetails.from_internal(c) for c in result.creators],
        total_count=result.pagination.total_items,
    )


@marketplace_router.get(
    path="/creators/{username}",
    summary="Get marketplace creator details",
    operation_id="getMarketplaceCreator",
)
async def get_creator_details(
    username: str,
    # This data is public, but we still require auth for access tracking and rate limits
    auth: TenantContext = Security(require_auth),
) -> MarketplaceCreatorDetails:
    """Get a marketplace creator's profile w/ stats."""
    username = urllib.parse.unquote(username).lower()
    creator = await store_cache._get_cached_creator_details(username=username)
    return MarketplaceCreatorDetails.from_internal(creator)


# ============================================================================
# Profile
# ============================================================================


@marketplace_router.get(
    path="/profile",
    summary="Get my marketplace profile",
    operation_id="getMarketplaceMyProfile",
)
async def get_profile(
    auth: TenantContext = Security(require_permission(APIKeyPermission.READ_STORE)),
) -> MarketplaceCreatorDetails:
    """Get the authenticated user's marketplace profile w/ creator stats."""
    profile = await store_db.get_user_profile(auth.user_id)
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Profile not found",
        )

    creator = await store_cache._get_cached_creator_details(username=profile.username)
    return MarketplaceCreatorDetails.from_internal(creator)


@marketplace_router.patch(
    path="/profile",
    summary="Update my marketplace profile",
    operation_id="updateMarketplaceMyProfile",
)
async def update_profile(
    request: MarketplaceUserProfileUpdateRequest,
    auth: TenantContext = Security(require_permission(APIKeyPermission.WRITE_STORE)),
) -> MarketplaceUserProfile:
    """Update the authenticated user's marketplace profile.

    Only the fields present in the request body are changed. Set `avatar_url`
    to `null` to remove the avatar. A user without a profile yet can create one
    here by sending at least `username`, `name` and `description`.
    """
    # exclude_unset keeps "omitted" distinct from "explicitly null" downstream
    profile = ProfileUpdateRequest.model_validate(
        request.model_dump(exclude_unset=True)
    )

    updated_profile = await store_db.update_profile(auth.user_id, profile)
    return MarketplaceUserProfile.from_internal(updated_profile)


# ============================================================================
# Submissions
# ============================================================================


@marketplace_router.get(
    path="/submissions",
    summary="List my marketplace submissions",
    operation_id="listMarketplaceSubmissions",
)
async def list_submissions(
    page: PageRequest = Depends(page_request),
    auth: TenantContext = Security(require_permission(APIKeyPermission.READ_STORE)),
) -> Page[MarketplaceAgentSubmission]:
    """List the authenticated user's marketplace listing submissions."""
    result = await store_db.get_store_submissions(
        user_id=auth.user_id,
        page=page.page,
        page_size=page.limit,
        organization_id=auth.organization_id,
    )

    return page.paged(
        [MarketplaceAgentSubmission.from_internal(s) for s in result.submissions],
        total_count=result.pagination.total_items,
    )


@marketplace_router.post(
    path="/submissions",
    summary="Create marketplace submission",
    operation_id="createMarketplaceSubmission",
    status_code=status.HTTP_201_CREATED,
)
async def create_submission(
    request: MarketplaceAgentSubmissionCreateRequest,
    auth: TenantContext = Security(require_permission(APIKeyPermission.WRITE_STORE)),
) -> MarketplaceAgentSubmission:
    """Submit a new marketplace listing for review."""
    submission = await store_db.create_store_submission(
        user_id=auth.user_id,
        graph_id=request.graph_id,
        graph_version=request.graph_version,
        slug=request.slug,
        name=request.name,
        sub_heading=request.sub_heading,
        description=request.description,
        instructions=request.instructions,
        categories=request.categories,
        image_urls=request.image_urls,
        video_url=request.video_url,
        agent_output_demo_url=request.agent_output_demo_url,
        changes_summary=request.changes_summary or "Initial Submission",
        recommended_schedule_cron=request.recommended_schedule_cron,
        organization_id=auth.organization_id,
    )

    return MarketplaceAgentSubmission.from_internal(submission)


@marketplace_router.put(
    path="/submissions/{version_id}",
    summary="Edit marketplace submission",
    operation_id="updateMarketplaceSubmission",
)
async def edit_submission(
    request: MarketplaceAgentSubmissionEditRequest,
    version_id: str = Path(description="Store listing version ID"),
    auth: TenantContext = Security(require_permission(APIKeyPermission.WRITE_STORE)),
) -> MarketplaceAgentSubmission:
    """Replace a pending marketplace listing submission.

    Every field is written, so send the whole listing; omitting one clears it.
    """
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
        organization_id=auth.organization_id,
    )
    return MarketplaceAgentSubmission.from_internal(submission)


@marketplace_router.delete(
    path="/submissions/{version_id}",
    summary="Delete marketplace submission",
    operation_id="deleteMarketplaceSubmission",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_submission(
    version_id: str,
    auth: TenantContext = Security(require_permission(APIKeyPermission.WRITE_STORE)),
) -> None:
    """Delete a marketplace listing submission. Approved listings can not be deleted."""
    success = await store_db.delete_store_submission(
        user_id=auth.user_id,
        store_listing_version_id=version_id,
        organization_id=auth.organization_id,
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Submission #{version_id} not found",
        )


# ============================================================================
# Submission Media
# ============================================================================


@marketplace_router.post(
    path="/submissions/media",
    summary="Upload marketplace submission media",
    operation_id="uploadMarketplaceSubmissionMedia",
    status_code=status.HTTP_201_CREATED,
)
async def upload_submission_media(
    file: UploadFile = File(...),
    auth: TenantContext = Security(require_permission(APIKeyPermission.WRITE_STORE)),
) -> MarketplaceMediaUploadResponse:
    """
    Upload an image or video for a marketplace submission. Max size: 10MB.

    **Rate limit:** 10 requests per 5 minutes per user.
    """
    await media_upload_limiter.check(auth.user_id)

    max_size = 10 * 1024 * 1024  # 10MB limit for external API

    content = await file.read()
    if len(content) > max_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
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
