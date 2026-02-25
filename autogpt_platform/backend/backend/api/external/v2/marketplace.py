"""
V2 External API - Marketplace Endpoints

Provides access to the agent marketplace (store).
"""

import logging
import urllib.parse
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Path, Query, Security
from prisma.enums import APIKeyPermission

from backend.api.external.middleware import require_permission
from backend.api.features.store import cache as store_cache
from backend.api.features.store import db as store_db
from backend.api.features.store import model as store_model
from backend.data.auth.base import APIAuthorizationInfo

from .common import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
from .models import (
    CreateSubmissionRequest,
    MarketplaceAgent,
    MarketplaceAgentDetails,
    MarketplaceAgentsResponse,
    MarketplaceCreator,
    MarketplaceCreatorDetails,
    MarketplaceCreatorsResponse,
    MarketplaceSubmission,
    SubmissionsListResponse,
)

logger = logging.getLogger(__name__)

marketplace_router = APIRouter()


# ============================================================================
# Conversion Functions
# ============================================================================


def _convert_store_agent(agent: store_model.StoreAgent) -> MarketplaceAgent:
    """Convert internal StoreAgent to v2 API model."""
    return MarketplaceAgent(
        slug=agent.slug,
        name=agent.agent_name,
        description=agent.description,
        sub_heading=agent.sub_heading,
        creator=agent.creator,
        creator_avatar=agent.creator_avatar,
        runs=agent.runs,
        rating=agent.rating,
        image_url=agent.agent_image,
    )


def _convert_store_agent_details(
    agent: store_model.StoreAgentDetails,
) -> MarketplaceAgentDetails:
    """Convert internal StoreAgentDetails to v2 API model."""
    return MarketplaceAgentDetails(
        store_listing_version_id=agent.store_listing_version_id,
        slug=agent.slug,
        name=agent.agent_name,
        description=agent.description,
        sub_heading=agent.sub_heading,
        instructions=agent.instructions,
        creator=agent.creator,
        creator_avatar=agent.creator_avatar,
        categories=agent.categories,
        runs=agent.runs,
        rating=agent.rating,
        image_urls=agent.agent_image,
        video_url=agent.agent_video,
        versions=agent.versions,
        agent_graph_versions=agent.agentGraphVersions,
        agent_graph_id=agent.agentGraphId,
        last_updated=agent.last_updated,
    )


def _convert_creator(creator: store_model.Creator) -> MarketplaceCreator:
    """Convert internal Creator to v2 API model."""
    return MarketplaceCreator(
        name=creator.name,
        username=creator.username,
        description=creator.description,
        avatar_url=creator.avatar_url,
        num_agents=creator.num_agents,
        agent_rating=creator.agent_rating,
        agent_runs=creator.agent_runs,
        is_featured=creator.is_featured,
    )


def _convert_creator_details(
    creator: store_model.CreatorDetails,
) -> MarketplaceCreatorDetails:
    """Convert internal CreatorDetails to v2 API model."""
    return MarketplaceCreatorDetails(
        name=creator.name,
        username=creator.username,
        description=creator.description,
        avatar_url=creator.avatar_url,
        agent_rating=creator.agent_rating,
        agent_runs=creator.agent_runs,
        top_categories=creator.top_categories,
        links=creator.links,
    )


def _convert_submission(sub: store_model.StoreSubmission) -> MarketplaceSubmission:
    """Convert internal StoreSubmission to v2 API model."""
    return MarketplaceSubmission(
        graph_id=sub.agent_id,
        graph_version=sub.agent_version,
        name=sub.name,
        sub_heading=sub.sub_heading,
        slug=sub.slug,
        description=sub.description,
        instructions=sub.instructions,
        image_urls=sub.image_urls,
        date_submitted=sub.date_submitted,
        status=sub.status.value,
        runs=sub.runs,
        rating=sub.rating,
        store_listing_version_id=sub.store_listing_version_id,
        version=sub.version,
        review_comments=sub.review_comments,
        reviewed_at=sub.reviewed_at,
        video_url=sub.video_url,
        categories=sub.categories,
    )


# ============================================================================
# Endpoints - Read (authenticated)
# ============================================================================


@marketplace_router.get(
    path="/agents",
    summary="List marketplace agents",
    response_model=MarketplaceAgentsResponse,
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
) -> MarketplaceAgentsResponse:
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

    return MarketplaceAgentsResponse(
        agents=[_convert_store_agent(a) for a in result.agents],
        total_count=result.pagination.total_items,
        page=result.pagination.current_page,
        page_size=result.pagination.page_size,
        total_pages=result.pagination.total_pages,
    )


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

    return _convert_store_agent_details(agent)


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
        creators=[_convert_creator(c) for c in result.creators],
        total_count=result.pagination.total_items,
        page=result.pagination.current_page,
        page_size=result.pagination.page_size,
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

    return _convert_creator_details(creator)


# ============================================================================
# Endpoints - Submissions (CRUD)
# ============================================================================


@marketplace_router.get(
    path="/submissions",
    summary="List my submissions",
    response_model=SubmissionsListResponse,
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
) -> SubmissionsListResponse:
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

    return SubmissionsListResponse(
        submissions=[_convert_submission(s) for s in result.submissions],
        total_count=result.pagination.total_items,
        page=result.pagination.current_page,
        page_size=result.pagination.page_size,
        total_pages=result.pagination.total_pages,
    )


@marketplace_router.post(
    path="/submissions",
    summary="Create a submission",
    response_model=MarketplaceSubmission,
)
async def create_submission(
    request: CreateSubmissionRequest,
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.WRITE_STORE)
    ),
) -> MarketplaceSubmission:
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

    return _convert_submission(submission)


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
