import logging
import tempfile
import urllib.parse

import autogpt_libs.auth
import fastapi
import fastapi.responses
import prisma.enums
from fastapi import Query, Security
from pydantic import BaseModel

import backend.data.graph
import backend.util.json
from backend.util.exceptions import NotFoundError
from backend.util.models import Pagination

from . import cache as store_cache
from . import db as store_db
from . import hybrid_search as store_hybrid_search
from . import image_gen as store_image_gen
from . import media as store_media
from . import model as store_model

logger = logging.getLogger(__name__)

router = fastapi.APIRouter()


##############################################
############### Profile Endpoints ############
##############################################


@router.get(
    "/profile",
    summary="Get user profile",
    tags=["store", "private"],
    dependencies=[Security(autogpt_libs.auth.requires_user)],
)
async def get_profile(
    user_id: str = Security(autogpt_libs.auth.get_user_id),
) -> store_model.ProfileDetails:
    """Get the profile details for the authenticated user."""
    profile = await store_db.get_user_profile(user_id)
    if profile is None:
        raise NotFoundError("User does not have a profile yet")
    return profile


@router.post(
    "/profile",
    summary="Update user profile",
    tags=["store", "private"],
    dependencies=[Security(autogpt_libs.auth.requires_user)],
)
async def update_or_create_profile(
    profile: store_model.Profile,
    user_id: str = Security(autogpt_libs.auth.get_user_id),
) -> store_model.ProfileDetails:
    """Update the store profile for the authenticated user."""
    updated_profile = await store_db.update_profile(user_id=user_id, profile=profile)
    return updated_profile


##############################################
############### Search Endpoints #############
##############################################


@router.get(
    "/search",
    summary="Unified search across all content types",
    tags=["store", "public"],
)
async def unified_search(
    query: str,
    content_types: list[prisma.enums.ContentType] | None = Query(
        default=None,
        description="Content types to search. If not specified, searches all.",
    ),
    page: int = Query(ge=1, default=1),
    page_size: int = Query(ge=1, default=20),
    user_id: str | None = Security(
        autogpt_libs.auth.get_optional_user_id, use_cache=False
    ),
) -> store_model.UnifiedSearchResponse:
    """
    Search across all content types (marketplace agents, blocks, documentation)
    using hybrid search.

    Combines semantic (embedding-based) and lexical (text-based) search for best results.
    """

    # Perform unified hybrid search
    results, total = await store_hybrid_search.unified_hybrid_search(
        query=query,
        content_types=content_types,
        user_id=user_id,
        page=page,
        page_size=page_size,
    )

    # Convert results to response model
    search_results = [
        store_model.UnifiedSearchResult(
            content_type=r["content_type"],
            content_id=r["content_id"],
            searchable_text=r.get("searchable_text", ""),
            metadata=r.get("metadata"),
            updated_at=r.get("updated_at"),
            combined_score=r.get("combined_score"),
            semantic_score=r.get("semantic_score"),
            lexical_score=r.get("lexical_score"),
        )
        for r in results
    ]

    total_pages = (total + page_size - 1) // page_size if total > 0 else 0

    return store_model.UnifiedSearchResponse(
        results=search_results,
        pagination=Pagination(
            total_items=total,
            total_pages=total_pages,
            current_page=page,
            page_size=page_size,
        ),
    )


##############################################
############### Agent Endpoints ##############
##############################################


@router.get(
    "/agents",
    summary="List store agents",
    tags=["store", "public"],
)
async def get_agents(
    featured: bool = Query(
        default=False, description="Filter to only show featured agents"
    ),
    creator: str | None = Query(
        default=None, description="Filter agents by creator username"
    ),
    category: str | None = Query(default=None, description="Filter agents by category"),
    search_query: str | None = Query(
        default=None, description="Literal + semantic search on names and descriptions"
    ),
    sorted_by: store_db.StoreAgentsSortOptions | None = Query(
        default=None,
        description="Property to sort results by. Ignored if search_query is provided.",
    ),
    page: int = Query(ge=1, default=1),
    page_size: int = Query(ge=1, default=20),
) -> store_model.StoreAgentsResponse:
    """
    Get a paginated list of agents from the marketplace,
    with optional filtering and sorting.

    Used for:
    - Home Page Featured Agents
    - Home Page Top Agents
    - Search Results
    - Agent Details - Other Agents By Creator
    - Agent Details - Similar Agents
    - Creator Details - Agents By Creator
    """
    agents = await store_cache._get_cached_store_agents(
        featured=featured,
        creator=creator,
        sorted_by=sorted_by,
        search_query=search_query,
        category=category,
        page=page,
        page_size=page_size,
    )
    return agents


@router.get(
    "/agents/{username}/{agent_name}",
    summary="Get specific agent",
    tags=["store", "public"],
)
async def get_agent_by_name(
    username: str,
    agent_name: str,
    include_changelog: bool = Query(default=False),
) -> store_model.StoreAgentDetails:
    """Get details of a marketplace agent"""
    username = urllib.parse.unquote(username).lower()
    # URL decode the agent name since it comes from the URL path
    agent_name = urllib.parse.unquote(agent_name).lower()
    agent = await store_cache._get_cached_agent_details(
        username=username, agent_name=agent_name, include_changelog=include_changelog
    )
    return agent


@router.post(
    "/agents/{username}/{agent_name}/review",
    summary="Create agent review",
    tags=["store"],
    dependencies=[Security(autogpt_libs.auth.requires_user)],
)
async def post_user_review_for_agent(
    username: str,
    agent_name: str,
    review: store_model.StoreReviewCreate,
    user_id: str = Security(autogpt_libs.auth.get_user_id),
) -> store_model.StoreReview:
    """Post a user review on a marketplace agent listing"""
    username = urllib.parse.unquote(username).lower()
    agent_name = urllib.parse.unquote(agent_name).lower()

    created_review = await store_db.create_store_review(
        user_id=user_id,
        store_listing_version_id=review.store_listing_version_id,
        score=review.score,
        comments=review.comments,
    )
    return created_review


@router.get(
    "/listings/versions/{store_listing_version_id}",
    summary="Get agent by version",
    tags=["store"],
    dependencies=[Security(autogpt_libs.auth.requires_user)],
)
async def get_agent_by_listing_version(
    store_listing_version_id: str,
) -> store_model.StoreAgentDetails:
    agent = await store_db.get_store_agent_by_version_id(store_listing_version_id)
    return agent


@router.get(
    "/listings/versions/{store_listing_version_id}/graph",
    summary="Get agent graph",
    tags=["store"],
    dependencies=[Security(autogpt_libs.auth.requires_user)],
)
async def get_graph_meta_by_store_listing_version_id(
    store_listing_version_id: str,
) -> backend.data.graph.GraphModelWithoutNodes:
    """Get outline of graph belonging to a specific marketplace listing version"""
    graph = await store_db.get_available_graph(store_listing_version_id)
    return graph


@router.get(
    "/listings/versions/{store_listing_version_id}/graph/download",
    summary="Download agent file",
    tags=["store", "public"],
)
async def download_agent_file(
    store_listing_version_id: str,
) -> fastapi.responses.FileResponse:
    """Download agent graph file for a specific marketplace listing version"""
    graph_data = await store_db.get_agent(store_listing_version_id)
    file_name = f"agent_{graph_data.id}_v{graph_data.version or 'latest'}.json"

    # Sending graph as a stream (similar to marketplace v1)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp_file:
        tmp_file.write(backend.util.json.dumps(graph_data))
        tmp_file.flush()

        return fastapi.responses.FileResponse(
            tmp_file.name, filename=file_name, media_type="application/json"
        )


##############################################
############# Creator Endpoints #############
##############################################


@router.get(
    "/creators",
    summary="List store creators",
    tags=["store", "public"],
)
async def get_creators(
    featured: bool = Query(
        default=False, description="Filter to only show featured creators"
    ),
    search_query: str | None = Query(
        default=None, description="Literal + semantic search on names and descriptions"
    ),
    sorted_by: store_db.StoreCreatorsSortOptions | None = None,
    page: int = Query(ge=1, default=1),
    page_size: int = Query(ge=1, default=20),
) -> store_model.CreatorsResponse:
    """List or search marketplace creators"""
    creators = await store_cache._get_cached_store_creators(
        featured=featured,
        search_query=search_query,
        sorted_by=sorted_by,
        page=page,
        page_size=page_size,
    )
    return creators


@router.get(
    "/creators/{username}",
    summary="Get creator details",
    tags=["store", "public"],
)
async def get_creator(username: str) -> store_model.CreatorDetails:
    """Get details on a marketplace creator"""
    username = urllib.parse.unquote(username).lower()
    creator = await store_cache._get_cached_creator_details(username=username)
    return creator


############################################
############# Store Submissions ###############
############################################


@router.get(
    "/my-unpublished-agents",
    summary="Get my agents",
    tags=["store", "private"],
    dependencies=[Security(autogpt_libs.auth.requires_user)],
)
async def get_my_unpublished_agents(
    user_id: str = Security(autogpt_libs.auth.get_user_id),
    page: int = Query(ge=1, default=1),
    page_size: int = Query(ge=1, default=20),
) -> store_model.MyUnpublishedAgentsResponse:
    """List the authenticated user's unpublished agents"""
    agents = await store_db.get_my_agents(user_id, page=page, page_size=page_size)
    return agents


@router.delete(
    "/submissions/{submission_id}",
    summary="Delete store submission",
    tags=["store", "private"],
    dependencies=[Security(autogpt_libs.auth.requires_user)],
)
async def delete_submission(
    submission_id: str,
    user_id: str = Security(autogpt_libs.auth.get_user_id),
) -> bool:
    """Delete a marketplace listing submission"""
    result = await store_db.delete_store_submission(
        user_id=user_id,
        submission_id=submission_id,
    )
    return result


@router.get(
    "/submissions",
    summary="List my submissions",
    tags=["store", "private"],
    dependencies=[Security(autogpt_libs.auth.requires_user)],
)
async def get_submissions(
    user_id: str = Security(autogpt_libs.auth.get_user_id),
    page: int = Query(ge=1, default=1),
    page_size: int = Query(ge=1, default=20),
) -> store_model.StoreSubmissionsResponse:
    """List the authenticated user's marketplace listing submissions"""
    listings = await store_db.get_store_submissions(
        user_id=user_id,
        page=page,
        page_size=page_size,
    )
    return listings


@router.post(
    "/submissions",
    summary="Create store submission",
    tags=["store", "private"],
    dependencies=[Security(autogpt_libs.auth.requires_user)],
)
async def create_submission(
    submission_request: store_model.StoreSubmissionRequest,
    user_id: str = Security(autogpt_libs.auth.get_user_id),
) -> store_model.StoreSubmission:
    """Submit a new marketplace listing for review"""
    result = await store_db.create_store_submission(
        user_id=user_id,
        graph_id=submission_request.graph_id,
        graph_version=submission_request.graph_version,
        slug=submission_request.slug,
        name=submission_request.name,
        video_url=submission_request.video_url,
        agent_output_demo_url=submission_request.agent_output_demo_url,
        image_urls=submission_request.image_urls,
        description=submission_request.description,
        instructions=submission_request.instructions,
        sub_heading=submission_request.sub_heading,
        categories=submission_request.categories,
        changes_summary=submission_request.changes_summary or "Initial Submission",
        recommended_schedule_cron=submission_request.recommended_schedule_cron,
    )
    return result


@router.put(
    "/submissions/{store_listing_version_id}",
    summary="Edit store submission",
    tags=["store", "private"],
    dependencies=[Security(autogpt_libs.auth.requires_user)],
)
async def edit_submission(
    store_listing_version_id: str,
    submission_request: store_model.StoreSubmissionEditRequest,
    user_id: str = Security(autogpt_libs.auth.get_user_id),
) -> store_model.StoreSubmission:
    """Update a pending marketplace listing submission"""
    result = await store_db.edit_store_submission(
        user_id=user_id,
        store_listing_version_id=store_listing_version_id,
        name=submission_request.name,
        video_url=submission_request.video_url,
        agent_output_demo_url=submission_request.agent_output_demo_url,
        image_urls=submission_request.image_urls,
        description=submission_request.description,
        instructions=submission_request.instructions,
        sub_heading=submission_request.sub_heading,
        categories=submission_request.categories,
        changes_summary=submission_request.changes_summary,
        recommended_schedule_cron=submission_request.recommended_schedule_cron,
    )
    return result


@router.post(
    "/submissions/media",
    summary="Upload submission media",
    tags=["store", "private"],
    dependencies=[Security(autogpt_libs.auth.requires_user)],
)
async def upload_submission_media(
    file: fastapi.UploadFile,
    user_id: str = Security(autogpt_libs.auth.get_user_id),
) -> str:
    """Upload media for a marketplace listing submission"""
    media_url = await store_media.upload_media(user_id=user_id, file=file)
    return media_url


class ImageURLResponse(BaseModel):
    image_url: str


@router.post(
    "/submissions/generate_image",
    summary="Generate submission image",
    tags=["store", "private"],
    dependencies=[Security(autogpt_libs.auth.requires_user)],
)
async def generate_image(
    graph_id: str,
    user_id: str = Security(autogpt_libs.auth.get_user_id),
) -> ImageURLResponse:
    """
    Generate an image for a marketplace listing submission based on the properties
    of a given graph.
    """
    graph = await backend.data.graph.get_graph(
        graph_id=graph_id, version=None, user_id=user_id
    )

    if not graph:
        raise NotFoundError(f"Agent graph #{graph_id} not found")
    # Use .jpeg here since we are generating JPEG images
    filename = f"agent_{graph_id}.jpeg"

    existing_url = await store_media.check_media_exists(user_id, filename)
    if existing_url:
        logger.info(f"Using existing image for agent graph {graph_id}")
        return ImageURLResponse(image_url=existing_url)
    # Generate agent image as JPEG
    image = await store_image_gen.generate_agent_image(agent=graph)

    # Create UploadFile with the correct filename and content_type
    image_file = fastapi.UploadFile(
        file=image,
        filename=filename,
    )
    image_url = await store_media.upload_media(
        user_id=user_id, file=image_file, use_file_name=True
    )

    return ImageURLResponse(image_url=image_url)


##############################################
############### Cache Management #############
##############################################


@router.get(
    "/metrics/cache",
    summary="Get cache metrics in Prometheus format",
    tags=["store", "metrics"],
    response_class=fastapi.responses.PlainTextResponse,
)
async def get_cache_metrics():
    """
    Get cache metrics in Prometheus text format.

    Returns Prometheus-compatible metrics for monitoring cache performance.
    Metrics include size, maxsize, TTL, and hit rate for each cache.

    Returns:
        str: Prometheus-formatted metrics text
    """
    metrics = []

    # Helper to add metrics for a cache
    def add_cache_metrics(cache_name: str, cache_func):
        info = cache_func.cache_info()
        # Cache size metric (dynamic - changes as items are cached/expired)
        metrics.append(f'store_cache_entries{{cache="{cache_name}"}} {info["size"]}')
        # Cache utilization percentage (dynamic - useful for monitoring)
        utilization = (
            (info["size"] / info["maxsize"] * 100) if info["maxsize"] > 0 else 0
        )
        metrics.append(
            f'store_cache_utilization_percent{{cache="{cache_name}"}} {utilization:.2f}'
        )

    # Add metrics for each cache
    add_cache_metrics("store_agents", store_cache._get_cached_store_agents)
    add_cache_metrics("agent_details", store_cache._get_cached_agent_details)
    add_cache_metrics("store_creators", store_cache._get_cached_store_creators)
    add_cache_metrics("creator_details", store_cache._get_cached_creator_details)

    # Add metadata/help text at the beginning
    prometheus_output = [
        "# HELP store_cache_entries Number of entries currently in cache",
        "# TYPE store_cache_entries gauge",
        "# HELP store_cache_utilization_percent Cache utilization as percentage (0-100)",
        "# TYPE store_cache_utilization_percent gauge",
        "",  # Empty line before metrics
    ]
    prometheus_output.extend(metrics)

    return "\n".join(prometheus_output)
