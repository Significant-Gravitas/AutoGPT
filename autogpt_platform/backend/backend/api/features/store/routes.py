import logging
import tempfile
import typing
import urllib.parse
from typing import Literal

import autogpt_libs.auth
import fastapi
import fastapi.responses
import prisma.enums

import backend.data.graph
import backend.util.json
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
    dependencies=[fastapi.Security(autogpt_libs.auth.requires_user)],
    response_model=store_model.ProfileDetails,
)
async def get_profile(
    user_id: str = fastapi.Security(autogpt_libs.auth.get_user_id),
):
    """
    Get the profile details for the authenticated user.
    Cached for 1 hour per user.
    """
    profile = await store_db.get_user_profile(user_id)
    if profile is None:
        return fastapi.responses.JSONResponse(
            status_code=404,
            content={"detail": "Profile not found"},
        )
    return profile


@router.post(
    "/profile",
    summary="Update user profile",
    tags=["store", "private"],
    dependencies=[fastapi.Security(autogpt_libs.auth.requires_user)],
    response_model=store_model.CreatorDetails,
)
async def update_or_create_profile(
    profile: store_model.Profile,
    user_id: str = fastapi.Security(autogpt_libs.auth.get_user_id),
):
    """
    Update the store profile for the authenticated user.

    Args:
        profile (Profile): The updated profile details
        user_id (str): ID of the authenticated user

    Returns:
        CreatorDetails: The updated profile

    Raises:
        HTTPException: If there is an error updating the profile
    """
    updated_profile = await store_db.update_profile(user_id=user_id, profile=profile)
    return updated_profile


##############################################
############### Agent Endpoints ##############
##############################################


@router.get(
    "/agents",
    summary="List store agents",
    tags=["store", "public"],
    response_model=store_model.StoreAgentsResponse,
)
async def get_agents(
    featured: bool = False,
    creator: str | None = None,
    sorted_by: Literal["rating", "runs", "name", "updated_at"] | None = None,
    search_query: str | None = None,
    category: str | None = None,
    page: int = 1,
    page_size: int = 20,
):
    """
    Get a paginated list of agents from the store with optional filtering and sorting.

    Args:
        featured (bool, optional): Filter to only show featured agents. Defaults to False.
        creator (str | None, optional): Filter agents by creator username. Defaults to None.
        sorted_by (str | None, optional): Sort agents by "runs" or "rating". Defaults to None.
        search_query (str | None, optional): Search agents by name, subheading and description. Defaults to None.
        category (str | None, optional): Filter agents by category. Defaults to None.
        page (int, optional): Page number for pagination. Defaults to 1.
        page_size (int, optional): Number of agents per page. Defaults to 20.

    Returns:
        StoreAgentsResponse: Paginated list of agents matching the filters

    Raises:
        HTTPException: If page or page_size are less than 1

    Used for:
    - Home Page Featured Agents
    - Home Page Top Agents
    - Search Results
    - Agent Details - Other Agents By Creator
    - Agent Details - Similar Agents
    - Creator Details - Agents By Creator
    """
    if page < 1:
        raise fastapi.HTTPException(
            status_code=422, detail="Page must be greater than 0"
        )

    if page_size < 1:
        raise fastapi.HTTPException(
            status_code=422, detail="Page size must be greater than 0"
        )

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


##############################################
############### Search Endpoints #############
##############################################


@router.get(
    "/search",
    summary="Unified search across all content types",
    tags=["store", "public"],
    response_model=store_model.UnifiedSearchResponse,
)
async def unified_search(
    query: str,
    content_types: list[str] | None = fastapi.Query(
        default=None,
        description="Content types to search: STORE_AGENT, BLOCK, DOCUMENTATION. If not specified, searches all.",
    ),
    page: int = 1,
    page_size: int = 20,
    user_id: str | None = fastapi.Security(
        autogpt_libs.auth.get_optional_user_id, use_cache=False
    ),
):
    """
    Search across all content types (store agents, blocks, documentation) using hybrid search.

    Combines semantic (embedding-based) and lexical (text-based) search for best results.

    Args:
        query: The search query string
        content_types: Optional list of content types to filter by (STORE_AGENT, BLOCK, DOCUMENTATION)
        page: Page number for pagination (default 1)
        page_size: Number of results per page (default 20)
        user_id: Optional authenticated user ID (for user-scoped content in future)

    Returns:
        UnifiedSearchResponse: Paginated list of search results with relevance scores
    """
    if page < 1:
        raise fastapi.HTTPException(
            status_code=422, detail="Page must be greater than 0"
        )

    if page_size < 1:
        raise fastapi.HTTPException(
            status_code=422, detail="Page size must be greater than 0"
        )

    # Convert string content types to enum
    content_type_enums: list[prisma.enums.ContentType] | None = None
    if content_types:
        try:
            content_type_enums = [prisma.enums.ContentType(ct) for ct in content_types]
        except ValueError as e:
            raise fastapi.HTTPException(
                status_code=422,
                detail=f"Invalid content type. Valid values: STORE_AGENT, BLOCK, DOCUMENTATION. Error: {e}",
            )

    # Perform unified hybrid search
    results, total = await store_hybrid_search.unified_hybrid_search(
        query=query,
        content_types=content_type_enums,
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


@router.get(
    "/agents/{username}/{agent_name}",
    summary="Get specific agent",
    tags=["store", "public"],
    response_model=store_model.StoreAgentDetails,
)
async def get_agent(
    username: str,
    agent_name: str,
    include_changelog: bool = fastapi.Query(default=False),
):
    """
    This is only used on the AgentDetails Page.

    It returns the store listing agents details.
    """
    username = urllib.parse.unquote(username).lower()
    # URL decode the agent name since it comes from the URL path
    agent_name = urllib.parse.unquote(agent_name).lower()
    agent = await store_cache._get_cached_agent_details(
        username=username, agent_name=agent_name, include_changelog=include_changelog
    )
    return agent


@router.get(
    "/graph/{store_listing_version_id}",
    summary="Get agent graph",
    tags=["store"],
    dependencies=[fastapi.Security(autogpt_libs.auth.requires_user)],
)
async def get_graph_meta_by_store_listing_version_id(
    store_listing_version_id: str,
) -> backend.data.graph.GraphMeta:
    """
    Get Agent Graph from Store Listing Version ID.
    """
    graph = await store_db.get_available_graph(store_listing_version_id)
    return graph


@router.get(
    "/agents/{store_listing_version_id}",
    summary="Get agent by version",
    tags=["store"],
    dependencies=[fastapi.Security(autogpt_libs.auth.requires_user)],
    response_model=store_model.StoreAgentDetails,
)
async def get_store_agent(store_listing_version_id: str):
    """
    Get Store Agent Details from Store Listing Version ID.
    """
    agent = await store_db.get_store_agent_by_version_id(store_listing_version_id)

    return agent


@router.post(
    "/agents/{username}/{agent_name}/review",
    summary="Create agent review",
    tags=["store"],
    dependencies=[fastapi.Security(autogpt_libs.auth.requires_user)],
    response_model=store_model.StoreReview,
)
async def create_review(
    username: str,
    agent_name: str,
    review: store_model.StoreReviewCreate,
    user_id: str = fastapi.Security(autogpt_libs.auth.get_user_id),
):
    """
    Create a review for a store agent.

    Args:
        username: Creator's username
        agent_name: Name/slug of the agent
        review: Review details including score and optional comments
        user_id: ID of authenticated user creating the review

    Returns:
        The created review
    """
    username = urllib.parse.unquote(username).lower()
    agent_name = urllib.parse.unquote(agent_name).lower()
    # Create the review
    created_review = await store_db.create_store_review(
        user_id=user_id,
        store_listing_version_id=review.store_listing_version_id,
        score=review.score,
        comments=review.comments,
    )

    return created_review


##############################################
############# Creator Endpoints #############
##############################################


@router.get(
    "/creators",
    summary="List store creators",
    tags=["store", "public"],
    response_model=store_model.CreatorsResponse,
)
async def get_creators(
    featured: bool = False,
    search_query: str | None = None,
    sorted_by: Literal["agent_rating", "agent_runs", "num_agents"] | None = None,
    page: int = 1,
    page_size: int = 20,
):
    """
    This is needed for:
    - Home Page Featured Creators
    - Search Results Page

    ---

    To support this functionality we need:
    - featured: bool - to limit the list to just featured agents
    - search_query: str - vector search based on the creators profile description.
    - sorted_by: [agent_rating, agent_runs] -
    """
    if page < 1:
        raise fastapi.HTTPException(
            status_code=422, detail="Page must be greater than 0"
        )

    if page_size < 1:
        raise fastapi.HTTPException(
            status_code=422, detail="Page size must be greater than 0"
        )

    creators = await store_cache._get_cached_store_creators(
        featured=featured,
        search_query=search_query,
        sorted_by=sorted_by,
        page=page,
        page_size=page_size,
    )
    return creators


@router.get(
    "/creator/{username}",
    summary="Get creator details",
    tags=["store", "public"],
    response_model=store_model.CreatorDetails,
)
async def get_creator(
    username: str,
):
    """
    Get the details of a creator.
    - Creator Details Page
    """
    username = urllib.parse.unquote(username).lower()
    creator = await store_cache._get_cached_creator_details(username=username)
    return creator


############################################
############# Store Submissions ###############
############################################


@router.get(
    "/myagents",
    summary="Get my agents",
    tags=["store", "private"],
    dependencies=[fastapi.Security(autogpt_libs.auth.requires_user)],
    response_model=store_model.MyAgentsResponse,
)
async def get_my_agents(
    user_id: str = fastapi.Security(autogpt_libs.auth.get_user_id),
    page: typing.Annotated[int, fastapi.Query(ge=1)] = 1,
    page_size: typing.Annotated[int, fastapi.Query(ge=1)] = 20,
):
    """
    Get user's own agents.
    """
    agents = await store_db.get_my_agents(user_id, page=page, page_size=page_size)
    return agents


@router.delete(
    "/submissions/{submission_id}",
    summary="Delete store submission",
    tags=["store", "private"],
    dependencies=[fastapi.Security(autogpt_libs.auth.requires_user)],
    response_model=bool,
)
async def delete_submission(
    submission_id: str,
    user_id: str = fastapi.Security(autogpt_libs.auth.get_user_id),
):
    """
    Delete a store listing submission.

    Args:
        user_id (str): ID of the authenticated user
        submission_id (str): ID of the submission to be deleted

    Returns:
        bool: True if the submission was successfully deleted, False otherwise
    """
    result = await store_db.delete_store_submission(
        user_id=user_id,
        submission_id=submission_id,
    )

    return result


@router.get(
    "/submissions",
    summary="List my submissions",
    tags=["store", "private"],
    dependencies=[fastapi.Security(autogpt_libs.auth.requires_user)],
    response_model=store_model.StoreSubmissionsResponse,
)
async def get_submissions(
    user_id: str = fastapi.Security(autogpt_libs.auth.get_user_id),
    page: int = 1,
    page_size: int = 20,
):
    """
    Get a paginated list of store submissions for the authenticated user.

    Args:
        user_id (str): ID of the authenticated user
        page (int, optional): Page number for pagination. Defaults to 1.
        page_size (int, optional): Number of submissions per page. Defaults to 20.

    Returns:
        StoreListingsResponse: Paginated list of store submissions

    Raises:
        HTTPException: If page or page_size are less than 1
    """
    if page < 1:
        raise fastapi.HTTPException(
            status_code=422, detail="Page must be greater than 0"
        )

    if page_size < 1:
        raise fastapi.HTTPException(
            status_code=422, detail="Page size must be greater than 0"
        )
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
    dependencies=[fastapi.Security(autogpt_libs.auth.requires_user)],
    response_model=store_model.StoreSubmission,
)
async def create_submission(
    submission_request: store_model.StoreSubmissionRequest,
    user_id: str = fastapi.Security(autogpt_libs.auth.get_user_id),
):
    """
    Create a new store listing submission.

    Args:
        submission_request (StoreSubmissionRequest): The submission details
        user_id (str): ID of the authenticated user submitting the listing

    Returns:
        StoreSubmission: The created store submission

    Raises:
        HTTPException: If there is an error creating the submission
    """
    result = await store_db.create_store_submission(
        user_id=user_id,
        agent_id=submission_request.agent_id,
        agent_version=submission_request.agent_version,
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
    dependencies=[fastapi.Security(autogpt_libs.auth.requires_user)],
    response_model=store_model.StoreSubmission,
)
async def edit_submission(
    store_listing_version_id: str,
    submission_request: store_model.StoreSubmissionEditRequest,
    user_id: str = fastapi.Security(autogpt_libs.auth.get_user_id),
):
    """
    Edit an existing store listing submission.

    Args:
        store_listing_version_id (str): ID of the store listing version to edit
        submission_request (StoreSubmissionRequest): The updated submission details
        user_id (str): ID of the authenticated user editing the listing

    Returns:
        StoreSubmission: The updated store submission

    Raises:
        HTTPException: If there is an error editing the submission
    """
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
    dependencies=[fastapi.Security(autogpt_libs.auth.requires_user)],
)
async def upload_submission_media(
    file: fastapi.UploadFile,
    user_id: str = fastapi.Security(autogpt_libs.auth.get_user_id),
):
    """
    Upload media (images/videos) for a store listing submission.

    Args:
        file (UploadFile): The media file to upload
        user_id (str): ID of the authenticated user uploading the media

    Returns:
        str: URL of the uploaded media file

    Raises:
        HTTPException: If there is an error uploading the media
    """
    media_url = await store_media.upload_media(user_id=user_id, file=file)
    return media_url


@router.post(
    "/submissions/generate_image",
    summary="Generate submission image",
    tags=["store", "private"],
    dependencies=[fastapi.Security(autogpt_libs.auth.requires_user)],
)
async def generate_image(
    agent_id: str,
    user_id: str = fastapi.Security(autogpt_libs.auth.get_user_id),
) -> fastapi.responses.Response:
    """
    Generate an image for a store listing submission.

    Args:
        agent_id (str): ID of the agent to generate an image for
        user_id (str): ID of the authenticated user

    Returns:
        JSONResponse: JSON containing the URL of the generated image
    """
    agent = await backend.data.graph.get_graph(
        graph_id=agent_id, version=None, user_id=user_id
    )

    if not agent:
        raise fastapi.HTTPException(
            status_code=404, detail=f"Agent with ID {agent_id} not found"
        )
    # Use .jpeg here since we are generating JPEG images
    filename = f"agent_{agent_id}.jpeg"

    existing_url = await store_media.check_media_exists(user_id, filename)
    if existing_url:
        logger.info(f"Using existing image for agent {agent_id}")
        return fastapi.responses.JSONResponse(content={"image_url": existing_url})
    # Generate agent image as JPEG
    image = await store_image_gen.generate_agent_image(agent=agent)

    # Create UploadFile with the correct filename and content_type
    image_file = fastapi.UploadFile(
        file=image,
        filename=filename,
    )

    image_url = await store_media.upload_media(
        user_id=user_id, file=image_file, use_file_name=True
    )

    return fastapi.responses.JSONResponse(content={"image_url": image_url})


@router.get(
    "/download/agents/{store_listing_version_id}",
    summary="Download agent file",
    tags=["store", "public"],
)
async def download_agent_file(
    store_listing_version_id: str = fastapi.Path(
        ..., description="The ID of the agent to download"
    ),
) -> fastapi.responses.FileResponse:
    """
    Download the agent file by streaming its content.

    Args:
        store_listing_version_id (str): The ID of the agent to download

    Returns:
        StreamingResponse: A streaming response containing the agent's graph data.

    Raises:
        HTTPException: If the agent is not found or an unexpected error occurs.
    """
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
