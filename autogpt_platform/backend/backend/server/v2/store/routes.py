import logging
import tempfile
import typing
import urllib.parse

import autogpt_libs.auth
import fastapi
import fastapi.responses

import backend.data.graph
import backend.server.v2.store.db
import backend.server.v2.store.exceptions
import backend.server.v2.store.image_gen
import backend.server.v2.store.media
import backend.server.v2.store.model
import backend.util.json

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
    response_model=backend.server.v2.store.model.ProfileDetails,
)
async def get_profile(
    user_id: str = fastapi.Security(autogpt_libs.auth.get_user_id),
):
    """
    Get the profile details for the authenticated user.
    """
    try:
        profile = await backend.server.v2.store.db.get_user_profile(user_id)
        if profile is None:
            return fastapi.responses.JSONResponse(
                status_code=404,
                content={"detail": "Profile not found"},
            )
        return profile
    except Exception as e:
        logger.exception("Failed to fetch user profile for %s: %s", user_id, e)
        return fastapi.responses.JSONResponse(
            status_code=500,
            content={
                "detail": "Failed to retrieve user profile",
                "hint": "Check database connection.",
            },
        )


@router.post(
    "/profile",
    summary="Update user profile",
    tags=["store", "private"],
    dependencies=[fastapi.Security(autogpt_libs.auth.requires_user)],
    response_model=backend.server.v2.store.model.CreatorDetails,
)
async def update_or_create_profile(
    profile: backend.server.v2.store.model.Profile,
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
    try:
        updated_profile = await backend.server.v2.store.db.update_profile(
            user_id=user_id, profile=profile
        )
        return updated_profile
    except Exception as e:
        logger.exception("Failed to update profile for user %s: %s", user_id, e)
        return fastapi.responses.JSONResponse(
            status_code=500,
            content={
                "detail": "Failed to update user profile",
                "hint": "Validate request data.",
            },
        )


##############################################
############### Agent Endpoints ##############
##############################################


@router.get(
    "/agents",
    summary="List store agents",
    tags=["store", "public"],
    response_model=backend.server.v2.store.model.StoreAgentsResponse,
)
async def get_agents(
    featured: bool = False,
    creator: str | None = None,
    sorted_by: str | None = None,
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

    try:
        agents = await backend.server.v2.store.db.get_store_agents(
            featured=featured,
            creators=[creator] if creator else None,
            sorted_by=sorted_by,
            search_query=search_query,
            category=category,
            page=page,
            page_size=page_size,
        )
        return agents
    except Exception as e:
        logger.exception("Failed to retrieve store agents: %s", e)
        return fastapi.responses.JSONResponse(
            status_code=500,
            content={
                "detail": "Failed to retrieve store agents",
                "hint": "Check database or search parameters.",
            },
        )


@router.get(
    "/agents/{username}/{agent_name}",
    summary="Get specific agent",
    tags=["store", "public"],
    response_model=backend.server.v2.store.model.StoreAgentDetails,
)
async def get_agent(username: str, agent_name: str):
    """
    This is only used on the AgentDetails Page

    It returns the store listing agents details.
    """
    try:
        username = urllib.parse.unquote(username).lower()
        # URL decode the agent name since it comes from the URL path
        agent_name = urllib.parse.unquote(agent_name).lower()
        agent = await backend.server.v2.store.db.get_store_agent_details(
            username=username, agent_name=agent_name
        )
        return agent
    except Exception:
        logger.exception("Exception occurred whilst getting store agent details")
        return fastapi.responses.JSONResponse(
            status_code=500,
            content={
                "detail": "An error occurred while retrieving the store agent details"
            },
        )


@router.get(
    "/graph/{store_listing_version_id}",
    summary="Get agent graph",
    tags=["store"],
    dependencies=[fastapi.Security(autogpt_libs.auth.requires_user)],
)
async def get_graph_meta_by_store_listing_version_id(store_listing_version_id: str):
    """
    Get Agent Graph from Store Listing Version ID.
    """
    try:
        graph = await backend.server.v2.store.db.get_available_graph(
            store_listing_version_id
        )
        return graph
    except Exception:
        logger.exception("Exception occurred whilst getting agent graph")
        return fastapi.responses.JSONResponse(
            status_code=500,
            content={"detail": "An error occurred while retrieving the agent graph"},
        )


@router.get(
    "/agents/{store_listing_version_id}",
    summary="Get agent by version",
    tags=["store"],
    dependencies=[fastapi.Security(autogpt_libs.auth.requires_user)],
    response_model=backend.server.v2.store.model.StoreAgentDetails,
)
async def get_store_agent(store_listing_version_id: str):
    """
    Get Store Agent Details from Store Listing Version ID.
    """
    try:
        agent = await backend.server.v2.store.db.get_store_agent_by_version_id(
            store_listing_version_id
        )
        return agent
    except Exception:
        logger.exception("Exception occurred whilst getting store agent")
        return fastapi.responses.JSONResponse(
            status_code=500,
            content={"detail": "An error occurred while retrieving the store agent"},
        )


@router.post(
    "/agents/{username}/{agent_name}/review",
    summary="Create agent review",
    tags=["store"],
    dependencies=[fastapi.Security(autogpt_libs.auth.requires_user)],
    response_model=backend.server.v2.store.model.StoreReview,
)
async def create_review(
    username: str,
    agent_name: str,
    review: backend.server.v2.store.model.StoreReviewCreate,
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
    try:
        username = urllib.parse.unquote(username).lower()
        agent_name = urllib.parse.unquote(agent_name)
        # Create the review
        created_review = await backend.server.v2.store.db.create_store_review(
            user_id=user_id,
            store_listing_version_id=review.store_listing_version_id,
            score=review.score,
            comments=review.comments,
        )

        return created_review
    except Exception:
        logger.exception("Exception occurred whilst creating store review")
        return fastapi.responses.JSONResponse(
            status_code=500,
            content={"detail": "An error occurred while creating the store review"},
        )


##############################################
############# Creator Endpoints #############
##############################################


@router.get(
    "/creators",
    summary="List store creators",
    tags=["store", "public"],
    response_model=backend.server.v2.store.model.CreatorsResponse,
)
async def get_creators(
    featured: bool = False,
    search_query: str | None = None,
    sorted_by: str | None = None,
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

    try:
        creators = await backend.server.v2.store.db.get_store_creators(
            featured=featured,
            search_query=search_query,
            sorted_by=sorted_by,
            page=page,
            page_size=page_size,
        )
        return creators
    except Exception:
        logger.exception("Exception occurred whilst getting store creators")
        return fastapi.responses.JSONResponse(
            status_code=500,
            content={"detail": "An error occurred while retrieving the store creators"},
        )


@router.get(
    "/creator/{username}",
    summary="Get creator details",
    tags=["store", "public"],
    response_model=backend.server.v2.store.model.CreatorDetails,
)
async def get_creator(
    username: str,
):
    """
    Get the details of a creator
    - Creator Details Page
    """
    try:
        username = urllib.parse.unquote(username).lower()
        creator = await backend.server.v2.store.db.get_store_creator_details(
            username=username.lower()
        )
        return creator
    except Exception:
        logger.exception("Exception occurred whilst getting creator details")
        return fastapi.responses.JSONResponse(
            status_code=500,
            content={
                "detail": "An error occurred while retrieving the creator details"
            },
        )


############################################
############# Store Submissions ###############
############################################
@router.get(
    "/myagents",
    summary="Get my agents",
    tags=["store", "private"],
    dependencies=[fastapi.Security(autogpt_libs.auth.requires_user)],
    response_model=backend.server.v2.store.model.MyAgentsResponse,
)
async def get_my_agents(
    user_id: str = fastapi.Security(autogpt_libs.auth.get_user_id),
    page: typing.Annotated[int, fastapi.Query(ge=1)] = 1,
    page_size: typing.Annotated[int, fastapi.Query(ge=1)] = 20,
):
    try:
        agents = await backend.server.v2.store.db.get_my_agents(
            user_id, page=page, page_size=page_size
        )
        return agents
    except Exception:
        logger.exception("Exception occurred whilst getting my agents")
        return fastapi.responses.JSONResponse(
            status_code=500,
            content={"detail": "An error occurred while retrieving the my agents"},
        )


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
    try:
        result = await backend.server.v2.store.db.delete_store_submission(
            user_id=user_id,
            submission_id=submission_id,
        )
        return result
    except Exception:
        logger.exception("Exception occurred whilst deleting store submission")
        return fastapi.responses.JSONResponse(
            status_code=500,
            content={"detail": "An error occurred while deleting the store submission"},
        )


@router.get(
    "/submissions",
    summary="List my submissions",
    tags=["store", "private"],
    dependencies=[fastapi.Security(autogpt_libs.auth.requires_user)],
    response_model=backend.server.v2.store.model.StoreSubmissionsResponse,
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
    try:
        listings = await backend.server.v2.store.db.get_store_submissions(
            user_id=user_id,
            page=page,
            page_size=page_size,
        )
        return listings
    except Exception:
        logger.exception("Exception occurred whilst getting store submissions")
        return fastapi.responses.JSONResponse(
            status_code=500,
            content={
                "detail": "An error occurred while retrieving the store submissions"
            },
        )


@router.post(
    "/submissions",
    summary="Create store submission",
    tags=["store", "private"],
    dependencies=[fastapi.Security(autogpt_libs.auth.requires_user)],
    response_model=backend.server.v2.store.model.StoreSubmission,
)
async def create_submission(
    submission_request: backend.server.v2.store.model.StoreSubmissionRequest,
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
    try:
        return await backend.server.v2.store.db.create_store_submission(
            user_id=user_id,
            agent_id=submission_request.agent_id,
            agent_version=submission_request.agent_version,
            slug=submission_request.slug,
            name=submission_request.name,
            video_url=submission_request.video_url,
            image_urls=submission_request.image_urls,
            description=submission_request.description,
            instructions=submission_request.instructions,
            sub_heading=submission_request.sub_heading,
            categories=submission_request.categories,
            changes_summary=submission_request.changes_summary or "Initial Submission",
            recommended_schedule_cron=submission_request.recommended_schedule_cron,
        )
    except Exception:
        logger.exception("Exception occurred whilst creating store submission")
        return fastapi.responses.JSONResponse(
            status_code=500,
            content={"detail": "An error occurred while creating the store submission"},
        )


@router.put(
    "/submissions/{store_listing_version_id}",
    summary="Edit store submission",
    tags=["store", "private"],
    dependencies=[fastapi.Security(autogpt_libs.auth.requires_user)],
    response_model=backend.server.v2.store.model.StoreSubmission,
)
async def edit_submission(
    store_listing_version_id: str,
    submission_request: backend.server.v2.store.model.StoreSubmissionEditRequest,
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
    return await backend.server.v2.store.db.edit_store_submission(
        user_id=user_id,
        store_listing_version_id=store_listing_version_id,
        name=submission_request.name,
        video_url=submission_request.video_url,
        image_urls=submission_request.image_urls,
        description=submission_request.description,
        instructions=submission_request.instructions,
        sub_heading=submission_request.sub_heading,
        categories=submission_request.categories,
        changes_summary=submission_request.changes_summary,
        recommended_schedule_cron=submission_request.recommended_schedule_cron,
    )


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
    try:
        media_url = await backend.server.v2.store.media.upload_media(
            user_id=user_id, file=file
        )
        return media_url
    except backend.server.v2.store.exceptions.VirusDetectedError as e:
        logger.warning(f"Virus detected in uploaded file: {e.threat_name}")
        return fastapi.responses.JSONResponse(
            status_code=400,
            content={
                "detail": f"File rejected due to virus detection: {e.threat_name}",
                "error_type": "virus_detected",
                "threat_name": e.threat_name,
            },
        )
    except backend.server.v2.store.exceptions.VirusScanError as e:
        logger.error(f"Virus scanning failed: {str(e)}")
        return fastapi.responses.JSONResponse(
            status_code=503,
            content={
                "detail": "Virus scanning service unavailable. Please try again later.",
                "error_type": "virus_scan_failed",
            },
        )
    except Exception:
        logger.exception("Exception occurred whilst uploading submission media")
        return fastapi.responses.JSONResponse(
            status_code=500,
            content={"detail": "An error occurred while uploading the media file"},
        )


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
    try:
        agent = await backend.data.graph.get_graph(agent_id, user_id=user_id)

        if not agent:
            raise fastapi.HTTPException(
                status_code=404, detail=f"Agent with ID {agent_id} not found"
            )
        # Use .jpeg here since we are generating JPEG images
        filename = f"agent_{agent_id}.jpeg"

        existing_url = await backend.server.v2.store.media.check_media_exists(
            user_id, filename
        )
        if existing_url:
            logger.info(f"Using existing image for agent {agent_id}")
            return fastapi.responses.JSONResponse(content={"image_url": existing_url})
        # Generate agent image as JPEG
        image = await backend.server.v2.store.image_gen.generate_agent_image(
            agent=agent
        )

        # Create UploadFile with the correct filename and content_type
        image_file = fastapi.UploadFile(
            file=image,
            filename=filename,
        )

        image_url = await backend.server.v2.store.media.upload_media(
            user_id=user_id, file=image_file, use_file_name=True
        )

        return fastapi.responses.JSONResponse(content={"image_url": image_url})
    except Exception:
        logger.exception("Exception occurred whilst generating submission image")
        return fastapi.responses.JSONResponse(
            status_code=500,
            content={"detail": "An error occurred while generating the image"},
        )


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
    graph_data = await backend.server.v2.store.db.get_agent(store_listing_version_id)
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
