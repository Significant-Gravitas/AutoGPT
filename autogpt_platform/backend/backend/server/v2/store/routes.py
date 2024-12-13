import logging
import typing

import autogpt_libs.auth.depends
import autogpt_libs.auth.middleware
import fastapi
import fastapi.responses

import backend.server.v2.store.db
import backend.server.v2.store.media
import backend.server.v2.store.model

logger = logging.getLogger(__name__)

router = fastapi.APIRouter()


##############################################
############### Profile Endpoints ############
##############################################


@router.get("/profile", tags=["store", "private"])
async def get_profile(
    user_id: typing.Annotated[
        str, fastapi.Depends(autogpt_libs.auth.depends.get_user_id)
    ]
) -> backend.server.v2.store.model.ProfileDetails:
    """
    Get the profile details for the authenticated user.
    """
    try:
        profile = await backend.server.v2.store.db.get_user_profile(user_id)
        return profile
    except Exception:
        logger.exception("Exception occurred whilst getting user profile")
        raise


@router.post(
    "/profile",
    tags=["store", "private"],
    dependencies=[fastapi.Depends(autogpt_libs.auth.middleware.auth_middleware)],
)
async def update_or_create_profile(
    profile: backend.server.v2.store.model.Profile,
    user_id: typing.Annotated[
        str, fastapi.Depends(autogpt_libs.auth.depends.get_user_id)
    ],
) -> backend.server.v2.store.model.CreatorDetails:
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
        updated_profile = await backend.server.v2.store.db.update_or_create_profile(
            user_id=user_id, profile=profile
        )
        return updated_profile
    except Exception:
        logger.exception("Exception occurred whilst updating profile")
        raise


##############################################
############### Agent Endpoints ##############
##############################################


@router.get("/agents", tags=["store", "public"])
async def get_agents(
    featured: bool = False,
    creator: str | None = None,
    sorted_by: str | None = None,
    search_query: str | None = None,
    category: str | None = None,
    page: int = 1,
    page_size: int = 20,
) -> backend.server.v2.store.model.StoreAgentsResponse:
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
            creator=creator,
            sorted_by=sorted_by,
            search_query=search_query,
            category=category,
            page=page,
            page_size=page_size,
        )
        return agents
    except Exception:
        logger.exception("Exception occured whilst getting store agents")
        raise


@router.get("/agents/{username}/{agent_name}", tags=["store", "public"])
async def get_agent(
    username: str, agent_name: str
) -> backend.server.v2.store.model.StoreAgentDetails:
    """
    This is only used on the AgentDetails Page

    It returns the store listing agents details.
    """
    try:
        agent = await backend.server.v2.store.db.get_store_agent_details(
            username=username, agent_name=agent_name
        )
        return agent
    except Exception:
        logger.exception("Exception occurred whilst getting store agent details")
        raise


@router.post(
    "/agents/{username}/{agent_name}/review",
    tags=["store"],
    dependencies=[fastapi.Depends(autogpt_libs.auth.middleware.auth_middleware)],
)
async def create_review(
    username: str,
    agent_name: str,
    review: backend.server.v2.store.model.StoreReviewCreate,
    user_id: typing.Annotated[
        str, fastapi.Depends(autogpt_libs.auth.depends.get_user_id)
    ],
) -> backend.server.v2.store.model.StoreReview:
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
        raise


##############################################
############# Creator Endpoints #############
##############################################


@router.get("/creators", tags=["store", "public"])
async def get_creators(
    featured: bool = False,
    search_query: str | None = None,
    sorted_by: str | None = None,
    page: int = 1,
    page_size: int = 20,
) -> backend.server.v2.store.model.CreatorsResponse:
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
        raise


@router.get("/creator/{username}", tags=["store", "public"])
async def get_creator(username: str) -> backend.server.v2.store.model.CreatorDetails:
    """
    Get the details of a creator
    - Creator Details Page
    """
    try:
        creator = await backend.server.v2.store.db.get_store_creator_details(
            username=username
        )
        return creator
    except Exception:
        logger.exception("Exception occurred whilst getting creator details")
        raise


############################################
############# Store Submissions ###############
############################################
@router.get(
    "/myagents",
    tags=["store", "private"],
    dependencies=[fastapi.Depends(autogpt_libs.auth.middleware.auth_middleware)],
)
async def get_my_agents(
    user_id: typing.Annotated[
        str, fastapi.Depends(autogpt_libs.auth.depends.get_user_id)
    ]
) -> backend.server.v2.store.model.MyAgentsResponse:
    try:
        agents = await backend.server.v2.store.db.get_my_agents(user_id)
        return agents
    except Exception:
        logger.exception("Exception occurred whilst getting my agents")
        raise


@router.delete(
    "/submissions/{submission_id}",
    tags=["store", "private"],
    dependencies=[fastapi.Depends(autogpt_libs.auth.middleware.auth_middleware)],
)
async def delete_submission(
    user_id: typing.Annotated[
        str, fastapi.Depends(autogpt_libs.auth.depends.get_user_id)
    ],
    submission_id: str,
) -> bool:
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
        raise


@router.get(
    "/submissions",
    tags=["store", "private"],
    dependencies=[fastapi.Depends(autogpt_libs.auth.middleware.auth_middleware)],
)
async def get_submissions(
    user_id: typing.Annotated[
        str, fastapi.Depends(autogpt_libs.auth.depends.get_user_id)
    ],
    page: int = 1,
    page_size: int = 20,
) -> backend.server.v2.store.model.StoreSubmissionsResponse:
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
        raise


@router.post(
    "/submissions",
    tags=["store", "private"],
    dependencies=[fastapi.Depends(autogpt_libs.auth.middleware.auth_middleware)],
)
async def create_submission(
    submission_request: backend.server.v2.store.model.StoreSubmissionRequest,
    user_id: typing.Annotated[
        str, fastapi.Depends(autogpt_libs.auth.depends.get_user_id)
    ],
) -> backend.server.v2.store.model.StoreSubmission:
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
        submission = await backend.server.v2.store.db.create_store_submission(
            user_id=user_id,
            agent_id=submission_request.agent_id,
            agent_version=submission_request.agent_version,
            slug=submission_request.slug,
            name=submission_request.name,
            video_url=submission_request.video_url,
            image_urls=submission_request.image_urls,
            description=submission_request.description,
            sub_heading=submission_request.sub_heading,
            categories=submission_request.categories,
        )
        return submission
    except Exception:
        logger.exception("Exception occurred whilst creating store submission")
        raise


@router.post(
    "/submissions/media",
    tags=["store", "private"],
    dependencies=[fastapi.Depends(autogpt_libs.auth.middleware.auth_middleware)],
)
async def upload_submission_media(
    file: fastapi.UploadFile,
    user_id: typing.Annotated[
        str, fastapi.Depends(autogpt_libs.auth.depends.get_user_id)
    ],
) -> str:
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
    except Exception:
        logger.exception("Exception occurred whilst uploading submission media")
        raise
