import logging
import tempfile
import typing

import autogpt_libs.auth
import fastapi
import fastapi.responses
import prisma.enums

import backend.api.features.store.cache as store_cache
import backend.api.features.store.db as store_db
import backend.api.features.store.model as store_model
import backend.util.json

logger = logging.getLogger(__name__)

router = fastapi.APIRouter(
    prefix="/admin",
    tags=["store", "admin"],
    dependencies=[fastapi.Security(autogpt_libs.auth.requires_admin_user)],
)


@router.get(
    "/listings",
    summary="Get Admin Listings History",
    response_model=store_model.StoreListingsWithVersionsResponse,
)
async def get_admin_listings_with_versions(
    status: typing.Optional[prisma.enums.SubmissionStatus] = None,
    search: typing.Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
):
    """
    Get store listings with their version history for admins.

    This provides a consolidated view of listings with their versions,
    allowing for an expandable UI in the admin dashboard.

    Args:
        status: Filter by submission status (PENDING, APPROVED, REJECTED)
        search: Search by name, description, or user email
        page: Page number for pagination
        page_size: Number of items per page

    Returns:
        StoreListingsWithVersionsResponse with listings and their versions
    """
    try:
        listings = await store_db.get_admin_listings_with_versions(
            status=status,
            search_query=search,
            page=page,
            page_size=page_size,
        )
        return listings
    except Exception as e:
        logger.exception("Error getting admin listings with versions: %s", e)
        return fastapi.responses.JSONResponse(
            status_code=500,
            content={
                "detail": "An error occurred while retrieving listings with versions"
            },
        )


@router.post(
    "/submissions/{store_listing_version_id}/review",
    summary="Review Store Submission",
    response_model=store_model.StoreSubmission,
)
async def review_submission(
    store_listing_version_id: str,
    request: store_model.ReviewSubmissionRequest,
    user_id: str = fastapi.Security(autogpt_libs.auth.get_user_id),
):
    """
    Review a store listing submission.

    Args:
        store_listing_version_id: ID of the submission to review
        request: Review details including approval status and comments
        user_id: Authenticated admin user performing the review

    Returns:
        StoreSubmission with updated review information
    """
    try:
        already_approved = await store_db.check_submission_already_approved(
            store_listing_version_id=store_listing_version_id,
        )
        submission = await store_db.review_store_submission(
            store_listing_version_id=store_listing_version_id,
            is_approved=request.is_approved,
            external_comments=request.comments,
            internal_comments=request.internal_comments or "",
            reviewer_id=user_id,
        )

        state_changed = already_approved != request.is_approved
        # Clear caches when the request is approved as it updates what is shown on the store
        if state_changed:
            store_cache.clear_all_caches()
        return submission
    except Exception as e:
        logger.exception("Error reviewing submission: %s", e)
        return fastapi.responses.JSONResponse(
            status_code=500,
            content={"detail": "An error occurred while reviewing the submission"},
        )


@router.get(
    "/submissions/download/{store_listing_version_id}",
    summary="Admin Download Agent File",
    tags=["store", "admin"],
)
async def admin_download_agent_file(
    user_id: str = fastapi.Security(autogpt_libs.auth.get_user_id),
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
    graph_data = await store_db.get_agent_as_admin(
        user_id=user_id,
        store_listing_version_id=store_listing_version_id,
    )
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
