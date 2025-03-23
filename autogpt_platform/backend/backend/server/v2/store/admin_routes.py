import logging
import typing

import autogpt_libs.auth.depends
import fastapi
import fastapi.responses
import prisma.enums

import backend.server.v2.store.db
import backend.server.v2.store.exceptions
import backend.server.v2.store.model

logger = logging.getLogger(__name__)

router = fastapi.APIRouter(prefix="/admin", tags=["store", "admin"])


@router.get(
    "/submissions",
    response_model=backend.server.v2.store.model.StoreSubmissionsResponse,
    dependencies=[fastapi.Depends(autogpt_libs.auth.depends.requires_admin_user)],
)
async def get_submissions(
    status: typing.Optional[prisma.enums.SubmissionStatus] = None,
    search: typing.Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
):
    """
    Get all store submissions with filtering options.

    Admin only endpoint for managing agent submissions.

    Args:
        status: Filter by submission status (ALL, PENDING, APPROVED, REJECTED)
        search: Search by name, creator, or description
        page: Page number for pagination
        page_size: Number of items per page

    Returns:
        StoreSubmissionsResponse with filtered submissions
    """
    try:
        submissions = await backend.server.v2.store.db.get_admin_submissions(
            status=status,
            search_query=search,
            page=page,
            page_size=page_size,
        )
        return submissions
    except Exception as e:
        logger.exception("Error getting admin submissions: %s", e)
        return fastapi.responses.JSONResponse(
            status_code=500,
            content={"detail": "An error occurred while retrieving submissions"},
        )


@router.get(
    "/submissions/pending",
    response_model=backend.server.v2.store.model.StoreSubmissionsResponse,
    dependencies=[fastapi.Depends(autogpt_libs.auth.depends.requires_admin_user)],
)
async def get_pending_submissions(
    page: int = 1,
    page_size: int = 20,
):
    """
    Get pending submissions that need admin review.

    Convenience endpoint for the admin dashboard.

    Args:
        page: Page number for pagination
        page_size: Number of items per page

    Returns:
        StoreSubmissionsResponse with pending submissions
    """
    try:
        submissions = await backend.server.v2.store.db.get_pending_submissions(
            page=page,
            page_size=page_size,
        )
        return submissions
    except Exception as e:
        logger.exception("Error getting pending submissions: %s", e)
        return fastapi.responses.JSONResponse(
            status_code=500,
            content={
                "detail": "An error occurred while retrieving pending submissions"
            },
        )


@router.get(
    "/submissions/{store_listing_version_id}",
    response_model=backend.server.v2.store.model.StoreSubmission,
    dependencies=[fastapi.Depends(autogpt_libs.auth.depends.requires_admin_user)],
)
async def get_submission_details(
    store_listing_version_id: str,
):
    """
    Get detailed information about a specific submission.

    Args:
        store_listing_version_id: ID of the submission version

    Returns:
        StoreSubmission with full details including internal comments
    """
    try:
        submission = await backend.server.v2.store.db.get_submission_details(
            store_listing_version_id=store_listing_version_id,
        )
        return submission
    except backend.server.v2.store.exceptions.SubmissionNotFoundError:
        return fastapi.responses.JSONResponse(
            status_code=404,
            content={"detail": "Submission not found"},
        )
    except Exception as e:
        logger.exception("Error getting submission details: %s", e)
        return fastapi.responses.JSONResponse(
            status_code=500,
            content={"detail": "An error occurred while retrieving submission details"},
        )


@router.get(
    "/submissions/listing/{listing_id}/history",
    response_model=backend.server.v2.store.model.StoreSubmissionsResponse,
    dependencies=[fastapi.Depends(autogpt_libs.auth.depends.requires_admin_user)],
)
async def get_listing_history(
    listing_id: str,
    page: int = 1,
    page_size: int = 20,
):
    """
    Get all submissions for a specific listing.

    This shows the version history of a listing over time.

    Args:
        listing_id: The ID of the store listing
        page: Page number for pagination
        page_size: Number of items per page

    Returns:
        StoreSubmissionsResponse with all versions of a specific listing
    """
    try:
        submissions = await backend.server.v2.store.db.get_listing_submissions_history(
            listing_id=listing_id,
            page=page,
            page_size=page_size,
        )
        return submissions
    except Exception as e:
        logger.exception("Error getting listing history: %s", e)
        return fastapi.responses.JSONResponse(
            status_code=500,
            content={"detail": "An error occurred while retrieving listing history"},
        )


@router.post(
    "/submissions/{store_listing_version_id}/review",
    response_model=backend.server.v2.store.model.StoreSubmission,
    dependencies=[fastapi.Depends(autogpt_libs.auth.depends.requires_admin_user)],
)
async def review_submission(
    store_listing_version_id: str,
    request: backend.server.v2.store.model.ReviewSubmissionRequest,
    user: typing.Annotated[
        autogpt_libs.auth.models.User,
        fastapi.Depends(autogpt_libs.auth.depends.requires_admin_user),
    ],
):
    """
    Review a store listing submission.

    Args:
        store_listing_version_id: ID of the submission to review
        request: Review details including approval status and comments
        user: Authenticated admin user performing the review

    Returns:
        StoreSubmission with updated review information
    """
    try:
        submission = await backend.server.v2.store.db.review_store_submission(
            store_listing_version_id=store_listing_version_id,
            is_approved=request.is_approved,
            external_comments=request.comments,
            internal_comments=request.internal_comments or "",
            reviewer_id=user.user_id,
        )
        return submission
    except Exception as e:
        logger.exception("Error reviewing submission: %s", e)
        return fastapi.responses.JSONResponse(
            status_code=500,
            content={"detail": "An error occurred while reviewing the submission"},
        )
