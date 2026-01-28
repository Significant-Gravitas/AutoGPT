import logging

import autogpt_libs.auth
import fastapi
import fastapi.responses

import backend.api.features.store.db as store_db
import backend.api.features.store.model as store_model

logger = logging.getLogger(__name__)

router = fastapi.APIRouter(
    prefix="/admin/waitlist",
    tags=["store", "admin", "waitlist"],
    dependencies=[fastapi.Security(autogpt_libs.auth.requires_admin_user)],
)


@router.post(
    "",
    summary="Create Waitlist",
    response_model=store_model.WaitlistAdminResponse,
)
async def create_waitlist(
    request: store_model.WaitlistCreateRequest,
    user_id: str = fastapi.Security(autogpt_libs.auth.get_user_id),
):
    """
    Create a new waitlist (admin only).

    Args:
        request: Waitlist creation details
        user_id: Authenticated admin user creating the waitlist

    Returns:
        WaitlistAdminResponse with the created waitlist details
    """
    try:
        waitlist = await store_db.create_waitlist_admin(
            admin_user_id=user_id,
            data=request,
        )
        return waitlist
    except Exception as e:
        logger.exception("Error creating waitlist: %s", e)
        return fastapi.responses.JSONResponse(
            status_code=500,
            content={"detail": "An error occurred while creating the waitlist"},
        )


@router.get(
    "",
    summary="List All Waitlists",
    response_model=store_model.WaitlistAdminListResponse,
)
async def list_waitlists():
    """
    Get all waitlists with admin details (admin only).

    Returns:
        WaitlistAdminListResponse with all waitlists
    """
    try:
        return await store_db.get_waitlists_admin()
    except Exception as e:
        logger.exception("Error listing waitlists: %s", e)
        return fastapi.responses.JSONResponse(
            status_code=500,
            content={"detail": "An error occurred while fetching waitlists"},
        )


@router.get(
    "/{waitlist_id}",
    summary="Get Waitlist Details",
    response_model=store_model.WaitlistAdminResponse,
)
async def get_waitlist(
    waitlist_id: str = fastapi.Path(..., description="The ID of the waitlist"),
):
    """
    Get a single waitlist with admin details (admin only).

    Args:
        waitlist_id: ID of the waitlist to retrieve

    Returns:
        WaitlistAdminResponse with waitlist details
    """
    try:
        return await store_db.get_waitlist_admin(waitlist_id)
    except ValueError:
        logger.warning("Waitlist not found: %s", waitlist_id)
        return fastapi.responses.JSONResponse(
            status_code=404,
            content={"detail": "Waitlist not found"},
        )
    except Exception as e:
        logger.exception("Error fetching waitlist: %s", e)
        return fastapi.responses.JSONResponse(
            status_code=500,
            content={"detail": "An error occurred while fetching the waitlist"},
        )


@router.put(
    "/{waitlist_id}",
    summary="Update Waitlist",
    response_model=store_model.WaitlistAdminResponse,
)
async def update_waitlist(
    request: store_model.WaitlistUpdateRequest,
    waitlist_id: str = fastapi.Path(..., description="The ID of the waitlist"),
):
    """
    Update a waitlist (admin only).

    Args:
        waitlist_id: ID of the waitlist to update
        request: Fields to update

    Returns:
        WaitlistAdminResponse with updated waitlist details
    """
    try:
        return await store_db.update_waitlist_admin(waitlist_id, request)
    except ValueError:
        logger.warning("Waitlist not found for update: %s", waitlist_id)
        return fastapi.responses.JSONResponse(
            status_code=404,
            content={"detail": "Waitlist not found"},
        )
    except Exception as e:
        logger.exception("Error updating waitlist: %s", e)
        return fastapi.responses.JSONResponse(
            status_code=500,
            content={"detail": "An error occurred while updating the waitlist"},
        )


@router.delete(
    "/{waitlist_id}",
    summary="Delete Waitlist",
)
async def delete_waitlist(
    waitlist_id: str = fastapi.Path(..., description="The ID of the waitlist"),
):
    """
    Soft delete a waitlist (admin only).

    Args:
        waitlist_id: ID of the waitlist to delete

    Returns:
        Success message
    """
    try:
        await store_db.delete_waitlist_admin(waitlist_id)
        return {"message": "Waitlist deleted successfully"}
    except ValueError:
        logger.warning(f"Waitlist not found for deletion: {waitlist_id}")
        return fastapi.responses.JSONResponse(
            status_code=404,
            content={"detail": "Waitlist not found"},
        )
    except Exception as e:
        logger.exception("Error deleting waitlist: %s", e)
        return fastapi.responses.JSONResponse(
            status_code=500,
            content={"detail": "An error occurred while deleting the waitlist"},
        )


@router.get(
    "/{waitlist_id}/signups",
    summary="Get Waitlist Signups",
    response_model=store_model.WaitlistSignupListResponse,
)
async def get_waitlist_signups(
    waitlist_id: str = fastapi.Path(..., description="The ID of the waitlist"),
):
    """
    Get all signups for a waitlist (admin only).

    Args:
        waitlist_id: ID of the waitlist

    Returns:
        WaitlistSignupListResponse with all signups
    """
    try:
        return await store_db.get_waitlist_signups_admin(waitlist_id)
    except ValueError:
        logger.warning("Waitlist not found for signups: %s", waitlist_id)
        return fastapi.responses.JSONResponse(
            status_code=404,
            content={"detail": "Waitlist not found"},
        )
    except Exception as e:
        logger.exception("Error fetching waitlist signups: %s", e)
        return fastapi.responses.JSONResponse(
            status_code=500,
            content={"detail": "An error occurred while fetching waitlist signups"},
        )


@router.post(
    "/{waitlist_id}/link",
    summary="Link Waitlist to Store Listing",
    response_model=store_model.WaitlistAdminResponse,
)
async def link_waitlist_to_listing(
    waitlist_id: str = fastapi.Path(..., description="The ID of the waitlist"),
    store_listing_id: str = fastapi.Body(
        ..., embed=True, description="The ID of the store listing"
    ),
):
    """
    Link a waitlist to a store listing (admin only).

    When the linked store listing is approved/published, waitlist users
    will be automatically notified.

    Args:
        waitlist_id: ID of the waitlist
        store_listing_id: ID of the store listing to link

    Returns:
        WaitlistAdminResponse with updated waitlist details
    """
    try:
        return await store_db.link_waitlist_to_listing_admin(
            waitlist_id, store_listing_id
        )
    except ValueError:
        logger.warning(
            "Link failed - waitlist or listing not found: %s, %s",
            waitlist_id,
            store_listing_id,
        )
        return fastapi.responses.JSONResponse(
            status_code=404,
            content={"detail": "Waitlist or store listing not found"},
        )
    except Exception as e:
        logger.exception("Error linking waitlist to listing: %s", e)
        return fastapi.responses.JSONResponse(
            status_code=500,
            content={"detail": "An error occurred while linking the waitlist"},
        )
