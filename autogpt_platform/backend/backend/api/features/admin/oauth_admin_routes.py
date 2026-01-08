"""
OAuth Application Admin Routes

Provides admin-only endpoints for managing OAuth applications:
- List all OAuth applications
- Create new OAuth applications
- Update OAuth applications
- Delete OAuth applications
- Regenerate client secrets
"""

import logging
from typing import Optional

from autogpt_libs.auth import get_user_id, requires_admin_user
from fastapi import APIRouter, Body, HTTPException, Query, Security, status
from prisma.enums import APIKeyPermission
from pydantic import BaseModel, Field

from backend.data.auth.oauth import (
    OAuthApplicationCreationResult,
    OAuthApplicationInfo,
    admin_update_oauth_application,
    create_oauth_application,
    delete_oauth_application,
    get_oauth_application_by_id,
    list_all_oauth_applications,
    regenerate_client_secret,
)

logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/admin",
    tags=["oauth", "admin"],
    dependencies=[Security(requires_admin_user)],
)


# ============================================================================
# Request/Response Models
# ============================================================================


class CreateOAuthAppRequest(BaseModel):
    """Request to create a new OAuth application"""

    name: str = Field(description="Application name")
    description: Optional[str] = Field(None, description="Application description")
    redirect_uris: list[str] = Field(description="Allowed redirect URIs")
    scopes: list[str] = Field(
        description="List of scopes (e.g., EXECUTE_GRAPH, READ_GRAPH)"
    )
    grant_types: Optional[list[str]] = Field(
        None,
        description="Grant types (default: authorization_code, refresh_token)",
    )
    owner_id: str = Field(description="User ID who will own this application")


class UpdateOAuthAppRequest(BaseModel):
    """Request to update an OAuth application"""

    name: Optional[str] = Field(None, description="Application name")
    description: Optional[str] = Field(None, description="Application description")
    redirect_uris: Optional[list[str]] = Field(None, description="Allowed redirect URIs")
    scopes: Optional[list[str]] = Field(None, description="List of scopes")
    is_active: Optional[bool] = Field(None, description="Whether the app is active")


class OAuthAppsListResponse(BaseModel):
    """Response for listing OAuth applications"""

    applications: list[OAuthApplicationInfo]
    total: int
    page: int
    page_size: int
    total_pages: int


class RegenerateSecretResponse(BaseModel):
    """Response when regenerating a client secret"""

    client_secret: str = Field(
        description="New plaintext client secret - shown only once"
    )


# ============================================================================
# Admin Endpoints
# ============================================================================


@router.get(
    "/apps",
    response_model=OAuthAppsListResponse,
    summary="List All OAuth Applications",
)
async def list_oauth_apps(
    admin_user_id: str = Security(get_user_id),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    search: Optional[str] = Query(None, description="Search by name, client ID, or description"),
):
    """
    List all OAuth applications in the system.

    Admin-only endpoint. Returns paginated list of all OAuth applications
    with their details (excluding client secrets).
    """
    logger.info(f"Admin user {admin_user_id} is listing OAuth applications")

    applications, total = await list_all_oauth_applications(
        page=page,
        page_size=page_size,
        search=search,
    )

    total_pages = (total + page_size - 1) // page_size

    return OAuthAppsListResponse(
        applications=applications,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get(
    "/apps/{app_id}",
    response_model=OAuthApplicationInfo,
    summary="Get OAuth Application Details",
)
async def get_oauth_app(
    app_id: str,
    admin_user_id: str = Security(get_user_id),
):
    """
    Get details of a specific OAuth application.

    Admin-only endpoint. Returns application details (excluding client secret).
    """
    logger.info(f"Admin user {admin_user_id} is getting OAuth app {app_id}")

    app = await get_oauth_application_by_id(app_id)
    if not app:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="OAuth application not found",
        )

    return app


@router.post(
    "/apps",
    response_model=OAuthApplicationCreationResult,
    summary="Create OAuth Application",
    status_code=status.HTTP_201_CREATED,
)
async def create_oauth_app(
    request: CreateOAuthAppRequest = Body(),
    admin_user_id: str = Security(get_user_id),
):
    """
    Create a new OAuth application.

    Admin-only endpoint. Returns the created application including the
    plaintext client secret (which is only shown once).

    The client secret is hashed before storage and cannot be retrieved later.
    If lost, a new secret must be generated using the regenerate endpoint.
    """
    logger.info(
        f"Admin user {admin_user_id} is creating OAuth app '{request.name}' "
        f"for user {request.owner_id}"
    )

    # Validate scopes
    try:
        validated_scopes = [APIKeyPermission(s.strip()) for s in request.scopes if s.strip()]
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid scope: {e}",
        )

    if not validated_scopes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one scope is required",
        )

    # Validate redirect URIs
    if not request.redirect_uris:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one redirect URI is required",
        )

    result = await create_oauth_application(
        name=request.name,
        description=request.description,
        redirect_uris=request.redirect_uris,
        scopes=validated_scopes,
        owner_id=request.owner_id,
        grant_types=request.grant_types,
    )

    logger.info(
        f"Created OAuth app '{result.application.name}' "
        f"(client_id: {result.application.client_id})"
    )

    return result


@router.patch(
    "/apps/{app_id}",
    response_model=OAuthApplicationInfo,
    summary="Update OAuth Application",
)
async def update_oauth_app(
    app_id: str,
    request: UpdateOAuthAppRequest = Body(),
    admin_user_id: str = Security(get_user_id),
):
    """
    Update an OAuth application.

    Admin-only endpoint. Can update name, description, redirect URIs,
    scopes, and active status.
    """
    logger.info(f"Admin user {admin_user_id} is updating OAuth app {app_id}")

    # Validate scopes if provided
    validated_scopes = None
    if request.scopes is not None:
        try:
            validated_scopes = [
                APIKeyPermission(s.strip()) for s in request.scopes if s.strip()
            ]
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid scope: {e}",
            )

        if not validated_scopes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one scope is required",
            )

    updated_app = await admin_update_oauth_application(
        app_id=app_id,
        name=request.name,
        description=request.description,
        redirect_uris=request.redirect_uris,
        scopes=validated_scopes,
        is_active=request.is_active,
    )

    if not updated_app:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="OAuth application not found",
        )

    action = "updated"
    if request.is_active is not None:
        action = "enabled" if request.is_active else "disabled"
    logger.info(f"OAuth app {updated_app.name} (#{app_id}) {action}")

    return updated_app


@router.delete(
    "/apps/{app_id}",
    summary="Delete OAuth Application",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_oauth_app(
    app_id: str,
    admin_user_id: str = Security(get_user_id),
):
    """
    Delete an OAuth application.

    Admin-only endpoint. This will also delete all associated authorization
    codes, access tokens, and refresh tokens.

    This action is irreversible.
    """
    logger.info(f"Admin user {admin_user_id} is deleting OAuth app {app_id}")

    deleted = await delete_oauth_application(app_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="OAuth application not found",
        )

    logger.info(f"Deleted OAuth app {app_id}")
    return None


@router.post(
    "/apps/{app_id}/regenerate-secret",
    response_model=RegenerateSecretResponse,
    summary="Regenerate Client Secret",
)
async def regenerate_oauth_secret(
    app_id: str,
    admin_user_id: str = Security(get_user_id),
):
    """
    Regenerate the client secret for an OAuth application.

    Admin-only endpoint. The old secret will be invalidated immediately.
    Returns the new plaintext client secret (shown only once).

    All existing tokens will continue to work, but new token requests
    must use the new client secret.
    """
    logger.info(
        f"Admin user {admin_user_id} is regenerating secret for OAuth app {app_id}"
    )

    new_secret = await regenerate_client_secret(app_id)

    if not new_secret:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="OAuth application not found",
        )

    logger.info(f"Regenerated client secret for OAuth app {app_id}")

    return RegenerateSecretResponse(client_secret=new_secret)
