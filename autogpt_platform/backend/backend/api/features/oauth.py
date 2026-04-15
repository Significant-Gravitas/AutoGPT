"""
OAuth 2.0 Provider Endpoints

Implements OAuth 2.0 Authorization Code flow with PKCE support.

Flow:
1. User clicks "Login with AutoGPT" in 3rd party app
2. App redirects user to /auth/authorize with client_id, redirect_uri, scope, state
3. User sees consent screen (if not already logged in, redirects to login first)
4. User approves → backend creates authorization code
5. User redirected back to app with code
6. App exchanges code for access/refresh tokens at /api/oauth/token
7. App uses access token to call external API endpoints
"""

import io
import logging
import os
import uuid
from datetime import datetime
from typing import Literal, Optional
from urllib.parse import urlencode

from autogpt_libs.auth import get_user_id
from fastapi import APIRouter, Body, HTTPException, Security, UploadFile, status
from gcloud.aio import storage as async_storage
from PIL import Image
from prisma.enums import APIKeyPermission
from pydantic import BaseModel, Field

from backend.data.auth.oauth import (
    InvalidClientError,
    InvalidGrantError,
    OAuthApplicationInfo,
    TokenIntrospectionResult,
    consume_authorization_code,
    create_access_token,
    create_authorization_code,
    create_refresh_token,
    get_oauth_application,
    get_oauth_application_by_id,
    introspect_token,
    list_user_oauth_applications,
    refresh_tokens,
    revoke_access_token,
    revoke_refresh_token,
    update_oauth_application,
    validate_client_credentials,
    validate_redirect_uri,
    validate_scopes,
)
from backend.util.settings import Settings
from backend.util.virus_scanner import scan_content_safe

settings = Settings()
logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# Request/Response Models
# ============================================================================


class TokenResponse(BaseModel):
    """OAuth 2.0 token response"""

    token_type: Literal["Bearer"] = "Bearer"
    access_token: str
    access_token_expires_at: datetime
    refresh_token: str
    refresh_token_expires_at: datetime
    scopes: list[str]


class ErrorResponse(BaseModel):
    """OAuth 2.0 error response"""

    error: str
    error_description: Optional[str] = None


class OAuthApplicationPublicInfo(BaseModel):
    """Public information about an OAuth application (for consent screen)"""

    name: str
    description: Optional[str] = None
    logo_url: Optional[str] = None
    scopes: list[str]


# ============================================================================
# Application Info Endpoint
# ============================================================================


@router.get(
    "/app/{client_id}",
    responses={
        404: {"description": "Application not found or disabled"},
    },
)
async def get_oauth_app_info(
    client_id: str, user_id: str = Security(get_user_id)
) -> OAuthApplicationPublicInfo:
    """
    Get public information about an OAuth application.

    This endpoint is used by the consent screen to display application details
    to the user before they authorize access.

    Returns:
    - name: Application name
    - description: Application description (if provided)
    - scopes: List of scopes the application is allowed to request
    """
    app = await get_oauth_application(client_id)
    if not app or not app.is_active:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Application not found",
        )

    return OAuthApplicationPublicInfo(
        name=app.name,
        description=app.description,
        logo_url=app.logo_url,
        scopes=[s.value for s in app.scopes],
    )


# ============================================================================
# Authorization Endpoint
# ============================================================================


class AuthorizeRequest(BaseModel):
    """OAuth 2.0 authorization request"""

    client_id: str = Field(description="Client identifier")
    redirect_uri: str = Field(description="Redirect URI")
    scopes: list[str] = Field(description="List of scopes")
    state: str = Field(description="Anti-CSRF token from client")
    response_type: str = Field(
        default="code", description="Must be 'code' for authorization code flow"
    )
    code_challenge: str = Field(description="PKCE code challenge (required)")
    code_challenge_method: Literal["S256", "plain"] = Field(
        default="S256", description="PKCE code challenge method (S256 recommended)"
    )


class AuthorizeResponse(BaseModel):
    """OAuth 2.0 authorization response with redirect URL"""

    redirect_url: str = Field(description="URL to redirect the user to")


@router.post("/authorize")
async def authorize(
    request: AuthorizeRequest = Body(),
    user_id: str = Security(get_user_id),
) -> AuthorizeResponse:
    """
    OAuth 2.0 Authorization Endpoint

    User must be logged in (authenticated with Supabase JWT).
    This endpoint creates an authorization code and returns a redirect URL.

    PKCE (Proof Key for Code Exchange) is REQUIRED for all authorization requests.

    The frontend consent screen should call this endpoint after the user approves,
    then redirect the user to the returned `redirect_url`.

    Request Body:
    - client_id: The OAuth application's client ID
    - redirect_uri: Where to redirect after authorization (must match registered URI)
    - scopes: List of permissions (e.g., "EXECUTE_GRAPH READ_GRAPH")
    - state: Anti-CSRF token provided by client (will be returned in redirect)
    - response_type: Must be "code" (for authorization code flow)
    - code_challenge: PKCE code challenge (required)
    - code_challenge_method: "S256" (recommended) or "plain"

    Returns:
    - redirect_url: The URL to redirect the user to (includes authorization code)

    Error cases return a redirect_url with error parameters, or raise HTTPException
    for critical errors (like invalid redirect_uri).
    """
    try:
        # Validate response_type
        if request.response_type != "code":
            return _error_redirect_url(
                request.redirect_uri,
                request.state,
                "unsupported_response_type",
                "Only 'code' response type is supported",
            )

        # Get application
        app = await get_oauth_application(request.client_id)
        if not app:
            return _error_redirect_url(
                request.redirect_uri,
                request.state,
                "invalid_client",
                "Unknown client_id",
            )

        if not app.is_active:
            return _error_redirect_url(
                request.redirect_uri,
                request.state,
                "invalid_client",
                "Application is not active",
            )

        # Validate redirect URI
        if not validate_redirect_uri(app, request.redirect_uri):
            # For invalid redirect_uri, we can't redirect safely
            # Must return error instead
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Invalid redirect_uri. "
                    f"Must be one of: {', '.join(app.redirect_uris)}"
                ),
            )

        # Parse and validate scopes
        try:
            requested_scopes = [APIKeyPermission(s.strip()) for s in request.scopes]
        except ValueError as e:
            return _error_redirect_url(
                request.redirect_uri,
                request.state,
                "invalid_scope",
                f"Invalid scope: {e}",
            )

        if not requested_scopes:
            return _error_redirect_url(
                request.redirect_uri,
                request.state,
                "invalid_scope",
                "At least one scope is required",
            )

        if not validate_scopes(app, requested_scopes):
            return _error_redirect_url(
                request.redirect_uri,
                request.state,
                "invalid_scope",
                "Application is not authorized for all requested scopes. "
                f"Allowed: {', '.join(s.value for s in app.scopes)}",
            )

        # Create authorization code
        auth_code = await create_authorization_code(
            application_id=app.id,
            user_id=user_id,
            scopes=requested_scopes,
            redirect_uri=request.redirect_uri,
            code_challenge=request.code_challenge,
            code_challenge_method=request.code_challenge_method,
        )

        # Build redirect URL with authorization code
        params = {
            "code": auth_code.code,
            "state": request.state,
        }
        redirect_url = f"{request.redirect_uri}?{urlencode(params)}"

        logger.info(
            f"Authorization code issued for user #{user_id} "
            f"and app {app.name} (#{app.id})"
        )

        return AuthorizeResponse(redirect_url=redirect_url)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in authorization endpoint: {e}", exc_info=True)
        return _error_redirect_url(
            request.redirect_uri,
            request.state,
            "server_error",
            "An unexpected error occurred",
        )


def _error_redirect_url(
    redirect_uri: str,
    state: str,
    error: str,
    error_description: Optional[str] = None,
) -> AuthorizeResponse:
    """Helper to build redirect URL with OAuth error parameters"""
    params = {
        "error": error,
        "state": state,
    }
    if error_description:
        params["error_description"] = error_description

    redirect_url = f"{redirect_uri}?{urlencode(params)}"
    return AuthorizeResponse(redirect_url=redirect_url)


# ============================================================================
# Token Endpoint
# ============================================================================


class TokenRequestByCode(BaseModel):
    grant_type: Literal["authorization_code"]
    code: str = Field(description="Authorization code")
    redirect_uri: str = Field(
        description="Redirect URI (must match authorization request)"
    )
    client_id: str
    client_secret: str
    code_verifier: str = Field(description="PKCE code verifier")


class TokenRequestByRefreshToken(BaseModel):
    grant_type: Literal["refresh_token"]
    refresh_token: str
    client_id: str
    client_secret: str


@router.post("/token")
async def token(
    request: TokenRequestByCode | TokenRequestByRefreshToken = Body(),
) -> TokenResponse:
    """
    OAuth 2.0 Token Endpoint

    Exchanges authorization code or refresh token for access token.

    Grant Types:
    1. authorization_code: Exchange authorization code for tokens
       - Required: grant_type, code, redirect_uri, client_id, client_secret
       - Optional: code_verifier (required if PKCE was used)

    2. refresh_token: Exchange refresh token for new access token
       - Required: grant_type, refresh_token, client_id, client_secret

    Returns:
    - access_token: Bearer token for API access (1 hour TTL)
    - token_type: "Bearer"
    - expires_in: Seconds until access token expires
    - refresh_token: Token for refreshing access (30 days TTL)
    - scopes: List of scopes
    """
    # Validate client credentials
    try:
        app = await validate_client_credentials(
            request.client_id, request.client_secret
        )
    except InvalidClientError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
        )

    # Handle authorization_code grant
    if request.grant_type == "authorization_code":
        # Consume authorization code
        try:
            user_id, scopes = await consume_authorization_code(
                code=request.code,
                application_id=app.id,
                redirect_uri=request.redirect_uri,
                code_verifier=request.code_verifier,
            )
        except InvalidGrantError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )

        # Create access and refresh tokens
        access_token = await create_access_token(app.id, user_id, scopes)
        refresh_token = await create_refresh_token(app.id, user_id, scopes)

        logger.info(
            f"Access token issued for user #{user_id} and app {app.name} (#{app.id})"
            "via authorization code"
        )

        if not access_token.token or not refresh_token.token:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate tokens",
            )

        return TokenResponse(
            token_type="Bearer",
            access_token=access_token.token.get_secret_value(),
            access_token_expires_at=access_token.expires_at,
            refresh_token=refresh_token.token.get_secret_value(),
            refresh_token_expires_at=refresh_token.expires_at,
            scopes=list(s.value for s in scopes),
        )

    # Handle refresh_token grant
    elif request.grant_type == "refresh_token":
        # Refresh access token
        try:
            new_access_token, new_refresh_token = await refresh_tokens(
                request.refresh_token, app.id
            )
        except InvalidGrantError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )

        logger.info(
            f"Tokens refreshed for user #{new_access_token.user_id} "
            f"by app {app.name} (#{app.id})"
        )

        if not new_access_token.token or not new_refresh_token.token:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate tokens",
            )

        return TokenResponse(
            token_type="Bearer",
            access_token=new_access_token.token.get_secret_value(),
            access_token_expires_at=new_access_token.expires_at,
            refresh_token=new_refresh_token.token.get_secret_value(),
            refresh_token_expires_at=new_refresh_token.expires_at,
            scopes=list(s.value for s in new_access_token.scopes),
        )

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported grant_type: {request.grant_type}. "
            "Must be 'authorization_code' or 'refresh_token'",
        )


# ============================================================================
# Token Introspection Endpoint
# ============================================================================


@router.post("/introspect")
async def introspect(
    token: str = Body(description="Token to introspect"),
    token_type_hint: Optional[Literal["access_token", "refresh_token"]] = Body(
        None, description="Hint about token type ('access_token' or 'refresh_token')"
    ),
    client_id: str = Body(description="Client identifier"),
    client_secret: str = Body(description="Client secret"),
) -> TokenIntrospectionResult:
    """
    OAuth 2.0 Token Introspection Endpoint (RFC 7662)

    Allows clients to check if a token is valid and get its metadata.

    Returns:
    - active: Whether the token is currently active
    - scopes: List of authorized scopes (if active)
    - client_id: The client the token was issued to (if active)
    - user_id: The user the token represents (if active)
    - exp: Expiration timestamp (if active)
    - token_type: "access_token" or "refresh_token" (if active)
    """
    # Validate client credentials
    try:
        await validate_client_credentials(client_id, client_secret)
    except InvalidClientError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
        )

    # Introspect the token
    return await introspect_token(token, token_type_hint)


# ============================================================================
# Token Revocation Endpoint
# ============================================================================


@router.post("/revoke")
async def revoke(
    token: str = Body(description="Token to revoke"),
    token_type_hint: Optional[Literal["access_token", "refresh_token"]] = Body(
        None, description="Hint about token type ('access_token' or 'refresh_token')"
    ),
    client_id: str = Body(description="Client identifier"),
    client_secret: str = Body(description="Client secret"),
):
    """
    OAuth 2.0 Token Revocation Endpoint (RFC 7009)

    Allows clients to revoke an access or refresh token.

    Note: Revoking a refresh token does NOT revoke associated access tokens.
    Revoking an access token does NOT revoke the associated refresh token.
    """
    # Validate client credentials
    try:
        app = await validate_client_credentials(client_id, client_secret)
    except InvalidClientError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
        )

    # Try to revoke as access token first
    # Note: We pass app.id to ensure the token belongs to the authenticated app
    if token_type_hint != "refresh_token":
        revoked = await revoke_access_token(token, app.id)
        if revoked:
            logger.info(
                f"Access token revoked for app {app.name} (#{app.id}); "
                f"user #{revoked.user_id}"
            )
            return {"status": "ok"}

    # Try to revoke as refresh token
    revoked = await revoke_refresh_token(token, app.id)
    if revoked:
        logger.info(
            f"Refresh token revoked for app {app.name} (#{app.id}); "
            f"user #{revoked.user_id}"
        )
        return {"status": "ok"}

    # Per RFC 7009, revocation endpoint returns 200 even if token not found
    # or if token belongs to a different application.
    # This prevents token scanning attacks.
    logger.warning(f"Unsuccessful token revocation attempt by app {app.name} #{app.id}")
    return {"status": "ok"}


# ============================================================================
# Application Management Endpoints (for app owners)
# ============================================================================


@router.get("/apps/mine")
async def list_my_oauth_apps(
    user_id: str = Security(get_user_id),
) -> list[OAuthApplicationInfo]:
    """
    List all OAuth applications owned by the current user.

    Returns a list of OAuth applications with their details including:
    - id, name, description, logo_url
    - client_id (public identifier)
    - redirect_uris, grant_types, scopes
    - is_active status
    - created_at, updated_at timestamps

    Note: client_secret is never returned for security reasons.
    """
    return await list_user_oauth_applications(user_id)


@router.patch("/apps/{app_id}/status")
async def update_app_status(
    app_id: str,
    user_id: str = Security(get_user_id),
    is_active: bool = Body(description="Whether the app should be active", embed=True),
) -> OAuthApplicationInfo:
    """
    Enable or disable an OAuth application.

    Only the application owner can update the status.
    When disabled, the application cannot be used for new authorizations
    and existing access tokens will fail validation.

    Returns the updated application info.
    """
    updated_app = await update_oauth_application(
        app_id=app_id,
        owner_id=user_id,
        is_active=is_active,
    )

    if not updated_app:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Application not found or you don't have permission to update it",
        )

    action = "enabled" if is_active else "disabled"
    logger.info(f"OAuth app {updated_app.name} (#{app_id}) {action} by user #{user_id}")

    return updated_app


class UpdateAppLogoRequest(BaseModel):
    logo_url: str = Field(description="URL of the uploaded logo image")


@router.patch("/apps/{app_id}/logo")
async def update_app_logo(
    app_id: str,
    request: UpdateAppLogoRequest = Body(),
    user_id: str = Security(get_user_id),
) -> OAuthApplicationInfo:
    """
    Update the logo URL for an OAuth application.

    Only the application owner can update the logo.
    The logo should be uploaded first using the media upload endpoint,
    then this endpoint is called with the resulting URL.

    Logo requirements:
    - Must be square (1:1 aspect ratio)
    - Minimum 512x512 pixels
    - Maximum 2048x2048 pixels

    Returns the updated application info.
    """
    if (
        not (app := await get_oauth_application_by_id(app_id))
        or app.owner_id != user_id
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="OAuth App not found",
        )

    # Delete the current app logo file (if any and it's in our cloud storage)
    await _delete_app_current_logo_file(app)

    updated_app = await update_oauth_application(
        app_id=app_id,
        owner_id=user_id,
        logo_url=request.logo_url,
    )

    if not updated_app:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Application not found or you don't have permission to update it",
        )

    logger.info(
        f"OAuth app {updated_app.name} (#{app_id}) logo updated by user #{user_id}"
    )

    return updated_app


# Logo upload constraints
LOGO_MIN_SIZE = 512
LOGO_MAX_SIZE = 2048
LOGO_ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
LOGO_MAX_FILE_SIZE = 3 * 1024 * 1024  # 3MB


@router.post("/apps/{app_id}/logo/upload")
async def upload_app_logo(
    app_id: str,
    file: UploadFile,
    user_id: str = Security(get_user_id),
) -> OAuthApplicationInfo:
    """
    Upload a logo image for an OAuth application.

    Requirements:
    - Image must be square (1:1 aspect ratio)
    - Minimum 512x512 pixels
    - Maximum 2048x2048 pixels
    - Allowed formats: JPEG, PNG, WebP
    - Maximum file size: 3MB

    The image is uploaded to cloud storage and the app's logoUrl is updated.
    Returns the updated application info.
    """
    # Verify ownership to reduce vulnerability to DoS(torage) or DoM(oney) attacks
    if (
        not (app := await get_oauth_application_by_id(app_id))
        or app.owner_id != user_id
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="OAuth App not found",
        )

    # Check GCS configuration
    if not settings.config.media_gcs_bucket_name:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Media storage is not configured",
        )

    # Validate content type
    content_type = file.content_type
    if content_type not in LOGO_ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed: JPEG, PNG, WebP. Got: {content_type}",
        )

    # Read file content
    try:
        file_bytes = await file.read()
    except Exception as e:
        logger.error(f"Error reading logo file: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read uploaded file",
        )

    # Check file size
    if len(file_bytes) > LOGO_MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "File too large. "
                f"Maximum size is {LOGO_MAX_FILE_SIZE // 1024 // 1024}MB"
            ),
        )

    # Validate image dimensions
    try:
        image = Image.open(io.BytesIO(file_bytes))
        width, height = image.size

        if width != height:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Logo must be square. Got {width}x{height}",
            )

        if width < LOGO_MIN_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Logo too small. Minimum {LOGO_MIN_SIZE}x{LOGO_MIN_SIZE}. "
                f"Got {width}x{height}",
            )

        if width > LOGO_MAX_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Logo too large. Maximum {LOGO_MAX_SIZE}x{LOGO_MAX_SIZE}. "
                f"Got {width}x{height}",
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating logo image: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image file",
        )

    # Scan for viruses
    filename = file.filename or "logo"
    await scan_content_safe(file_bytes, filename=filename)

    # Generate unique filename
    file_ext = os.path.splitext(filename)[1].lower() or ".png"
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    storage_path = f"oauth-apps/{app_id}/logo/{unique_filename}"

    # Upload to GCS
    try:
        async with async_storage.Storage() as async_client:
            bucket_name = settings.config.media_gcs_bucket_name

            await async_client.upload(
                bucket_name, storage_path, file_bytes, content_type=content_type
            )

            logo_url = f"https://storage.googleapis.com/{bucket_name}/{storage_path}"
    except Exception as e:
        logger.error(f"Error uploading logo to GCS: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload logo",
        )

    # Delete the current app logo file (if any and it's in our cloud storage)
    await _delete_app_current_logo_file(app)

    # Update the app with the new logo URL
    updated_app = await update_oauth_application(
        app_id=app_id,
        owner_id=user_id,
        logo_url=logo_url,
    )

    if not updated_app:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Application not found or you don't have permission to update it",
        )

    logger.info(
        f"OAuth app {updated_app.name} (#{app_id}) logo uploaded by user #{user_id}"
    )

    return updated_app


async def _delete_app_current_logo_file(app: OAuthApplicationInfo):
    """
    Delete the current logo file for the given app, if there is one in our cloud storage
    """
    bucket_name = settings.config.media_gcs_bucket_name
    storage_base_url = f"https://storage.googleapis.com/{bucket_name}/"

    if app.logo_url and app.logo_url.startswith(storage_base_url):
        # Parse blob path from URL: https://storage.googleapis.com/{bucket}/{path}
        old_path = app.logo_url.replace(storage_base_url, "")
        try:
            async with async_storage.Storage() as async_client:
                await async_client.delete(bucket_name, old_path)
            logger.info(f"Deleted old logo for OAuth app #{app.id}: {old_path}")
        except Exception as e:
            # Log but don't fail - the new logo was uploaded successfully
            logger.warning(
                f"Failed to delete old logo for OAuth app #{app.id}: {e}", exc_info=e
            )
