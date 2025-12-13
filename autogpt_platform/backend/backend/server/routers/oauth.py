"""
OAuth 2.0 Provider Endpoints

Implements OAuth 2.0 Authorization Code flow with PKCE support.

Flow:
1. User clicks "Login with AutoGPT" in 3rd party app
2. App redirects user to /oauth/authorize with client_id, redirect_uri, scope, state
3. User sees consent screen (if not already logged in, redirects to login first)
4. User approves → backend creates authorization code
5. User redirected back to app with code
6. App exchanges code for access/refresh tokens at /oauth/token
7. App uses access token to call external API endpoints
"""

import logging
from datetime import datetime
from typing import Literal, Optional
from urllib.parse import urlencode

from autogpt_libs.auth import get_user_id
from fastapi import APIRouter, Body, HTTPException, Query, Security, status
from fastapi.responses import RedirectResponse
from prisma.enums import APIKeyPermission
from pydantic import BaseModel, Field

from backend.data.auth.oauth import (
    InvalidClientError,
    InvalidGrantError,
    TokenIntrospectionResult,
    consume_authorization_code,
    create_access_token,
    create_authorization_code,
    create_refresh_token,
    get_oauth_application,
    introspect_token,
    refresh_tokens,
    revoke_access_token,
    revoke_refresh_token,
    validate_client_credentials,
    validate_redirect_uri,
    validate_scopes,
)

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


# ============================================================================
# Authorization Endpoint
# ============================================================================


@router.get("/authorize")
async def authorize(
    client_id: str = Query(description="Client identifier"),
    redirect_uri: str = Query(description="Redirect URI"),
    scope: str = Query(description="Space-separated list of scopes"),
    state: str = Query(description="Anti-CSRF token from client"),
    response_type: str = Query(
        default="code", description="Must be 'code' for authorization code flow"
    ),
    code_challenge: str = Query(description="PKCE code challenge (required)"),
    code_challenge_method: Literal["S256", "plain"] = Query(
        default="S256", description="PKCE code challenge method (S256 recommended)"
    ),
    user_id: str = Security(get_user_id),
) -> RedirectResponse:
    """
    OAuth 2.0 Authorization Endpoint

    User must be logged in (authenticated with Supabase JWT).
    This endpoint creates an authorization code and redirects back to the client.

    PKCE (Proof Key for Code Exchange) is REQUIRED for all authorization requests.

    Query Parameters:
    - client_id: The OAuth application's client ID
    - redirect_uri: Where to redirect after authorization (must match registered URI)
    - scope: Space-separated list of permissions (e.g., "EXECUTE_GRAPH READ_GRAPH")
    - state: Anti-CSRF token provided by client (will be returned in redirect)
    - response_type: Must be "code" (for authorization code flow)
    - code_challenge: PKCE code challenge (required)
    - code_challenge_method: "S256" (recommended) or "plain"

    Returns:
    - Redirect to redirect_uri with authorization code and state

    Error cases redirect to redirect_uri with error parameters.
    """
    try:
        # Validate response_type
        if response_type != "code":
            return _redirect_error(
                redirect_uri,
                state,
                "unsupported_response_type",
                "Only 'code' response type is supported",
            )

        # Get application
        app = await get_oauth_application(client_id)
        if not app:
            return _redirect_error(
                redirect_uri,
                state,
                "invalid_client",
                "Unknown client_id",
            )

        if not app.is_active:
            return _redirect_error(
                redirect_uri,
                state,
                "invalid_client",
                "Application is not active",
            )

        # Validate redirect URI
        if not validate_redirect_uri(app, redirect_uri):
            # For invalid redirect_uri, we can't redirect safely
            # Must show error page instead
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Invalid redirect_uri. "
                    f"Must be one of: {', '.join(app.redirect_uris)}"
                ),
            )

        # Parse and validate scopes
        try:
            requested_scopes = [
                APIKeyPermission(s.strip()) for s in scope.split() if s.strip()
            ]
        except ValueError as e:
            return _redirect_error(
                redirect_uri,
                state,
                "invalid_scope",
                f"Invalid scope: {e}",
            )

        if not requested_scopes:
            return _redirect_error(
                redirect_uri,
                state,
                "invalid_scope",
                "At least one scope is required",
            )

        if not validate_scopes(app, requested_scopes):
            return _redirect_error(
                redirect_uri,
                state,
                "invalid_scope",
                "Application is not authorized for all requested scopes. "
                f"Allowed: {', '.join(s.value for s in app.scopes)}",
            )

        # Create authorization code
        auth_code = await create_authorization_code(
            application_id=app.id,
            user_id=user_id,
            scopes=requested_scopes,
            redirect_uri=redirect_uri,
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method,
        )

        # Redirect back to client with authorization code
        params = {
            "code": auth_code.code,
            "state": state,
        }
        redirect_url = f"{redirect_uri}?{urlencode(params)}"

        logger.info(
            f"Authorization code issued for user #{user_id} "
            f"and app {app.name} (#{client_id})"
        )

        return RedirectResponse(url=redirect_url, status_code=status.HTTP_302_FOUND)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in authorization endpoint: {e}", exc_info=True)
        return _redirect_error(
            redirect_uri,
            state,
            "server_error",
            "An unexpected error occurred",
        )


def _redirect_error(
    redirect_uri: str,
    state: str,
    error: str,
    error_description: Optional[str] = None,
) -> RedirectResponse:
    """Helper to redirect with OAuth error parameters"""
    params = {
        "error": error,
        "state": state,
    }
    if error_description:
        params["error_description"] = error_description

    redirect_url = f"{redirect_uri}?{urlencode(params)}"
    return RedirectResponse(url=redirect_url, status_code=status.HTTP_302_FOUND)


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
    - scope: Space-separated scopes
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
            f"Access token issued for user {user_id} and app {app.name} (#{app.id})"
            "via authorization_code"
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
            f"Access token refreshed for user #{new_access_token.user_id} "
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
    - scope: Space-separated scopes (if active)
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
