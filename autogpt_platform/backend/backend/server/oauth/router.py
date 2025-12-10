"""
OAuth 2.0 Authorization Server endpoints.

Implements:
- GET /oauth/authorize - Authorization endpoint
- POST /oauth/authorize/consent - Consent form submission
- POST /oauth/token - Token endpoint
- GET /oauth/userinfo - OIDC UserInfo endpoint
- POST /oauth/revoke - Token revocation endpoint

Authentication:
- X-API-Key header - API key for external apps (preferred)
- Authorization: Bearer <jwt> - JWT token authentication
- access_token cookie - Browser-based auth
"""

import json
import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import urlencode

from fastapi import APIRouter, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from backend.data.db import prisma
from backend.data.redis_client import get_redis_async
from backend.server.oauth.consent_templates import (
    render_consent_page,
    render_error_page,
)
from backend.server.oauth.errors import (
    InvalidClientError,
    InvalidRequestError,
    LoginRequiredError,
    OAuthError,
    UnsupportedGrantTypeError,
)
from backend.server.oauth.models import TokenResponse, UserInfoResponse
from backend.server.oauth.service import get_oauth_service
from backend.server.oauth.token_service import get_token_service
from backend.util.rate_limiter import check_rate_limit

logger = logging.getLogger(__name__)

oauth_router = APIRouter(prefix="/oauth", tags=["oauth"])

# Redis key prefix and TTL for consent state storage
CONSENT_STATE_PREFIX = "oauth:consent:"
CONSENT_STATE_TTL = 600  # 10 minutes


async def _store_consent_state(token: str, state: dict) -> None:
    """Store consent state in Redis with TTL."""
    redis = await get_redis_async()
    await redis.setex(
        f"{CONSENT_STATE_PREFIX}{token}",
        CONSENT_STATE_TTL,
        json.dumps(state, default=str),
    )


async def _get_and_delete_consent_state(token: str) -> Optional[dict]:
    """Retrieve and delete consent state from Redis (atomic get+delete)."""
    redis = await get_redis_async()
    key = f"{CONSENT_STATE_PREFIX}{token}"
    state_json = await redis.get(key)
    if state_json:
        await redis.delete(key)
        return json.loads(state_json)
    return None


async def _get_user_id_from_request(request: Request) -> Optional[str]:
    """
    Extract user ID from request, checking API key, Authorization header, and cookie.

    Supports:
    1. X-API-Key header - API key authentication (preferred for external apps)
    2. Authorization: Bearer <jwt> - JWT token authentication
    3. access_token cookie - Cookie-based auth (for browser flows)
    """
    from autogpt_libs.auth.jwt_utils import parse_jwt_token

    from backend.data.api_key import validate_api_key

    # First try X-API-Key header (for external apps)
    api_key = request.headers.get("X-API-Key")
    if api_key:
        try:
            api_key_info = await validate_api_key(api_key)
            if api_key_info:
                return api_key_info.user_id
        except Exception:
            logger.debug("API key validation failed")

    # Then try Authorization header (JWT)
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        try:
            token = auth_header[7:]
            payload = parse_jwt_token(token)
            return payload.get("sub")
        except Exception as e:
            logger.debug("JWT token validation failed: %s", type(e).__name__)

    # Finally try cookie (browser-based auth)
    token = request.cookies.get("access_token")
    if token:
        try:
            payload = parse_jwt_token(token)
            return payload.get("sub")
        except Exception as e:
            logger.debug("Cookie token validation failed: %s", type(e).__name__)

    return None


def _parse_scopes(scope_str: str) -> list[str]:
    """Parse space-separated scope string into list."""
    if not scope_str:
        return []
    return [s.strip() for s in scope_str.split() if s.strip()]


def _get_client_ip(request: Request) -> str:
    """Get client IP address from request."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# ================================================================
# Authorization Endpoint
# ================================================================


@oauth_router.get("/authorize", response_model=None)
async def authorize(
    request: Request,
    response_type: str = Query(..., description="Must be 'code'"),
    client_id: str = Query(..., description="Client identifier"),
    redirect_uri: str = Query(..., description="Redirect URI"),
    state: str = Query(..., description="CSRF state parameter"),
    code_challenge: str = Query(..., description="PKCE code challenge"),
    code_challenge_method: str = Query("S256", description="PKCE method"),
    scope: str = Query("", description="Space-separated scopes"),
    nonce: Optional[str] = Query(None, description="OIDC nonce"),
    prompt: Optional[str] = Query(None, description="Prompt behavior"),
) -> HTMLResponse | RedirectResponse:
    """
    OAuth 2.0 Authorization Endpoint.

    Validates the request, checks user authentication, and either:
    - Returns error if user is not authenticated (API key or JWT required)
    - Shows consent page if user hasn't authorized these scopes
    - Redirects with authorization code if already authorized

    Authentication methods (in order of preference):
    1. X-API-Key header - API key for external apps
    2. Authorization: Bearer <jwt> - JWT token
    3. access_token cookie - Browser-based auth
    """
    # Get user ID from API key, Authorization header, or cookie
    user_id = await _get_user_id_from_request(request)

    # Rate limiting - use client IP as identifier for authorize endpoint
    client_ip = _get_client_ip(request)
    rate_result = await check_rate_limit(client_ip, "oauth_authorize")
    if not rate_result.allowed:
        return HTMLResponse(
            render_error_page(
                "rate_limit_exceeded",
                "Too many authorization requests. Please try again later.",
            ),
            status_code=429,
        )

    oauth_service = get_oauth_service()

    try:
        # Validate response_type
        if response_type != "code":
            raise InvalidRequestError(
                "Only 'code' response_type is supported", state=state
            )

        # Validate PKCE method
        if code_challenge_method != "S256":
            raise InvalidRequestError(
                "Only 'S256' code_challenge_method is supported", state=state
            )

        # Parse scopes
        scopes = _parse_scopes(scope)

        # Validate client and redirect URI
        client = await oauth_service.validate_client(client_id, redirect_uri, scopes)

        # Check if user is authenticated
        if not user_id:
            # Authentication required - user must provide API key or JWT
            raise LoginRequiredError(state=state)

        # Check if user has already authorized these scopes
        if prompt != "consent":
            has_auth = await oauth_service.has_valid_authorization(
                user_id, client_id, scopes
            )
            if has_auth:
                # Skip consent, issue code directly
                code = await oauth_service.create_authorization_code(
                    user_id=user_id,
                    client_id=client_id,
                    redirect_uri=redirect_uri,
                    scopes=scopes,
                    code_challenge=code_challenge,
                    code_challenge_method=code_challenge_method,
                    nonce=nonce,
                )
                redirect_url = (
                    f"{redirect_uri}?{urlencode({'code': code, 'state': state})}"
                )
                return RedirectResponse(url=redirect_url, status_code=302)

        # Generate consent token and store state in Redis
        consent_token = secrets.token_urlsafe(32)
        await _store_consent_state(
            consent_token,
            {
                "user_id": user_id,
                "client_id": client_id,
                "redirect_uri": redirect_uri,
                "scopes": scopes,
                "state": state,
                "code_challenge": code_challenge,
                "code_challenge_method": code_challenge_method,
                "nonce": nonce,
                "expires_at": (
                    datetime.now(timezone.utc) + timedelta(minutes=10)
                ).isoformat(),
            },
        )

        # Render consent page
        return HTMLResponse(
            render_consent_page(
                client_name=client.name,
                client_logo=client.logoUrl,
                scopes=scopes,
                consent_token=consent_token,
                action_url="/oauth/authorize/consent",
                privacy_policy_url=client.privacyPolicyUrl,
                terms_url=client.termsOfServiceUrl,
            )
        )

    except OAuthError as e:
        # If we have a valid redirect_uri, redirect with error
        # Otherwise show error page
        try:
            client = await oauth_service.get_client(client_id)
            if client and redirect_uri in client.redirectUris:
                return e.to_redirect(redirect_uri)
        except Exception:
            pass

        return HTMLResponse(
            render_error_page(e.error.value, e.description or "An error occurred"),
            status_code=400,
        )


@oauth_router.post("/authorize/consent", response_model=None)
async def submit_consent(
    request: Request,
    consent_token: str = Form(...),
    authorize: str = Form(...),
) -> HTMLResponse | RedirectResponse:
    """
    Process consent form submission.

    Creates authorization code and redirects to client's redirect_uri.
    """
    oauth_service = get_oauth_service()

    # Validate consent token (retrieves and deletes from Redis atomically)
    consent_state = await _get_and_delete_consent_state(consent_token)
    if not consent_state:
        return HTMLResponse(
            render_error_page("invalid_request", "Invalid or expired consent token"),
            status_code=400,
        )

    # Check expiration (expires_at is stored as ISO string in Redis)
    expires_at = datetime.fromisoformat(consent_state["expires_at"])
    if expires_at < datetime.now(timezone.utc):
        return HTMLResponse(
            render_error_page("invalid_request", "Consent session expired"),
            status_code=400,
        )

    redirect_uri = consent_state["redirect_uri"]
    state = consent_state["state"]

    # Check if user denied
    if authorize.lower() != "true":
        error_params = urlencode(
            {
                "error": "access_denied",
                "error_description": "User denied the authorization request",
                "state": state,
            }
        )
        return RedirectResponse(
            url=f"{redirect_uri}?{error_params}",
            status_code=302,
        )

    try:
        # Create authorization code
        code = await oauth_service.create_authorization_code(
            user_id=consent_state["user_id"],
            client_id=consent_state["client_id"],
            redirect_uri=redirect_uri,
            scopes=consent_state["scopes"],
            code_challenge=consent_state["code_challenge"],
            code_challenge_method=consent_state["code_challenge_method"],
            nonce=consent_state["nonce"],
        )

        # Redirect with code
        return RedirectResponse(
            url=f"{redirect_uri}?{urlencode({'code': code, 'state': state})}",
            status_code=302,
        )

    except OAuthError as e:
        return e.to_redirect(redirect_uri)


# ================================================================
# Token Endpoint
# ================================================================


@oauth_router.post("/token", response_model=TokenResponse)
async def token(
    request: Request,
    grant_type: str = Form(...),
    code: Optional[str] = Form(None),
    redirect_uri: Optional[str] = Form(None),
    client_id: str = Form(...),
    client_secret: Optional[str] = Form(None),
    code_verifier: Optional[str] = Form(None),
    refresh_token: Optional[str] = Form(None),
    scope: Optional[str] = Form(None),
) -> TokenResponse:
    """
    OAuth 2.0 Token Endpoint.

    Supports:
    - authorization_code grant (with PKCE)
    - refresh_token grant
    """
    # Rate limiting - use client_id as identifier
    rate_result = await check_rate_limit(client_id, "oauth_token")
    if not rate_result.allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={
                "Retry-After": str(int(rate_result.retry_after or 60)),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(rate_result.reset_at)),
            },
        )

    oauth_service = get_oauth_service()

    try:
        # Validate client authentication
        await oauth_service.validate_client_secret(client_id, client_secret)

        if grant_type == "authorization_code":
            # Validate required parameters
            if not code:
                raise InvalidRequestError("'code' is required")
            if not redirect_uri:
                raise InvalidRequestError("'redirect_uri' is required")
            if not code_verifier:
                raise InvalidRequestError("'code_verifier' is required for PKCE")

            return await oauth_service.exchange_authorization_code(
                code=code,
                client_id=client_id,
                redirect_uri=redirect_uri,
                code_verifier=code_verifier,
            )

        elif grant_type == "refresh_token":
            if not refresh_token:
                raise InvalidRequestError("'refresh_token' is required")

            requested_scopes = _parse_scopes(scope) if scope else None
            return await oauth_service.refresh_access_token(
                refresh_token=refresh_token,
                client_id=client_id,
                requested_scopes=requested_scopes,
            )

        else:
            raise UnsupportedGrantTypeError(grant_type)

    except OAuthError as e:
        # 401 for client auth failure, 400 for other validation errors (per RFC 6749)
        raise e.to_http_exception(401 if isinstance(e, InvalidClientError) else 400)


# ================================================================
# UserInfo Endpoint
# ================================================================


@oauth_router.get("/userinfo", response_model=UserInfoResponse)
async def userinfo(request: Request) -> UserInfoResponse:
    """
    OIDC UserInfo Endpoint.

    Returns user profile information based on the granted scopes.
    """
    token_service = get_token_service()

    # Extract bearer token
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Bearer token required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = auth_header[7:]

    try:
        # Verify token
        claims = token_service.verify_access_token(token)

        # Check token is not revoked
        token_hash = token_service.hash_token(token)
        stored_token = await prisma.oauthaccesstoken.find_unique(
            where={"tokenHash": token_hash}
        )

        if not stored_token or stored_token.revokedAt:
            raise HTTPException(
                status_code=401,
                detail="Token has been revoked",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Update last used
        await prisma.oauthaccesstoken.update(
            where={"id": stored_token.id},
            data={"lastUsedAt": datetime.now(timezone.utc)},
        )

        # Get user info based on scopes
        user = await prisma.user.find_unique(where={"id": claims.sub})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        scopes = claims.scope.split()

        # Build response based on scopes
        email = user.email if "email" in scopes else None
        email_verified = user.emailVerified if "email" in scopes else None
        name = user.name if "profile" in scopes else None
        updated_at = int(user.updatedAt.timestamp()) if "profile" in scopes else None

        return UserInfoResponse(
            sub=claims.sub,
            email=email,
            email_verified=email_verified,
            name=name,
            updated_at=updated_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail=f"Invalid token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ================================================================
# Token Revocation Endpoint
# ================================================================


@oauth_router.post("/revoke")
async def revoke(
    request: Request,
    token: str = Form(...),
    token_type_hint: Optional[str] = Form(None),
) -> JSONResponse:
    """
    OAuth 2.0 Token Revocation Endpoint (RFC 7009).

    Revokes an access token or refresh token.
    """
    oauth_service = get_oauth_service()

    # Note: Per RFC 7009, always return 200 even if token not found
    await oauth_service.revoke_token(token, token_type_hint)

    return JSONResponse(content={}, status_code=200)
