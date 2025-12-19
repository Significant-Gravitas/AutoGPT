"""
Authentication API routes.

Provides endpoints for:
- User registration and login
- Token refresh and logout
- Password reset
- Email verification
- Google OAuth
"""

import logging
import secrets
import time
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel, EmailStr, Field

from backend.util.settings import Settings

from .email import get_auth_email_sender
from .service import AuthService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

# Singleton auth service instance
_auth_service: Optional[AuthService] = None

# In-memory state storage for OAuth CSRF protection
# Format: {state_token: {"created_at": timestamp, "redirect_uri": optional_uri}}
# In production, use Redis for distributed state management
_oauth_states: dict[str, dict] = {}
_STATE_TTL_SECONDS = 600  # 10 minutes


def _cleanup_expired_states() -> None:
    """Remove expired OAuth states."""
    now = time.time()
    expired = [
        k
        for k, v in _oauth_states.items()
        if now - v["created_at"] > _STATE_TTL_SECONDS
    ]
    for k in expired:
        del _oauth_states[k]


def _generate_state() -> str:
    """Generate a cryptographically secure state token."""
    _cleanup_expired_states()
    state = secrets.token_urlsafe(32)
    _oauth_states[state] = {"created_at": time.time()}
    return state


def _validate_state(state: str) -> bool:
    """Validate and consume a state token."""
    if state not in _oauth_states:
        return False
    state_data = _oauth_states.pop(state)
    if time.time() - state_data["created_at"] > _STATE_TTL_SECONDS:
        return False
    return True


def get_auth_service() -> AuthService:
    """Get or create the auth service singleton."""
    global _auth_service
    if _auth_service is None:
        _auth_service = AuthService()
    return _auth_service


# ============= Request/Response Models =============


class RegisterRequest(BaseModel):
    """Request model for user registration."""

    email: EmailStr
    password: str = Field(..., min_length=8)
    name: Optional[str] = None


class LoginRequest(BaseModel):
    """Request model for user login."""

    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """Response model for authentication tokens."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class RefreshRequest(BaseModel):
    """Request model for token refresh."""

    refresh_token: str


class LogoutRequest(BaseModel):
    """Request model for logout."""

    refresh_token: str


class PasswordResetRequest(BaseModel):
    """Request model for password reset request."""

    email: EmailStr


class PasswordResetConfirm(BaseModel):
    """Request model for password reset confirmation."""

    token: str
    new_password: str = Field(..., min_length=8)


class MessageResponse(BaseModel):
    """Generic message response."""

    message: str


class UserResponse(BaseModel):
    """Response model for user info."""

    id: str
    email: str
    name: Optional[str]
    email_verified: bool
    role: str


# ============= Auth Endpoints =============


@router.post("/register", response_model=TokenResponse)
async def register(request: RegisterRequest, background_tasks: BackgroundTasks):
    """
    Register a new user with email and password.

    Returns access and refresh tokens on successful registration.
    Sends a verification email in the background.
    """
    auth_service = get_auth_service()

    try:
        user = await auth_service.register_user(
            email=request.email,
            password=request.password,
            name=request.name,
        )

        # Create verification token and send email in background
        # This is non-critical - don't fail registration if email fails
        try:
            verification_token = await auth_service.create_email_verification_token(
                user.id
            )
            email_sender = get_auth_email_sender()
            background_tasks.add_task(
                email_sender.send_email_verification,
                to_email=user.email,
                verification_token=verification_token,
                user_name=user.name,
            )
        except Exception as e:
            logger.warning(f"Failed to queue verification email for {user.email}: {e}")

        tokens = await auth_service.create_tokens(user)
        return TokenResponse(**tokens)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """
    Login with email and password.

    Returns access and refresh tokens on successful authentication.
    """
    auth_service = get_auth_service()

    user = await auth_service.authenticate_user(request.email, request.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    tokens = await auth_service.create_tokens(user)
    return TokenResponse(**tokens)


@router.post("/logout", response_model=MessageResponse)
async def logout(request: LogoutRequest):
    """
    Logout by revoking the refresh token.

    This invalidates the refresh token so it cannot be used to get new access tokens.
    """
    auth_service = get_auth_service()

    revoked = await auth_service.revoke_refresh_token(request.refresh_token)
    if not revoked:
        raise HTTPException(status_code=400, detail="Invalid refresh token")

    return MessageResponse(message="Successfully logged out")


@router.post("/refresh", response_model=TokenResponse)
async def refresh_tokens(request: RefreshRequest):
    """
    Refresh access token using a refresh token.

    The old refresh token is invalidated and a new one is returned (token rotation).
    """
    auth_service = get_auth_service()

    tokens = await auth_service.refresh_access_token(request.refresh_token)
    if not tokens:
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")

    return TokenResponse(**tokens)


@router.post("/password-reset/request", response_model=MessageResponse)
async def request_password_reset(
    request: PasswordResetRequest, background_tasks: BackgroundTasks
):
    """
    Request a password reset email.

    Always returns success to prevent email enumeration attacks.
    If the email exists, a password reset email will be sent.
    """
    auth_service = get_auth_service()

    user = await auth_service.get_user_by_email(request.email)
    if user:
        token = await auth_service.create_password_reset_token(user.id)
        email_sender = get_auth_email_sender()
        background_tasks.add_task(
            email_sender.send_password_reset_email,
            to_email=user.email,
            reset_token=token,
            user_name=user.name,
        )
        logger.info(f"Password reset email queued for user {user.id}")

    # Always return success to prevent email enumeration
    return MessageResponse(
        message="If the email exists, a password reset link has been sent"
    )


@router.post("/password-reset/confirm", response_model=MessageResponse)
async def confirm_password_reset(request: PasswordResetConfirm):
    """
    Reset password using a password reset token.

    All existing sessions (refresh tokens) will be invalidated.
    """
    auth_service = get_auth_service()

    success = await auth_service.reset_password(request.token, request.new_password)
    if not success:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")

    return MessageResponse(message="Password has been reset successfully")


# ============= Email Verification Endpoints =============


class EmailVerificationRequest(BaseModel):
    """Request model for email verification."""

    token: str


class ResendVerificationRequest(BaseModel):
    """Request model for resending verification email."""

    email: EmailStr


@router.post("/email/verify", response_model=MessageResponse)
async def verify_email(request: EmailVerificationRequest):
    """
    Verify email address using a verification token.

    Marks the user's email as verified if the token is valid.
    """
    auth_service = get_auth_service()

    success = await auth_service.verify_email_token(request.token)
    if not success:
        raise HTTPException(
            status_code=400, detail="Invalid or expired verification token"
        )

    return MessageResponse(message="Email verified successfully")


@router.post("/email/resend-verification", response_model=MessageResponse)
async def resend_verification_email(
    request: ResendVerificationRequest, background_tasks: BackgroundTasks
):
    """
    Resend email verification email.

    Always returns success to prevent email enumeration attacks.
    If the email exists and is not verified, a new verification email will be sent.
    """
    auth_service = get_auth_service()

    user = await auth_service.get_user_by_email(request.email)
    if user and not user.emailVerified:
        token = await auth_service.create_email_verification_token(user.id)
        email_sender = get_auth_email_sender()
        background_tasks.add_task(
            email_sender.send_email_verification,
            to_email=user.email,
            verification_token=token,
            user_name=user.name,
        )
        logger.info(f"Verification email queued for user {user.id}")

    # Always return success to prevent email enumeration
    return MessageResponse(
        message="If the email exists and is not verified, a verification link has been sent"
    )


# ============= Google OAuth Endpoints =============

# Google userinfo endpoint for fetching user profile
GOOGLE_USERINFO_ENDPOINT = "https://www.googleapis.com/oauth2/v2/userinfo"


class GoogleLoginResponse(BaseModel):
    """Response model for Google OAuth login initiation."""

    url: str


def _get_google_oauth_handler():
    """Get a configured GoogleOAuthHandler instance."""
    # Lazy import to avoid circular imports
    from backend.integrations.oauth.google import GoogleOAuthHandler

    settings = Settings()

    client_id = settings.secrets.google_client_id
    client_secret = settings.secrets.google_client_secret

    if not client_id or not client_secret:
        raise HTTPException(
            status_code=500,
            detail="Google OAuth is not configured. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET.",
        )

    # Construct the redirect URI - this should point to the frontend's callback
    # which will then call our /auth/google/callback endpoint
    frontend_base_url = settings.config.frontend_base_url or "http://localhost:3000"
    redirect_uri = f"{frontend_base_url}/auth/callback"

    return GoogleOAuthHandler(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
    )


@router.get("/google/login", response_model=GoogleLoginResponse)
async def google_login(request: Request):
    """
    Initiate Google OAuth flow.

    Returns the Google OAuth authorization URL to redirect the user to.
    """
    try:
        handler = _get_google_oauth_handler()
        state = _generate_state()

        # Get the authorization URL with default scopes (email, profile, openid)
        auth_url = handler.get_login_url(
            scopes=[],  # Will use DEFAULT_SCOPES from handler
            state=state,
            code_challenge=None,  # Not using PKCE for server-side flow
        )

        logger.info(f"Generated Google OAuth URL for state: {state[:8]}...")
        return GoogleLoginResponse(url=auth_url)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to initiate Google OAuth: {e}")
        raise HTTPException(status_code=500, detail="Failed to initiate Google OAuth")


@router.get("/google/callback", response_model=TokenResponse)
async def google_callback(request: Request, code: str, state: Optional[str] = None):
    """
    Handle Google OAuth callback.

    Exchanges the authorization code for user info and creates/updates the user.
    Returns access and refresh tokens.
    """
    # Validate state to prevent CSRF attacks
    if not state or not _validate_state(state):
        logger.warning(
            f"Invalid or missing OAuth state: {state[:8] if state else 'None'}..."
        )
        raise HTTPException(status_code=400, detail="Invalid or expired OAuth state")

    try:
        handler = _get_google_oauth_handler()

        # Exchange the authorization code for Google credentials
        logger.info("Exchanging authorization code for tokens...")
        google_creds = await handler.exchange_code_for_tokens(
            code=code,
            scopes=[],  # Will use the scopes from the initial request
            code_verifier=None,
        )

        # The handler returns OAuth2Credentials with email in username field
        email = google_creds.username
        if not email:
            raise HTTPException(
                status_code=400, detail="Failed to retrieve email from Google"
            )

        # Fetch full user info to get Google user ID and name
        # Lazy import to avoid circular imports
        from google.auth.transport.requests import AuthorizedSession
        from google.oauth2.credentials import Credentials

        # We need to create Google Credentials object to use with AuthorizedSession
        creds = Credentials(
            token=google_creds.access_token.get_secret_value(),
            refresh_token=(
                google_creds.refresh_token.get_secret_value()
                if google_creds.refresh_token
                else None
            ),
            token_uri="https://oauth2.googleapis.com/token",
            client_id=handler.client_id,
            client_secret=handler.client_secret,
        )

        session = AuthorizedSession(creds)
        userinfo_response = session.get(GOOGLE_USERINFO_ENDPOINT)

        if not userinfo_response.ok:
            logger.error(
                f"Failed to fetch Google userinfo: {userinfo_response.status_code}"
            )
            raise HTTPException(
                status_code=400, detail="Failed to fetch user info from Google"
            )

        userinfo = userinfo_response.json()
        google_id = userinfo.get("id")
        name = userinfo.get("name")
        email_verified = userinfo.get("verified_email", False)

        if not google_id:
            raise HTTPException(
                status_code=400, detail="Failed to retrieve Google user ID"
            )

        logger.info(f"Google OAuth successful for user: {email}")

        # Create or update the user in our database
        auth_service = get_auth_service()
        user = await auth_service.create_or_update_google_user(
            google_id=google_id,
            email=email,
            name=name,
            email_verified=email_verified,
        )

        # Generate our JWT tokens
        tokens = await auth_service.create_tokens(user)

        return TokenResponse(**tokens)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Google OAuth callback failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to complete Google OAuth")
