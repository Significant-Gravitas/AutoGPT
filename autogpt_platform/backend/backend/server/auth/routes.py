"""
Authentication API routes.

Provides endpoints for:
- User registration and login
- Token refresh and logout
- Password reset
- Google OAuth
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, EmailStr, Field

from .service import AuthService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

# Singleton auth service instance
_auth_service: Optional[AuthService] = None


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
async def register(request: RegisterRequest):
    """
    Register a new user with email and password.

    Returns access and refresh tokens on successful registration.
    """
    auth_service = get_auth_service()

    try:
        user = await auth_service.register_user(
            email=request.email,
            password=request.password,
            name=request.name,
        )
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
async def request_password_reset(request: PasswordResetRequest):
    """
    Request a password reset email.

    Always returns success to prevent email enumeration attacks.
    If the email exists, a reset token will be created (email sending not implemented).
    """
    auth_service = get_auth_service()

    user = await auth_service.get_user_by_email(request.email)
    if user:
        token = await auth_service.create_password_reset_token(user.id)
        # TODO: Send password reset email with token
        logger.info(f"Password reset token created for user {user.id}: {token[:8]}...")

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


# ============= Google OAuth Endpoints =============


@router.get("/google/login")
async def google_login(request: Request):
    """
    Initiate Google OAuth flow.

    Returns the Google OAuth authorization URL to redirect the user to.
    """
    # TODO: Implement Google OAuth using authlib
    raise HTTPException(status_code=501, detail="Google OAuth not yet implemented")


@router.get("/google/callback")
async def google_callback(request: Request, code: str, state: Optional[str] = None):
    """
    Handle Google OAuth callback.

    Exchanges the authorization code for user info and creates/updates the user.
    Returns tokens or redirects to frontend with tokens.
    """
    # TODO: Implement Google OAuth callback
    raise HTTPException(status_code=501, detail="Google OAuth not yet implemented")
