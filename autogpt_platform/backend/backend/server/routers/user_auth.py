"""
User authentication router for native FastAPI auth.

This router provides endpoints that are compatible with the Supabase Auth API
structure, allowing the frontend to migrate without code changes.

Endpoints:
- POST /auth/signup - Register a new user
- POST /auth/login - Login with email/password
- POST /auth/logout - Logout (clear session)
- POST /auth/refresh - Refresh access token
- GET  /auth/me - Get current user
- POST /auth/password/reset - Request password reset email
- POST /auth/password/set - Set new password from reset link
- GET  /auth/verify-email - Verify email from magic link
- GET  /auth/oauth/google/authorize - Get Google OAuth URL
- GET  /auth/oauth/google/callback - Handle Google OAuth callback

Admin Endpoints:
- GET  /auth/admin/users - List users (admin only)
- GET  /auth/admin/users/{user_id} - Get user details (admin only)
- POST /auth/admin/users/{user_id}/impersonate - Get impersonation token (admin only)
"""

import logging
import os
import secrets
import uuid
from datetime import datetime, timezone
from typing import List, Optional
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, EmailStr
from prisma.models import User

from backend.data.auth.email_service import get_auth_email_service
from backend.data.auth.magic_links import (
    create_email_verification_link,
    create_password_reset_link,
    verify_email_token,
    verify_password_reset_token,
)
from backend.data.auth.password import hash_password, needs_rehash, verify_password
from backend.data.auth.tokens import (
    ACCESS_TOKEN_TTL,
    REFRESH_TOKEN_TTL,
    create_access_token,
    create_refresh_token_db,
    decode_access_token,
    revoke_all_user_refresh_tokens,
    revoke_refresh_token,
    validate_refresh_token,
)
from backend.util.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()

router = APIRouter(prefix="/auth", tags=["user-auth"])

# Cookie configuration
ACCESS_TOKEN_COOKIE = "access_token"
REFRESH_TOKEN_COOKIE = "refresh_token"
OAUTH_STATE_COOKIE = "oauth_state"

# Header for admin impersonation (matches existing autogpt_libs pattern)
IMPERSONATION_HEADER = "X-Act-As-User-Id"


# ============================================================================
# Admin Role Detection
# ============================================================================


def _get_admin_domains() -> set[str]:
    """Get set of email domains that grant admin role."""
    domains_str = settings.config.admin_email_domains
    if not domains_str:
        return set()
    return {d.strip().lower() for d in domains_str.split(",") if d.strip()}


def _get_admin_emails() -> set[str]:
    """Get set of specific email addresses that grant admin role."""
    emails_str = settings.config.admin_emails
    if not emails_str:
        return set()
    return {e.strip().lower() for e in emails_str.split(",") if e.strip()}


def get_user_role(email: str) -> str:
    """
    Determine user role based on email.

    Returns "admin" if:
    - Email domain is in admin_email_domains list
    - Email is in admin_emails list

    Otherwise returns "authenticated".
    """
    email_lower = email.lower()
    domain = email_lower.split("@")[-1] if "@" in email_lower else ""

    # Check specific emails first
    if email_lower in _get_admin_emails():
        return "admin"

    # Check domains
    if domain in _get_admin_domains():
        return "admin"

    return "authenticated"


# ============================================================================
# Request/Response Models
# ============================================================================


class SignupRequest(BaseModel):
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class PasswordResetRequest(BaseModel):
    email: EmailStr


class PasswordSetRequest(BaseModel):
    token: str
    password: str


class UserResponse(BaseModel):
    id: str
    email: str
    email_verified: bool
    name: Optional[str] = None
    created_at: datetime
    role: Optional[str] = None

    @staticmethod
    def from_db(user: User, include_role: bool = False) -> "UserResponse":
        return UserResponse(
            id=user.id,
            email=user.email,
            email_verified=user.emailVerified,
            name=user.name,
            created_at=user.createdAt,
            role=get_user_role(user.email) if include_role else None,
        )


class AuthResponse(BaseModel):
    """Response matching Supabase auth response structure."""

    user: UserResponse
    access_token: str
    refresh_token: str
    expires_in: int
    token_type: str = "bearer"


class MessageResponse(BaseModel):
    message: str


class AdminUserListResponse(BaseModel):
    users: List[UserResponse]
    total: int
    page: int
    page_size: int


class ImpersonationResponse(BaseModel):
    access_token: str
    impersonated_user: UserResponse
    expires_in: int


# ============================================================================
# Cookie Helpers
# ============================================================================


def _is_production() -> bool:
    return os.getenv("APP_ENV", "local").lower() in ("production", "prod")


def _set_auth_cookies(response: Response, access_token: str, refresh_token: str):
    """Set authentication cookies on the response."""
    secure = _is_production()

    # Access token: accessible to JavaScript for API calls
    response.set_cookie(
        key=ACCESS_TOKEN_COOKIE,
        value=access_token,
        httponly=False,  # JS needs access for Authorization header
        secure=secure,
        samesite="lax",
        max_age=int(ACCESS_TOKEN_TTL.total_seconds()),
        path="/",
    )

    # Refresh token: httpOnly, restricted path
    response.set_cookie(
        key=REFRESH_TOKEN_COOKIE,
        value=refresh_token,
        httponly=True,  # Not accessible to JavaScript
        secure=secure,
        samesite="strict",
        max_age=int(REFRESH_TOKEN_TTL.total_seconds()),
        path="/api/auth/refresh",  # Only sent to refresh endpoint
    )


def _clear_auth_cookies(response: Response):
    """Clear authentication cookies."""
    response.delete_cookie(key=ACCESS_TOKEN_COOKIE, path="/")
    response.delete_cookie(key=REFRESH_TOKEN_COOKIE, path="/api/auth/refresh")


def _get_access_token(request: Request) -> Optional[str]:
    """Get access token from cookie or Authorization header."""
    # Try cookie first
    token = request.cookies.get(ACCESS_TOKEN_COOKIE)
    if token:
        return token

    # Try Authorization header
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header[7:]

    return None


# ============================================================================
# Auth Dependencies
# ============================================================================


async def get_current_user_from_token(request: Request) -> Optional[User]:
    """Get the current user from the access token."""
    access_token = _get_access_token(request)
    if not access_token:
        return None

    payload = decode_access_token(access_token)
    if not payload:
        return None

    return await User.prisma().find_unique(where={"id": payload.sub})


async def require_auth(request: Request) -> User:
    """Require authentication - returns user or raises 401."""
    user = await get_current_user_from_token(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


async def require_admin(request: Request) -> User:
    """Require admin authentication - returns user or raises 401/403."""
    user = await require_auth(request)
    role = get_user_role(user.email)
    if role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


# ============================================================================
# Authentication Endpoints
# ============================================================================


@router.post("/signup", response_model=MessageResponse)
async def signup(data: SignupRequest):
    """
    Register a new user.

    Returns a message prompting the user to verify their email.
    No automatic login until email is verified.
    """
    # Check if email already exists
    existing = await User.prisma().find_unique(where={"email": data.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Validate password strength
    if len(data.password) < 8:
        raise HTTPException(
            status_code=400, detail="Password must be at least 8 characters"
        )

    # Create user with hashed password
    password_hash = hash_password(data.password)
    user = await User.prisma().create(
        data={
            "id": str(uuid.uuid4()),
            "email": data.email,
            "passwordHash": password_hash,
            "authProvider": "password",
            "emailVerified": False,
        }
    )

    # Create verification link and send email
    token = await create_email_verification_link(data.email)
    email_service = get_auth_email_service()
    email_sent = email_service.send_verification_email(data.email, token)

    if not email_sent:
        logger.warning(f"Failed to send verification email to {data.email}")
        # Still log the token for development
        logger.info(f"Verification token for {data.email}: {token}")

    return MessageResponse(
        message="Please check your email to verify your account"
    )


@router.post("/login", response_model=AuthResponse)
async def login(data: LoginRequest, response: Response):
    """
    Login with email and password.

    Sets httpOnly cookies for session management.
    """
    user = await User.prisma().find_unique(where={"email": data.email})

    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # Check if this is a migrated user without password
    if user.passwordHash is None:
        if user.migratedFromSupabase:
            # Send password reset email for migrated user
            token = await create_password_reset_link(data.email, user.id)
            email_service = get_auth_email_service()
            email_service.send_migrated_user_password_reset(data.email, token)
            raise HTTPException(
                status_code=400,
                detail="Please check your email to set your password",
            )
        else:
            # OAuth user trying to login with password
            raise HTTPException(
                status_code=400,
                detail=f"This account uses {user.authProvider} login",
            )

    # Verify password
    if not verify_password(user.passwordHash, data.password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # Check if email is verified
    if not user.emailVerified:
        raise HTTPException(
            status_code=400, detail="Please verify your email before logging in"
        )

    # Rehash password if needed (transparent security upgrade)
    if needs_rehash(user.passwordHash):
        new_hash = hash_password(data.password)
        await User.prisma().update(
            where={"id": user.id}, data={"passwordHash": new_hash}
        )

    # Create tokens
    role = get_user_role(user.email)
    access_token = create_access_token(user.id, user.email, role)
    refresh_token, _ = await create_refresh_token_db(user.id)

    # Set cookies
    _set_auth_cookies(response, access_token, refresh_token)

    return AuthResponse(
        user=UserResponse.from_db(user),
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=int(ACCESS_TOKEN_TTL.total_seconds()),
    )


@router.post("/logout", response_model=MessageResponse)
async def logout(request: Request, response: Response, scope: str = Query("local")):
    """
    Logout the current user.

    Args:
        scope: "local" to clear current session, "global" to revoke all sessions.
    """
    # Get refresh token to revoke
    refresh_token = request.cookies.get(REFRESH_TOKEN_COOKIE)

    if scope == "global":
        # Get user from access token
        access_token = _get_access_token(request)
        if access_token:
            payload = decode_access_token(access_token)
            if payload:
                await revoke_all_user_refresh_tokens(payload.sub)
    elif refresh_token:
        await revoke_refresh_token(refresh_token)

    # Clear cookies
    _clear_auth_cookies(response)

    return MessageResponse(message="Logged out successfully")


@router.post("/refresh", response_model=AuthResponse)
async def refresh(request: Request, response: Response):
    """
    Refresh the access token using the refresh token.
    """
    refresh_token = request.cookies.get(REFRESH_TOKEN_COOKIE)

    if not refresh_token:
        raise HTTPException(status_code=401, detail="No refresh token")

    # Validate refresh token
    user_id = await validate_refresh_token(refresh_token)
    if not user_id:
        _clear_auth_cookies(response)
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")

    # Get user
    user = await User.prisma().find_unique(where={"id": user_id})
    if not user:
        _clear_auth_cookies(response)
        raise HTTPException(status_code=401, detail="User not found")

    # Revoke old refresh token
    await revoke_refresh_token(refresh_token)

    # Create new tokens
    role = get_user_role(user.email)
    new_access_token = create_access_token(user.id, user.email, role)
    new_refresh_token, _ = await create_refresh_token_db(user.id)

    # Set new cookies
    _set_auth_cookies(response, new_access_token, new_refresh_token)

    return AuthResponse(
        user=UserResponse.from_db(user),
        access_token=new_access_token,
        refresh_token=new_refresh_token,
        expires_in=int(ACCESS_TOKEN_TTL.total_seconds()),
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user(request: Request):
    """
    Get the currently authenticated user.

    Supports admin impersonation via X-Act-As-User-Id header.
    """
    access_token = _get_access_token(request)

    if not access_token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    payload = decode_access_token(access_token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    # Check for impersonation header
    impersonate_user_id = request.headers.get(IMPERSONATION_HEADER, "").strip()
    if impersonate_user_id:
        # Verify caller is admin
        if payload.role != "admin":
            raise HTTPException(
                status_code=403, detail="Only admins can impersonate users"
            )

        # Log impersonation for audit
        logger.info(
            f"Admin impersonation: {payload.sub} ({payload.email}) "
            f"viewing as user {impersonate_user_id}"
        )

        user = await User.prisma().find_unique(where={"id": impersonate_user_id})
        if not user:
            raise HTTPException(status_code=404, detail="Impersonated user not found")
    else:
        user = await User.prisma().find_unique(where={"id": payload.sub})
        if not user:
            raise HTTPException(status_code=401, detail="User not found")

    return UserResponse.from_db(user, include_role=True)


# ============================================================================
# Password Reset Endpoints
# ============================================================================


@router.post("/password/reset", response_model=MessageResponse)
async def request_password_reset(data: PasswordResetRequest):
    """
    Request a password reset email.
    """
    user = await User.prisma().find_unique(where={"email": data.email})

    # Always return success to prevent email enumeration
    if not user:
        return MessageResponse(message="If the email exists, a reset link has been sent")

    # Don't allow password reset for OAuth-only users
    if user.authProvider not in ("password", "supabase"):
        return MessageResponse(message="If the email exists, a reset link has been sent")

    # Create reset link and send email
    token = await create_password_reset_link(data.email, user.id)
    email_service = get_auth_email_service()
    email_service.send_password_reset_email(data.email, token)

    return MessageResponse(message="If the email exists, a reset link has been sent")


@router.post("/password/set", response_model=MessageResponse)
async def set_password(data: PasswordSetRequest, response: Response):
    """
    Set a new password using a reset token.
    """
    # Validate token
    result = await verify_password_reset_token(data.token)
    if not result:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")

    user_id, email = result

    # Validate password strength
    if len(data.password) < 8:
        raise HTTPException(
            status_code=400, detail="Password must be at least 8 characters"
        )

    # Update password and verify email (if not already)
    password_hash = hash_password(data.password)
    await User.prisma().update(
        where={"id": user_id},
        data={
            "passwordHash": password_hash,
            "emailVerified": True,
            "emailVerifiedAt": datetime.now(timezone.utc),
            "authProvider": "password",
            "migratedFromSupabase": False,  # Clear migration flag
        },
    )

    # Send notification that password was changed
    email_service = get_auth_email_service()
    email_service.send_password_changed_notification(email)

    # Revoke all existing sessions for security
    await revoke_all_user_refresh_tokens(user_id)

    # Clear any existing cookies
    _clear_auth_cookies(response)

    return MessageResponse(message="Password updated successfully. Please log in.")


# ============================================================================
# Email Verification
# ============================================================================


@router.get("/verify-email", response_model=MessageResponse)
async def verify_email(token: str = Query(...)):
    """
    Verify email address from magic link.
    """
    email = await verify_email_token(token)
    if not email:
        raise HTTPException(status_code=400, detail="Invalid or expired verification link")

    # Update user as verified
    user = await User.prisma().find_unique(where={"email": email})
    if user:
        await User.prisma().update(
            where={"id": user.id},
            data={
                "emailVerified": True,
                "emailVerifiedAt": datetime.now(timezone.utc),
            },
        )

    return MessageResponse(message="Email verified successfully. You can now log in.")


# ============================================================================
# Google OAuth
# ============================================================================

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"


def _get_google_config():
    """Get Google OAuth configuration from environment."""
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
    redirect_uri = os.getenv("GOOGLE_REDIRECT_URI", "")

    if not client_id or not client_secret:
        raise HTTPException(
            status_code=503, detail="Google OAuth not configured"
        )

    return client_id, client_secret, redirect_uri


@router.get("/oauth/google/authorize")
async def google_authorize(
    response: Response,
    redirect_to: str = Query("/marketplace", description="URL to redirect after auth"),
):
    """
    Initiate Google OAuth flow.

    Returns the authorization URL to redirect the user to.
    """
    client_id, _, redirect_uri = _get_google_config()

    # Generate state for CSRF protection
    state = secrets.token_urlsafe(32)

    # Store state and redirect_to in cookie
    secure = _is_production()
    response.set_cookie(
        key=OAUTH_STATE_COOKIE,
        value=f"{state}|{redirect_to}",
        httponly=True,
        secure=secure,
        samesite="lax",
        max_age=600,  # 10 minutes
    )

    # Build authorization URL
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "access_type": "offline",
        "prompt": "consent",
    }

    auth_url = f"{GOOGLE_AUTH_URL}?{urlencode(params)}"

    return {"url": auth_url}


@router.get("/oauth/google/callback")
async def google_callback(
    request: Request,
    response: Response,
    code: str = Query(...),
    state: str = Query(...),
):
    """
    Handle Google OAuth callback.

    Exchanges the authorization code for tokens and creates/updates the user.
    """
    client_id, client_secret, redirect_uri = _get_google_config()

    # Verify state
    stored_state_cookie = request.cookies.get(OAUTH_STATE_COOKIE)
    if not stored_state_cookie:
        raise HTTPException(status_code=400, detail="Missing OAuth state")

    stored_state, redirect_to = stored_state_cookie.split("|", 1)
    if state != stored_state:
        raise HTTPException(status_code=400, detail="Invalid OAuth state")

    # Clear state cookie
    response.delete_cookie(key=OAUTH_STATE_COOKIE)

    # Exchange code for tokens
    async with httpx.AsyncClient() as client:
        token_response = await client.post(
            GOOGLE_TOKEN_URL,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": redirect_uri,
            },
        )

        if token_response.status_code != 200:
            logger.error(f"Google token exchange failed: {token_response.text}")
            raise HTTPException(status_code=400, detail="Failed to exchange code")

        tokens = token_response.json()
        google_access_token = tokens.get("access_token")

        # Get user info
        userinfo_response = await client.get(
            GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {google_access_token}"},
        )

        if userinfo_response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to get user info")

        userinfo = userinfo_response.json()

    email = userinfo.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="Email not provided by Google")

    # Get or create user
    user = await User.prisma().find_unique(where={"email": email})

    if user:
        # Update existing user if needed
        if user.authProvider == "supabase":
            await User.prisma().update(
                where={"id": user.id},
                data={"authProvider": "google"},
            )
    else:
        # Create new user
        user = await User.prisma().create(
            data={
                "id": str(uuid.uuid4()),
                "email": email,
                "name": userinfo.get("name"),
                "emailVerified": True,  # Google verifies emails
                "emailVerifiedAt": datetime.now(timezone.utc),
                "authProvider": "google",
            }
        )

    # Create tokens
    role = get_user_role(email)
    access_token = create_access_token(user.id, user.email, role)
    refresh_token, _ = await create_refresh_token_db(user.id)

    # Set cookies
    _set_auth_cookies(response, access_token, refresh_token)

    # Redirect to frontend
    frontend_url = os.getenv("FRONTEND_BASE_URL", "http://localhost:3000")
    from fastapi.responses import RedirectResponse

    return RedirectResponse(url=f"{frontend_url}{redirect_to}")


# ============================================================================
# Admin Routes
# ============================================================================


@router.get("/admin/users", response_model=AdminUserListResponse)
async def list_users(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    search: Optional[str] = Query(None, description="Search by email"),
    admin_user: User = Depends(require_admin),
):
    """
    List all users (admin only).
    """
    skip = (page - 1) * page_size

    where_clause = {}
    if search:
        where_clause["email"] = {"contains": search, "mode": "insensitive"}

    users = await User.prisma().find_many(
        where=where_clause,
        skip=skip,
        take=page_size,
        order={"createdAt": "desc"},
    )

    total = await User.prisma().count(where=where_clause)

    return AdminUserListResponse(
        users=[UserResponse.from_db(u, include_role=True) for u in users],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/admin/users/{user_id}", response_model=UserResponse)
async def get_user_by_id(
    user_id: str,
    admin_user: User = Depends(require_admin),
):
    """
    Get a specific user by ID (admin only).
    """
    user = await User.prisma().find_unique(where={"id": user_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return UserResponse.from_db(user, include_role=True)


@router.post("/admin/users/{user_id}/impersonate", response_model=ImpersonationResponse)
async def impersonate_user(
    request: Request,
    user_id: str,
    admin_user: User = Depends(require_admin),
):
    """
    Get an access token to impersonate a user (admin only).

    This token can be used with the Authorization header to act as the user.
    All actions are logged for audit purposes.
    """
    target_user = await User.prisma().find_unique(where={"id": user_id})
    if not target_user:
        raise HTTPException(status_code=404, detail="User not found")

    # Log the impersonation
    logger.warning(
        f"ADMIN IMPERSONATION: Admin {admin_user.id} ({admin_user.email}) "
        f"generated impersonation token for user {target_user.id} ({target_user.email})"
    )

    # Create an access token for the target user (but with original role for safety)
    # The impersonation is tracked via the audit log
    role = get_user_role(target_user.email)
    access_token = create_access_token(target_user.id, target_user.email, role)

    return ImpersonationResponse(
        access_token=access_token,
        impersonated_user=UserResponse.from_db(target_user, include_role=True),
        expires_in=int(ACCESS_TOKEN_TTL.total_seconds()),
    )


@router.post("/admin/users/{user_id}/force-password-reset", response_model=MessageResponse)
async def force_password_reset(
    user_id: str,
    admin_user: User = Depends(require_admin),
):
    """
    Force send a password reset email to a user (admin only).

    Useful for helping users who are locked out.
    """
    user = await User.prisma().find_unique(where={"id": user_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Create and send password reset
    token = await create_password_reset_link(user.email, user.id)
    email_service = get_auth_email_service()
    email_sent = email_service.send_password_reset_email(user.email, token)

    # Log the action
    logger.info(
        f"Admin {admin_user.id} ({admin_user.email}) "
        f"triggered password reset for user {user.id} ({user.email})"
    )

    if email_sent:
        return MessageResponse(message=f"Password reset email sent to {user.email}")
    else:
        return MessageResponse(message="Email service unavailable, reset link logged")


@router.post("/admin/users/{user_id}/revoke-sessions", response_model=MessageResponse)
async def revoke_user_sessions(
    user_id: str,
    admin_user: User = Depends(require_admin),
):
    """
    Revoke all sessions for a user (admin only).

    Useful for security incidents.
    """
    user = await User.prisma().find_unique(where={"id": user_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    count = await revoke_all_user_refresh_tokens(user_id)

    # Log the action
    logger.warning(
        f"Admin {admin_user.id} ({admin_user.email}) "
        f"revoked all sessions for user {user.id} ({user.email}). "
        f"Revoked {count} refresh tokens."
    )

    return MessageResponse(message=f"Revoked {count} sessions for user {user.email}")
