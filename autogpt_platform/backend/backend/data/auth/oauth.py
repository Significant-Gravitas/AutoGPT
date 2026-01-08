"""
OAuth 2.0 Provider Data Layer

Handles management of OAuth applications, authorization codes,
access tokens, and refresh tokens.

Hashing strategy:
- Access tokens & Refresh tokens: SHA256 (deterministic, allows direct lookup by hash)
- Client secrets: Scrypt with salt (lookup by client_id, then verify with salt)
"""

import hashlib
import logging
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

from autogpt_libs.api_key.keysmith import APIKeySmith
from prisma.enums import APIKeyPermission as APIPermission
from prisma.models import OAuthAccessToken as PrismaOAuthAccessToken
from prisma.models import OAuthApplication as PrismaOAuthApplication
from prisma.models import OAuthAuthorizationCode as PrismaOAuthAuthorizationCode
from prisma.models import OAuthRefreshToken as PrismaOAuthRefreshToken
from prisma.types import OAuthApplicationUpdateInput
from pydantic import BaseModel, Field, SecretStr

from .base import APIAuthorizationInfo

logger = logging.getLogger(__name__)
keysmith = APIKeySmith()  # Only used for client secret hashing (Scrypt)


def _generate_token() -> str:
    """Generate a cryptographically secure random token."""
    return secrets.token_urlsafe(32)


def _hash_token(token: str) -> str:
    """Hash a token using SHA256 (deterministic, for direct lookup)."""
    return hashlib.sha256(token.encode()).hexdigest()


# Token TTLs
AUTHORIZATION_CODE_TTL = timedelta(minutes=10)
ACCESS_TOKEN_TTL = timedelta(hours=1)
REFRESH_TOKEN_TTL = timedelta(days=30)

ACCESS_TOKEN_PREFIX = "agpt_xt_"
REFRESH_TOKEN_PREFIX = "agpt_rt_"


# ============================================================================
# Exception Classes
# ============================================================================


class OAuthError(Exception):
    """Base OAuth error"""

    pass


class InvalidClientError(OAuthError):
    """Invalid client_id or client_secret"""

    pass


class InvalidGrantError(OAuthError):
    """Invalid or expired authorization code/refresh token"""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Invalid grant: {reason}")


class InvalidTokenError(OAuthError):
    """Invalid, expired, or revoked token"""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Invalid token: {reason}")


# ============================================================================
# Data Models
# ============================================================================


class OAuthApplicationInfo(BaseModel):
    """OAuth application information (without client secret hash)"""

    id: str
    name: str
    description: Optional[str] = None
    logo_url: Optional[str] = None
    client_id: str
    redirect_uris: list[str]
    grant_types: list[str]
    scopes: list[APIPermission]
    owner_id: str
    is_active: bool
    created_at: datetime
    updated_at: datetime

    @staticmethod
    def from_db(app: PrismaOAuthApplication):
        return OAuthApplicationInfo(
            id=app.id,
            name=app.name,
            description=app.description,
            logo_url=app.logoUrl,
            client_id=app.clientId,
            redirect_uris=app.redirectUris,
            grant_types=app.grantTypes,
            scopes=[APIPermission(s) for s in app.scopes],
            owner_id=app.ownerId,
            is_active=app.isActive,
            created_at=app.createdAt,
            updated_at=app.updatedAt,
        )


class OAuthApplicationInfoWithSecret(OAuthApplicationInfo):
    """OAuth application with client secret hash (for validation)"""

    client_secret_hash: str
    client_secret_salt: str

    @staticmethod
    def from_db(app: PrismaOAuthApplication):
        return OAuthApplicationInfoWithSecret(
            **OAuthApplicationInfo.from_db(app).model_dump(),
            client_secret_hash=app.clientSecret,
            client_secret_salt=app.clientSecretSalt,
        )

    def verify_secret(self, plaintext_secret: str) -> bool:
        """Verify a plaintext client secret against the stored hash"""
        # Use keysmith.verify_key() with stored salt
        return keysmith.verify_key(
            plaintext_secret, self.client_secret_hash, self.client_secret_salt
        )


class OAuthAuthorizationCodeInfo(BaseModel):
    """Authorization code information"""

    id: str
    code: str
    created_at: datetime
    expires_at: datetime
    application_id: str
    user_id: str
    scopes: list[APIPermission]
    redirect_uri: str
    code_challenge: Optional[str] = None
    code_challenge_method: Optional[str] = None
    used_at: Optional[datetime] = None

    @property
    def is_used(self) -> bool:
        return self.used_at is not None

    @staticmethod
    def from_db(code: PrismaOAuthAuthorizationCode):
        return OAuthAuthorizationCodeInfo(
            id=code.id,
            code=code.code,
            created_at=code.createdAt,
            expires_at=code.expiresAt,
            application_id=code.applicationId,
            user_id=code.userId,
            scopes=[APIPermission(s) for s in code.scopes],
            redirect_uri=code.redirectUri,
            code_challenge=code.codeChallenge,
            code_challenge_method=code.codeChallengeMethod,
            used_at=code.usedAt,
        )


class OAuthAccessTokenInfo(APIAuthorizationInfo):
    """Access token information"""

    id: str
    expires_at: datetime  # type: ignore
    application_id: str

    type: Literal["oauth"] = "oauth"  # type: ignore

    @staticmethod
    def from_db(token: PrismaOAuthAccessToken):
        return OAuthAccessTokenInfo(
            id=token.id,
            user_id=token.userId,
            scopes=[APIPermission(s) for s in token.scopes],
            created_at=token.createdAt,
            expires_at=token.expiresAt,
            last_used_at=None,
            revoked_at=token.revokedAt,
            application_id=token.applicationId,
        )


class OAuthAccessToken(OAuthAccessTokenInfo):
    """Access token with plaintext token included (sensitive)"""

    token: SecretStr = Field(description="Plaintext token (sensitive)")

    @staticmethod
    def from_db(token: PrismaOAuthAccessToken, plaintext_token: str):  # type: ignore
        return OAuthAccessToken(
            **OAuthAccessTokenInfo.from_db(token).model_dump(),
            token=SecretStr(plaintext_token),
        )


class OAuthRefreshTokenInfo(BaseModel):
    """Refresh token information"""

    id: str
    user_id: str
    scopes: list[APIPermission]
    created_at: datetime
    expires_at: datetime
    application_id: str
    revoked_at: Optional[datetime] = None

    @property
    def is_revoked(self) -> bool:
        return self.revoked_at is not None

    @staticmethod
    def from_db(token: PrismaOAuthRefreshToken):
        return OAuthRefreshTokenInfo(
            id=token.id,
            user_id=token.userId,
            scopes=[APIPermission(s) for s in token.scopes],
            created_at=token.createdAt,
            expires_at=token.expiresAt,
            application_id=token.applicationId,
            revoked_at=token.revokedAt,
        )


class OAuthRefreshToken(OAuthRefreshTokenInfo):
    """Refresh token with plaintext token included (sensitive)"""

    token: SecretStr = Field(description="Plaintext token (sensitive)")

    @staticmethod
    def from_db(token: PrismaOAuthRefreshToken, plaintext_token: str):  # type: ignore
        return OAuthRefreshToken(
            **OAuthRefreshTokenInfo.from_db(token).model_dump(),
            token=SecretStr(plaintext_token),
        )


class TokenIntrospectionResult(BaseModel):
    """Result of token introspection (RFC 7662)"""

    active: bool
    scopes: Optional[list[str]] = None
    client_id: Optional[str] = None
    user_id: Optional[str] = None
    exp: Optional[int] = None  # Unix timestamp
    token_type: Optional[Literal["access_token", "refresh_token"]] = None


# ============================================================================
# OAuth Application Management
# ============================================================================


async def get_oauth_application(client_id: str) -> Optional[OAuthApplicationInfo]:
    """Get OAuth application by client ID (without secret)"""
    app = await PrismaOAuthApplication.prisma().find_unique(
        where={"clientId": client_id}
    )
    if not app:
        return None
    return OAuthApplicationInfo.from_db(app)


async def get_oauth_application_with_secret(
    client_id: str,
) -> Optional[OAuthApplicationInfoWithSecret]:
    """Get OAuth application by client ID (with secret hash for validation)"""
    app = await PrismaOAuthApplication.prisma().find_unique(
        where={"clientId": client_id}
    )
    if not app:
        return None
    return OAuthApplicationInfoWithSecret.from_db(app)


async def validate_client_credentials(
    client_id: str, client_secret: str
) -> OAuthApplicationInfo:
    """
    Validate client credentials and return application info.

    Raises:
        InvalidClientError: If client_id or client_secret is invalid, or app is inactive
    """
    app = await get_oauth_application_with_secret(client_id)
    if not app:
        raise InvalidClientError("Invalid client_id")

    if not app.is_active:
        raise InvalidClientError("Application is not active")

    # Verify client secret
    if not app.verify_secret(client_secret):
        raise InvalidClientError("Invalid client_secret")

    # Return without secret hash
    return OAuthApplicationInfo(**app.model_dump(exclude={"client_secret_hash"}))


def validate_redirect_uri(app: OAuthApplicationInfo, redirect_uri: str) -> bool:
    """Validate that redirect URI is registered for the application"""
    return redirect_uri in app.redirect_uris


def validate_scopes(
    app: OAuthApplicationInfo, requested_scopes: list[APIPermission]
) -> bool:
    """Validate that all requested scopes are allowed for the application"""
    return all(scope in app.scopes for scope in requested_scopes)


# ============================================================================
# Authorization Code Flow
# ============================================================================


def _generate_authorization_code() -> str:
    """Generate a cryptographically secure authorization code"""
    # 32 bytes = 256 bits of entropy
    return secrets.token_urlsafe(32)


async def create_authorization_code(
    application_id: str,
    user_id: str,
    scopes: list[APIPermission],
    redirect_uri: str,
    code_challenge: Optional[str] = None,
    code_challenge_method: Optional[Literal["S256", "plain"]] = None,
) -> OAuthAuthorizationCodeInfo:
    """
    Create a new authorization code.
    Expires in 10 minutes and can only be used once.
    """
    code = _generate_authorization_code()
    now = datetime.now(timezone.utc)
    expires_at = now + AUTHORIZATION_CODE_TTL

    saved_code = await PrismaOAuthAuthorizationCode.prisma().create(
        data={
            "id": str(uuid.uuid4()),
            "code": code,
            "expiresAt": expires_at,
            "applicationId": application_id,
            "userId": user_id,
            "scopes": [s for s in scopes],
            "redirectUri": redirect_uri,
            "codeChallenge": code_challenge,
            "codeChallengeMethod": code_challenge_method,
        }
    )

    return OAuthAuthorizationCodeInfo.from_db(saved_code)


async def consume_authorization_code(
    code: str,
    application_id: str,
    redirect_uri: str,
    code_verifier: Optional[str] = None,
) -> tuple[str, list[APIPermission]]:
    """
    Consume an authorization code and return (user_id, scopes).

    This marks the code as used and validates:
    - Code exists and matches application
    - Code is not expired
    - Code has not been used
    - Redirect URI matches
    - PKCE code verifier matches (if code challenge was provided)

    Raises:
        InvalidGrantError: If code is invalid, expired, used, or PKCE fails
    """
    auth_code = await PrismaOAuthAuthorizationCode.prisma().find_unique(
        where={"code": code}
    )

    if not auth_code:
        raise InvalidGrantError("authorization code not found")

    # Validate application
    if auth_code.applicationId != application_id:
        raise InvalidGrantError(
            "authorization code does not belong to this application"
        )

    # Check if already used
    if auth_code.usedAt is not None:
        raise InvalidGrantError(
            f"authorization code already used at {auth_code.usedAt}"
        )

    # Check expiration
    now = datetime.now(timezone.utc)
    if auth_code.expiresAt < now:
        raise InvalidGrantError("authorization code expired")

    # Validate redirect URI
    if auth_code.redirectUri != redirect_uri:
        raise InvalidGrantError("redirect_uri mismatch")

    # Validate PKCE if code challenge was provided
    if auth_code.codeChallenge:
        if not code_verifier:
            raise InvalidGrantError("code_verifier required but not provided")

        if not _verify_pkce(
            code_verifier, auth_code.codeChallenge, auth_code.codeChallengeMethod
        ):
            raise InvalidGrantError("PKCE verification failed")

    # Mark code as used
    await PrismaOAuthAuthorizationCode.prisma().update(
        where={"code": code},
        data={"usedAt": now},
    )

    return auth_code.userId, [APIPermission(s) for s in auth_code.scopes]


def _verify_pkce(
    code_verifier: str, code_challenge: str, code_challenge_method: Optional[str]
) -> bool:
    """
    Verify PKCE code verifier against code challenge.

    Supports:
    - S256: SHA256(code_verifier) == code_challenge
    - plain: code_verifier == code_challenge
    """
    if code_challenge_method == "S256":
        # Hash the verifier with SHA256 and base64url encode
        hashed = hashlib.sha256(code_verifier.encode("ascii")).digest()
        computed_challenge = (
            secrets.token_urlsafe(len(hashed)).encode("ascii").decode("ascii")
        )
        # For proper base64url encoding
        import base64

        computed_challenge = (
            base64.urlsafe_b64encode(hashed).decode("ascii").rstrip("=")
        )
        return secrets.compare_digest(computed_challenge, code_challenge)
    elif code_challenge_method == "plain" or code_challenge_method is None:
        # Plain comparison
        return secrets.compare_digest(code_verifier, code_challenge)
    else:
        logger.warning(f"Unsupported code challenge method: {code_challenge_method}")
        return False


# ============================================================================
# Access Token Management
# ============================================================================


async def create_access_token(
    application_id: str, user_id: str, scopes: list[APIPermission]
) -> OAuthAccessToken:
    """
    Create a new access token.
    Returns OAuthAccessToken (with plaintext token).
    """
    plaintext_token = ACCESS_TOKEN_PREFIX + _generate_token()
    token_hash = _hash_token(plaintext_token)
    now = datetime.now(timezone.utc)
    expires_at = now + ACCESS_TOKEN_TTL

    saved_token = await PrismaOAuthAccessToken.prisma().create(
        data={
            "id": str(uuid.uuid4()),
            "token": token_hash,  # SHA256 hash for direct lookup
            "expiresAt": expires_at,
            "applicationId": application_id,
            "userId": user_id,
            "scopes": [s for s in scopes],
        }
    )

    return OAuthAccessToken.from_db(saved_token, plaintext_token=plaintext_token)


async def validate_access_token(
    token: str,
) -> tuple[OAuthAccessTokenInfo, OAuthApplicationInfo]:
    """
    Validate an access token and return token info.

    Raises:
        InvalidTokenError: If token is invalid, expired, or revoked
        InvalidClientError: If the client application is not marked as active
    """
    token_hash = _hash_token(token)

    # Direct lookup by hash
    access_token = await PrismaOAuthAccessToken.prisma().find_unique(
        where={"token": token_hash}, include={"Application": True}
    )

    if not access_token:
        raise InvalidTokenError("access token not found")

    if not access_token.Application:  # should be impossible
        raise InvalidClientError("Client application not found")

    if not access_token.Application.isActive:
        raise InvalidClientError("Client application is disabled")

    if access_token.revokedAt is not None:
        raise InvalidTokenError("access token has been revoked")

    # Check expiration
    now = datetime.now(timezone.utc)
    if access_token.expiresAt < now:
        raise InvalidTokenError("access token expired")

    return (
        OAuthAccessTokenInfo.from_db(access_token),
        OAuthApplicationInfo.from_db(access_token.Application),
    )


async def revoke_access_token(
    token: str, application_id: str
) -> OAuthAccessTokenInfo | None:
    """
    Revoke an access token.

    Args:
        token: The plaintext access token to revoke
        application_id: The application ID making the revocation request.
            Only tokens belonging to this application will be revoked.

    Returns:
        OAuthAccessTokenInfo if token was found and revoked, None otherwise.

    Note:
        Always performs exactly 2 DB queries regardless of outcome to prevent
        timing side-channel attacks that could reveal token existence.
    """
    try:
        token_hash = _hash_token(token)

        # Use update_many to filter by both token and applicationId
        updated_count = await PrismaOAuthAccessToken.prisma().update_many(
            where={
                "token": token_hash,
                "applicationId": application_id,
                "revokedAt": None,
            },
            data={"revokedAt": datetime.now(timezone.utc)},
        )

        # Always perform second query to ensure constant time
        result = await PrismaOAuthAccessToken.prisma().find_unique(
            where={"token": token_hash}
        )

        # Only return result if we actually revoked something
        if updated_count == 0:
            return None

        return OAuthAccessTokenInfo.from_db(result) if result else None
    except Exception as e:
        logger.exception(f"Error revoking access token: {e}")
        return None


# ============================================================================
# Refresh Token Management
# ============================================================================


async def create_refresh_token(
    application_id: str, user_id: str, scopes: list[APIPermission]
) -> OAuthRefreshToken:
    """
    Create a new refresh token.
    Returns OAuthRefreshToken (with plaintext token).
    """
    plaintext_token = REFRESH_TOKEN_PREFIX + _generate_token()
    token_hash = _hash_token(plaintext_token)
    now = datetime.now(timezone.utc)
    expires_at = now + REFRESH_TOKEN_TTL

    saved_token = await PrismaOAuthRefreshToken.prisma().create(
        data={
            "id": str(uuid.uuid4()),
            "token": token_hash,  # SHA256 hash for direct lookup
            "expiresAt": expires_at,
            "applicationId": application_id,
            "userId": user_id,
            "scopes": [s for s in scopes],
        }
    )

    return OAuthRefreshToken.from_db(saved_token, plaintext_token=plaintext_token)


async def refresh_tokens(
    refresh_token: str, application_id: str
) -> tuple[OAuthAccessToken, OAuthRefreshToken]:
    """
    Use a refresh token to create new access and refresh tokens.
    Returns (new_access_token, new_refresh_token) both with plaintext tokens included.

    Raises:
        InvalidGrantError: If refresh token is invalid, expired, or revoked
    """
    token_hash = _hash_token(refresh_token)

    # Direct lookup by hash
    rt = await PrismaOAuthRefreshToken.prisma().find_unique(where={"token": token_hash})

    if not rt:
        raise InvalidGrantError("refresh token not found")

    # NOTE: no need to check Application.isActive, this is checked by the token endpoint

    if rt.revokedAt is not None:
        raise InvalidGrantError("refresh token has been revoked")

    # Validate application
    if rt.applicationId != application_id:
        raise InvalidGrantError("refresh token does not belong to this application")

    # Check expiration
    now = datetime.now(timezone.utc)
    if rt.expiresAt < now:
        raise InvalidGrantError("refresh token expired")

    # Revoke old refresh token
    await PrismaOAuthRefreshToken.prisma().update(
        where={"token": token_hash},
        data={"revokedAt": now},
    )

    # Create new access and refresh tokens with same scopes
    scopes = [APIPermission(s) for s in rt.scopes]
    new_access_token = await create_access_token(
        rt.applicationId,
        rt.userId,
        scopes,
    )
    new_refresh_token = await create_refresh_token(
        rt.applicationId,
        rt.userId,
        scopes,
    )

    return new_access_token, new_refresh_token


async def revoke_refresh_token(
    token: str, application_id: str
) -> OAuthRefreshTokenInfo | None:
    """
    Revoke a refresh token.

    Args:
        token: The plaintext refresh token to revoke
        application_id: The application ID making the revocation request.
            Only tokens belonging to this application will be revoked.

    Returns:
        OAuthRefreshTokenInfo if token was found and revoked, None otherwise.

    Note:
        Always performs exactly 2 DB queries regardless of outcome to prevent
        timing side-channel attacks that could reveal token existence.
    """
    try:
        token_hash = _hash_token(token)

        # Use update_many to filter by both token and applicationId
        updated_count = await PrismaOAuthRefreshToken.prisma().update_many(
            where={
                "token": token_hash,
                "applicationId": application_id,
                "revokedAt": None,
            },
            data={"revokedAt": datetime.now(timezone.utc)},
        )

        # Always perform second query to ensure constant time
        result = await PrismaOAuthRefreshToken.prisma().find_unique(
            where={"token": token_hash}
        )

        # Only return result if we actually revoked something
        if updated_count == 0:
            return None

        return OAuthRefreshTokenInfo.from_db(result) if result else None
    except Exception as e:
        logger.exception(f"Error revoking refresh token: {e}")
        return None


# ============================================================================
# Token Introspection
# ============================================================================


async def introspect_token(
    token: str,
    token_type_hint: Optional[Literal["access_token", "refresh_token"]] = None,
) -> TokenIntrospectionResult:
    """
    Introspect a token and return its metadata (RFC 7662).

    Returns TokenIntrospectionResult with active=True and metadata if valid,
    or active=False if the token is invalid/expired/revoked.
    """
    # Try as access token first (or if hint says "access_token")
    if token_type_hint != "refresh_token":
        try:
            token_info, app = await validate_access_token(token)
            return TokenIntrospectionResult(
                active=True,
                scopes=list(s.value for s in token_info.scopes),
                client_id=app.client_id if app else None,
                user_id=token_info.user_id,
                exp=int(token_info.expires_at.timestamp()),
                token_type="access_token",
            )
        except InvalidTokenError:
            pass  # Try as refresh token

    # Try as refresh token
    token_hash = _hash_token(token)
    refresh_token = await PrismaOAuthRefreshToken.prisma().find_unique(
        where={"token": token_hash}
    )

    if refresh_token and refresh_token.revokedAt is None:
        # Check if valid (not expired)
        now = datetime.now(timezone.utc)
        if refresh_token.expiresAt > now:
            app = await get_oauth_application_by_id(refresh_token.applicationId)
            return TokenIntrospectionResult(
                active=True,
                scopes=list(s for s in refresh_token.scopes),
                client_id=app.client_id if app else None,
                user_id=refresh_token.userId,
                exp=int(refresh_token.expiresAt.timestamp()),
                token_type="refresh_token",
            )

    # Token not found or inactive
    return TokenIntrospectionResult(active=False)


async def get_oauth_application_by_id(app_id: str) -> Optional[OAuthApplicationInfo]:
    """Get OAuth application by ID"""
    app = await PrismaOAuthApplication.prisma().find_unique(where={"id": app_id})
    if not app:
        return None
    return OAuthApplicationInfo.from_db(app)


async def list_user_oauth_applications(user_id: str) -> list[OAuthApplicationInfo]:
    """Get all OAuth applications owned by a user"""
    apps = await PrismaOAuthApplication.prisma().find_many(
        where={"ownerId": user_id},
        order={"createdAt": "desc"},
    )
    return [OAuthApplicationInfo.from_db(app) for app in apps]


async def update_oauth_application(
    app_id: str,
    *,
    owner_id: str,
    is_active: Optional[bool] = None,
    logo_url: Optional[str] = None,
) -> Optional[OAuthApplicationInfo]:
    """
    Update OAuth application active status.
    Only the owner can update their app's status.

    Returns the updated app info, or None if app not found or not owned by user.
    """
    # First verify ownership
    app = await PrismaOAuthApplication.prisma().find_first(
        where={"id": app_id, "ownerId": owner_id}
    )
    if not app:
        return None

    patch: OAuthApplicationUpdateInput = {}
    if is_active is not None:
        patch["isActive"] = is_active
    if logo_url:
        patch["logoUrl"] = logo_url
    if not patch:
        return OAuthApplicationInfo.from_db(app)  # return unchanged

    updated_app = await PrismaOAuthApplication.prisma().update(
        where={"id": app_id},
        data=patch,
    )
    return OAuthApplicationInfo.from_db(updated_app) if updated_app else None


# ============================================================================
# Token Cleanup
# ============================================================================


async def cleanup_expired_oauth_tokens() -> dict[str, int]:
    """
    Delete expired OAuth tokens from the database.

    This removes:
    - Expired authorization codes (10 min TTL)
    - Expired access tokens (1 hour TTL)
    - Expired refresh tokens (30 day TTL)

    Returns a dict with counts of deleted tokens by type.
    """
    now = datetime.now(timezone.utc)

    # Delete expired authorization codes
    codes_result = await PrismaOAuthAuthorizationCode.prisma().delete_many(
        where={"expiresAt": {"lt": now}}
    )

    # Delete expired access tokens
    access_result = await PrismaOAuthAccessToken.prisma().delete_many(
        where={"expiresAt": {"lt": now}}
    )

    # Delete expired refresh tokens
    refresh_result = await PrismaOAuthRefreshToken.prisma().delete_many(
        where={"expiresAt": {"lt": now}}
    )

    deleted = {
        "authorization_codes": codes_result,
        "access_tokens": access_result,
        "refresh_tokens": refresh_result,
    }

    total = sum(deleted.values())
    if total > 0:
        logger.info(f"Cleaned up {total} expired OAuth tokens: {deleted}")

    return deleted
