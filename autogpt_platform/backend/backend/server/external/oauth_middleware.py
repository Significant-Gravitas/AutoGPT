"""
OAuth Access Token middleware for external API.

Validates OAuth access tokens and provides user/client context
for external API endpoints that use OAuth authentication.
"""

from datetime import datetime, timezone
from typing import Optional

import jwt
from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from backend.data.db import prisma
from backend.server.oauth.token_service import get_token_service


class OAuthTokenInfo(BaseModel):
    """Information extracted from a validated OAuth access token."""

    user_id: str
    client_id: str
    scopes: list[str]
    token_id: str


# HTTP Bearer token extractor
oauth_bearer = HTTPBearer(auto_error=False)


async def require_oauth_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(oauth_bearer),
) -> OAuthTokenInfo:
    """
    Validate an OAuth access token and return token info.

    Extracts the Bearer token from the Authorization header,
    validates the JWT signature and claims, and checks that
    the token hasn't been revoked.

    Raises:
        HTTPException: 401 if token is missing, invalid, or revoked
    """
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Missing authorization token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    token_service = get_token_service()

    try:
        # Verify JWT signature and claims
        claims = token_service.verify_access_token(token)

        # Check if token is in database and not revoked
        token_hash = token_service.hash_token(token)
        stored_token = await prisma.oauthaccesstoken.find_unique(
            where={"tokenHash": token_hash}
        )

        if not stored_token:
            raise HTTPException(
                status_code=401,
                detail="Token not found",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if stored_token.revokedAt:
            raise HTTPException(
                status_code=401,
                detail="Token has been revoked",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if stored_token.expiresAt < datetime.now(timezone.utc):
            raise HTTPException(
                status_code=401,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Update last used timestamp (fire and forget)
        await prisma.oauthaccesstoken.update(
            where={"id": stored_token.id},
            data={"lastUsedAt": datetime.now(timezone.utc)},
        )

        return OAuthTokenInfo(
            user_id=claims.sub,
            client_id=claims.client_id,
            scopes=claims.scope.split() if claims.scope else [],
            token_id=stored_token.id,
        )

    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=401,
            detail=f"Invalid token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_scope(required_scope: str):
    """
    Dependency that validates OAuth token and checks for required scope.

    Args:
        required_scope: The scope required for this endpoint

    Returns:
        Dependency function that returns OAuthTokenInfo if authorized
    """

    async def check_scope(
        token: OAuthTokenInfo = Security(require_oauth_token),
    ) -> OAuthTokenInfo:
        if required_scope not in token.scopes:
            raise HTTPException(
                status_code=403,
                detail=f"Token lacks required scope '{required_scope}'",
                headers={"WWW-Authenticate": f'Bearer scope="{required_scope}"'},
            )
        return token

    return check_scope


def require_any_scope(*required_scopes: str):
    """
    Dependency that validates OAuth token and checks for any of the required scopes.

    Args:
        required_scopes: At least one of these scopes is required

    Returns:
        Dependency function that returns OAuthTokenInfo if authorized
    """

    async def check_scopes(
        token: OAuthTokenInfo = Security(require_oauth_token),
    ) -> OAuthTokenInfo:
        for scope in required_scopes:
            if scope in token.scopes:
                return token

        scope_list = " ".join(required_scopes)
        raise HTTPException(
            status_code=403,
            detail=f"Token lacks required scopes (need one of: {scope_list})",
            headers={"WWW-Authenticate": f'Bearer scope="{scope_list}"'},
        )

    return check_scopes
