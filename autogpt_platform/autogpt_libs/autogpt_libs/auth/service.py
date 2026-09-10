"""
Service-to-service authentication for trusted platform components.

The Next.js frontend signs short-lived JWTs with its Better Auth JWKS key --
the same key the backend already trusts for user tokens (`JWT_JWKS_URL`) --
but with a distinct audience and subject, so user tokens and service tokens
can never be replayed against each other's endpoints. This authenticates
pre-login frontend->backend calls (e.g. auth emails), where no user session
exists to vouch for the request, without introducing any new shared secret.
"""

import logging

import fastapi
import jwt
from fastapi.security import HTTPAuthorizationCredentials

from .config import get_settings
from .jwt_utils import bearer_jwt_auth, parse_jwt_token_async

logger = logging.getLogger(__name__)

SERVICE_TOKEN_AUDIENCE = "autogpt-platform-backend"
FRONTEND_SERVICE_SUBJECT = "service:frontend"


def requires_frontend_service(scope: str):
    """Factory returning a FastAPI dependency that authenticates the frontend
    service itself (not a user) for the given scope.

    Example::

        @router.post(
            "/send",
            dependencies=[Security(requires_frontend_service("auth-email:send"))],
        )
    """

    async def _dependency(
        credentials: HTTPAuthorizationCredentials | None = fastapi.Security(
            bearer_jwt_auth
        ),
    ) -> None:
        if not get_settings().JWT_JWKS_URL:
            raise fastapi.HTTPException(
                status_code=503,
                detail="Service authentication requires JWT_JWKS_URL to be set.",
            )
        if not credentials:
            raise fastapi.HTTPException(
                status_code=401, detail="Authorization header is missing"
            )
        token = credentials.credentials

        try:
            header = jwt.get_unverified_header(token)
        except jwt.InvalidTokenError as e:
            raise fastapi.HTTPException(
                status_code=401, detail=f"Invalid token: {e}"
            ) from e
        # Service tokens are only ever JWKS-signed; the legacy HS256 shared
        # secret must not be able to mint one.
        if header.get("alg", "").startswith("HS"):
            raise fastapi.HTTPException(
                status_code=401,
                detail=(
                    "Invalid token: symmetrically signed service tokens "
                    "are not accepted"
                ),
            )

        try:
            payload = await parse_jwt_token_async(
                token, audience=SERVICE_TOKEN_AUDIENCE
            )
        except ValueError as e:
            raise fastapi.HTTPException(status_code=401, detail=str(e)) from e

        if payload.get("sub") != FRONTEND_SERVICE_SUBJECT:
            raise fastapi.HTTPException(
                status_code=401, detail="Not a frontend service token"
            )
        granted_scopes = str(payload.get("scope", "")).split()
        if scope not in granted_scopes:
            raise fastapi.HTTPException(
                status_code=403, detail=f"Missing service scope: {scope}"
            )

    return _dependency
