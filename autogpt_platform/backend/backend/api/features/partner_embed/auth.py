"""Authentication boundary for partner-embedded API routes."""

from typing import Annotated, Literal

import fastapi
import jwt
from autogpt_libs.auth.jwt_utils import bearer_jwt_auth, parse_jwt_token_async
from fastapi import Security
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, ValidationError

EMBED_TOKEN_AUDIENCE = "autogpt-partner-embed"


class EmbedPrincipal(BaseModel):
    """Server-derived identity and tenancy for one embedded request."""

    user_id: str
    partner_id: str
    organization_id: str
    team_id: str | None
    external_account_id: str
    scopes: list[str]
    capabilities: list[str] = Field(default_factory=list)


class _EmbedTokenClaims(BaseModel):
    sub: str = Field(min_length=1)
    token_use: Literal["partner_embed"]
    partner_id: str = Field(min_length=1)
    organization_id: str = Field(min_length=1)
    team_id: str | None = None
    external_account_id: str = Field(min_length=1)
    scope: str = ""
    capabilities: list[str] = Field(default_factory=list)


async def requires_embed_principal(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Security(bearer_jwt_auth),
    ],
) -> EmbedPrincipal:
    """Validate an embed-only JWT and return its locked tenant context."""
    if credentials is None:
        raise fastapi.HTTPException(
            status_code=401,
            detail="Authorization header is missing",
        )

    token = credentials.credentials
    try:
        header = jwt.get_unverified_header(token)
    except jwt.InvalidTokenError as exc:
        raise fastapi.HTTPException(
            status_code=401,
            detail=f"Invalid token: {exc}",
        ) from exc

    if str(header.get("alg", "")).startswith("HS"):
        raise fastapi.HTTPException(
            status_code=401,
            detail="Invalid token: symmetrically signed embed tokens are not accepted",
        )

    try:
        payload = await parse_jwt_token_async(token, audience=EMBED_TOKEN_AUDIENCE)
        claims = _EmbedTokenClaims.model_validate(payload)
    except (ValueError, ValidationError) as exc:
        raise fastapi.HTTPException(
            status_code=401, detail="Invalid embed token"
        ) from exc

    return EmbedPrincipal(
        user_id=claims.sub,
        partner_id=claims.partner_id,
        organization_id=claims.organization_id,
        team_id=claims.team_id,
        external_account_id=claims.external_account_id,
        scopes=claims.scope.split(),
        capabilities=sorted(set(claims.capabilities)),
    )


def require_embed_scope(scope: str):
    """Build a dependency that requires one scope on an embed principal."""

    async def _dependency(
        principal: Annotated[EmbedPrincipal, Security(requires_embed_principal)],
    ) -> EmbedPrincipal:
        if scope not in principal.scopes:
            raise fastapi.HTTPException(
                status_code=403,
                detail=f"Missing embed scope: {scope}",
            )
        return principal

    return _dependency
