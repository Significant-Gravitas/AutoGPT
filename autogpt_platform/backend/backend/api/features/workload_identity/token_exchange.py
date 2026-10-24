"""The token-exchange grant of ``POST /api/oauth/token`` (RFC 8693).

A workload swaps the OIDC token its platform issued (GitHub Actions, Vercel,
GitLab, Google Cloud, ...) for a short-lived AutoGPT access token, acting as
the workload identity named by ``client_id``.
"""

from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import HTTPException, status
from pydantic import BaseModel, Field

from backend.data.auth.workload_exchange import (
    TokenExchangeError,
    exchange_for_workload_token,
)

TOKEN_EXCHANGE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:token-exchange"
ACCESS_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:access_token"


class TokenRequestByTokenExchange(BaseModel):
    grant_type: Literal["urn:ietf:params:oauth:grant-type:token-exchange"]
    subject_token: str = Field(
        max_length=16384, description="OIDC token the workload's platform issued"
    )
    subject_token_type: Literal[
        "urn:ietf:params:oauth:token-type:id_token",
        "urn:ietf:params:oauth:token-type:jwt",
    ]
    client_id: str = Field(description="ID of the workload identity to act as")
    scope: Optional[str] = Field(
        None,
        max_length=1024,
        description="Space-separated subset of the identity's permissions "
        "(all of them if omitted)",
    )


class TokenExchangeResponse(BaseModel):
    access_token: str
    issued_token_type: Literal["urn:ietf:params:oauth:token-type:access_token"] = (
        ACCESS_TOKEN_TYPE
    )
    token_type: Literal["Bearer"] = "Bearer"
    expires_in: int
    access_token_expires_at: datetime
    scope: str


async def exchange_token(request: TokenRequestByTokenExchange) -> TokenExchangeResponse:
    try:
        token, plaintext = await exchange_for_workload_token(
            request.client_id,
            request.subject_token,
            request.scope.split() if request.scope else None,
        )
    except TokenExchangeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": e.error, "error_description": str(e)},
        ) from e
    expires_at = token.expires_at or datetime.now(timezone.utc)
    return TokenExchangeResponse(
        access_token=plaintext,
        expires_in=max(
            0, int((expires_at - datetime.now(timezone.utc)).total_seconds())
        ),
        access_token_expires_at=expires_at,
        scope=" ".join(s.value for s in token.scopes),
    )
