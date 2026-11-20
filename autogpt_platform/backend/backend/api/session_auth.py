import asyncio
import logging
from typing import Any

from autogpt_libs.auth.dependencies import ACCOUNT_BANNED_DETAIL
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from backend.data import user as user_store
from backend.data.db_accessors import auth_session_db, user_db

logger = logging.getLogger(__name__)
_SESSION_LOOKUP_TIMEOUT = 5.0
SESSION_TOKEN_RENEWAL_REQUIRED = "session_token_renewal_required"


class SessionTokenClaims(BaseModel):
    model_config = ConfigDict(strict=True)

    sub: str = Field(min_length=1, max_length=128)
    sid: str = Field(min_length=1, max_length=128)


async def require_active_auth_session(payload: dict[str, Any]) -> None:
    if "sid" not in payload:
        # The auth service puts the session id in every token it signs, so a
        # token without one predates that: the client renews it.
        raise HTTPException(
            status_code=401,
            detail="Session token requires renewal",
            headers={"X-Error-Code": SESSION_TOKEN_RENEWAL_REQUIRED},
        )
    try:
        claims = SessionTokenClaims.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=401, detail="Invalid session token") from exc
    try:
        active = await asyncio.wait_for(
            _session_is_active(claims),
            timeout=_SESSION_LOOKUP_TIMEOUT,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning(f"Auth session lookup unavailable ({type(exc).__name__})")
        raise HTTPException(
            status_code=503,
            detail="Authentication service unavailable, retry shortly",
            headers={"Retry-After": "5"},
        ) from exc
    if not active:
        raise HTTPException(status_code=401, detail="Session expired or revoked")


async def _session_is_active(claims: SessionTokenClaims) -> bool:
    if await user_store.is_user_banned(claims.sub):
        raise HTTPException(status_code=403, detail=ACCOUNT_BANNED_DETAIL)
    active = await auth_session_db().is_auth_session_active(claims.sid, claims.sub)
    if not active and await user_db().is_auth_user_banned(claims.sub):
        raise HTTPException(status_code=403, detail=ACCOUNT_BANNED_DETAIL)
    return active
