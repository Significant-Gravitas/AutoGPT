import asyncio
import logging
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from backend.data.db_accessors import auth_session_db

logger = logging.getLogger(__name__)
_SESSION_LOOKUP_TIMEOUT = 5.0


class SessionTokenClaims(BaseModel):
    model_config = ConfigDict(strict=True)

    sub: str = Field(min_length=1, max_length=128)
    sid: str = Field(min_length=1, max_length=128)


async def require_active_auth_session(payload: dict[str, Any]) -> None:
    if "sid" not in payload:
        return
    try:
        claims = SessionTokenClaims.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=401, detail="Invalid session token") from exc
    try:
        active = await asyncio.wait_for(
            auth_session_db().is_auth_session_active(claims.sid, claims.sub),
            timeout=_SESSION_LOOKUP_TIMEOUT,
        )
    except Exception as exc:
        logger.warning("Auth session lookup unavailable (%s)", type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail="Authentication service unavailable, retry shortly",
            headers={"Retry-After": "5"},
        ) from exc
    if not active:
        raise HTTPException(status_code=401, detail="Session expired or revoked")
