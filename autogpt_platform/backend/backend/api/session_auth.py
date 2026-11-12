import asyncio
import logging
from time import time
from typing import Any

from autogpt_libs.auth.closed_accounts import reject_closed_account
from autogpt_libs.auth.dependencies import ACCOUNT_BANNED_DETAIL
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from backend.data import user as user_store
from backend.data.db_accessors import auth_session_db, user_db
from backend.util.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()
_SESSION_LOOKUP_TIMEOUT = 5.0
SESSION_TOKEN_RENEWAL_REQUIRED = "session_token_renewal_required"
# The frontend signs out on this code (lib/auth/account-banned.ts).
SESSION_REVOKED_DETAIL = {
    "code": "session_revoked",
    "message": "Session expired or revoked",
}


class SessionTokenClaims(BaseModel):
    model_config = ConfigDict(strict=True)

    sub: str = Field(min_length=1, max_length=128)
    sid: str = Field(min_length=1, max_length=128)


async def require_active_auth_session(payload: dict[str, Any]) -> None:
    if "sid" not in payload:
        deadline = settings.config.auth_legacy_token_deadline
        if deadline is None or time() < deadline:
            return
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
        # Closing an account ends every session, so its tokens land here: they
        # get the closed account's 401, as the WebSocket and routes answer it.
        await reject_closed_account(claims.sub)
        raise HTTPException(status_code=401, detail=SESSION_REVOKED_DETAIL)


async def _session_is_active(claims: SessionTokenClaims) -> bool:
    if await user_store.is_user_banned(claims.sub):
        raise HTTPException(status_code=403, detail=ACCOUNT_BANNED_DETAIL)
    active = await auth_session_db().is_auth_session_active(claims.sid, claims.sub)
    if not active and await user_db().is_auth_user_banned(claims.sub):
        raise HTTPException(status_code=403, detail=ACCOUNT_BANNED_DETAIL)
    return active
