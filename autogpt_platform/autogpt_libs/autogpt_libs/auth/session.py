"""Require that a user JWT's session is still signed in."""

import logging

import fastapi

from .closed_accounts import reject_closed_account
from .dependencies import get_unbanned_jwt_payload

logger = logging.getLogger(__name__)

SESSION_REVOKED_DETAIL = {
    "code": "session_revoked",
    "message": "Your session has ended. Please sign in again.",
}

SESSION_CHECK_UNAVAILABLE_DETAIL = {
    "code": "session_check_unavailable",
    "message": "We couldn't confirm your session. Please try again.",
}


async def requires_live_session(
    jwt_payload: dict = fastapi.Security(get_unbanned_jwt_payload),
) -> dict:
    """401 unless the token's ``sid`` session is live; tokens without one pass.

    A closed account is refused first, with the same 401 ``get_user_id`` gives.
    """
    user_id = jwt_payload.get("sub")
    if not user_id:
        raise fastapi.HTTPException(status_code=401, detail=SESSION_REVOKED_DETAIL)
    # Closing an account ends every session, so without this its tokens would
    # read as an ended session here, not as the closed account every other
    # route answers them as.
    await reject_closed_account(user_id)
    session_id = jwt_payload.get("sid")
    if not session_id:
        logger.warning(
            f"Token without a session id passed the session check for {user_id}"
        )
        return jwt_payload
    try:
        live = await _is_session_live(session_id, user_id)
    except Exception:
        logger.exception(f"Session check failed for {user_id}")
        raise fastapi.HTTPException(
            status_code=503, detail=SESSION_CHECK_UNAVAILABLE_DETAIL
        )
    if not live:
        logger.info(f"Refused a token from a session that has ended for {user_id}")
        raise fastapi.HTTPException(status_code=401, detail=SESSION_REVOKED_DETAIL)
    return jwt_payload


async def _is_session_live(session_id: str, user_id: str) -> bool:
    """Raises if the session can't be read, so an outage answers 503, not 401."""
    from backend.data import db  # deferred -- only needed at runtime
    from backend.data.auth.session import is_auth_session_live  # deferred

    if not db.is_connected():
        raise RuntimeError("Database is not connected")
    return await is_auth_session_live(session_id, user_id)
