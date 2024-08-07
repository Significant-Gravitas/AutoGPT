import functools
import logging
import typing

import fastapi

from .middleware import auth_middleware

logger = logging.getLogger("uvicorn.error")


def requires_user(payload: dict = fastapi.Depends(auth_middleware)) -> dict[str, str]:
    return verify_user(payload, admin_only=False)


def requires_admin_user(
    payload: dict = fastapi.Depends(auth_middleware),
) -> dict[str, str]:
    return verify_user(payload, admin_only=True)


def verify_user(payload: dict, admin_only: bool) -> dict[str, str]:
    if not payload:
        # This handles the case when authentication is disabled
        user_id = "3e53486c-cf57-477e-ba2a-cb02dc828e1a"

    user_id = payload.get("sub")
    logger.info(payload["role"])

    if not user_id:
        raise fastapi.HTTPException(
            status_code=401, detail="User ID not found in token"
        )

    if admin_only and payload["role"] != "admin":
        raise fastapi.HTTPException(status_code=403, detail="Admin access required")

    return {"user_id": user_id, "role": payload["role"]}
