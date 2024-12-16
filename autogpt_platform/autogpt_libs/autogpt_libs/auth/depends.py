import fastapi

from .config import Settings
from .middleware import auth_middleware
from .models import DEFAULT_USER_ID, User


def requires_user(payload: dict = fastapi.Depends(auth_middleware)) -> User:
    return verify_user(payload, admin_only=False)


def requires_admin_user(
    payload: dict = fastapi.Depends(auth_middleware),
) -> User:
    return verify_user(payload, admin_only=True)


def verify_user(payload: dict | None, admin_only: bool) -> User:
    if not payload:
        if Settings.ENABLE_AUTH:
            raise fastapi.HTTPException(
                status_code=401, detail="Authorization header is missing"
            )
        # This handles the case when authentication is disabled
        payload = {"sub": DEFAULT_USER_ID, "role": "admin"}

    user_id = payload.get("sub")

    if not user_id:
        raise fastapi.HTTPException(
            status_code=401, detail="User ID not found in token"
        )

    if admin_only and payload["role"] != "admin":
        raise fastapi.HTTPException(status_code=403, detail="Admin access required")

    return User.from_payload(payload)


def get_user_id(payload: dict = fastapi.Depends(auth_middleware)) -> str:
    user_id = payload.get("sub")
    if not user_id:
        raise fastapi.HTTPException(
            status_code=401, detail="User ID not found in token"
        )
    return user_id
