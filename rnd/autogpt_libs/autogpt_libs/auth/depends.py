import fastapi

from .middleware import auth_middleware
from .models import User


def requires_user(payload: dict = fastapi.Depends(auth_middleware)) -> User:
    return verify_user(payload, admin_only=False)


def requires_admin_user(
    payload: dict = fastapi.Depends(auth_middleware),
) -> User:
    return verify_user(payload, admin_only=True)


def verify_user(payload: dict | None, admin_only: bool) -> User:
    if not payload:
        # This handles the case when authentication is disabled
        payload = {"sub": "3e53486c-cf57-477e-ba2a-cb02dc828e1a", "role": "admin"}

    user_id = payload.get("sub")

    if not user_id:
        raise fastapi.HTTPException(
            status_code=401, detail="User ID not found in token"
        )

    if admin_only and payload["role"] != "admin":
        raise fastapi.HTTPException(status_code=403, detail="Admin access required")

    return User.from_payload(payload)
