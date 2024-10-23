from autogpt_libs.auth.middleware import auth_middleware
from fastapi import Depends, HTTPException

from backend.data.user import DEFAULT_USER_ID
from backend.util.settings import Settings

settings = Settings()


def get_user_id(payload: dict = Depends(auth_middleware)) -> str:
    if not payload:
        # This handles the case when authentication is disabled
        return DEFAULT_USER_ID

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")
    return user_id
