from autogpt_libs.auth import auth_middleware
from fastapi import Depends, HTTPException

from autogpt_server.data.user import DEFAULT_USER_ID


def get_user_id(payload: dict = Depends(auth_middleware)) -> str:
    if not payload:
        # This handles the case when authentication is disabled
        return DEFAULT_USER_ID

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")
    return user_id
