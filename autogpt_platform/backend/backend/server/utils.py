from autogpt_libs.auth.depends import requires_user
from autogpt_libs.auth.models import User
from fastapi import Depends
from fastapi.requests import Request

from backend.util.settings import Settings

settings = Settings()


def get_user_id(request: Request, user: User = Depends(requires_user)) -> str:
    if user.role == "admin" and (
        login_user_id := request.headers.get("X-LoggedIn-As-User")
    ):
        return login_user_id
    return user.user_id
