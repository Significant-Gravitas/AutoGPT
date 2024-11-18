from autogpt_libs.auth.depends import requires_user
from autogpt_libs.auth.models import User
from fastapi import Depends
from fastapi.requests import Request

from backend.util.settings import Settings

settings = Settings()


def get_user_id(request: Request, user: User = Depends(requires_user)) -> str:
    if user.role == "admin" and (
        logged_as := request.headers.get("X-LoggedIn-As-User")
    ):
        return logged_as
    return user.user_id
