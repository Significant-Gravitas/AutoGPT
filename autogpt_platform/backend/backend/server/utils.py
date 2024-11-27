from autogpt_libs.auth.depends import requires_user
from autogpt_libs.auth.models import User
from fastapi import Depends

from backend.util.settings import Settings

settings = Settings()


def get_user_id(user: User = Depends(requires_user)) -> str:
    return user.user_id
