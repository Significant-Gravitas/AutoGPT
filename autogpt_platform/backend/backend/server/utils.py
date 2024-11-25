from autogpt_libs.auth.depends import requires_user
from autogpt_libs.auth.models import User
from autogpt_libs.utils.settings import Settings
from fastapi import Depends

settings = Settings()


def get_user_id(user: User = Depends(requires_user)) -> str:
    return user.user_id
