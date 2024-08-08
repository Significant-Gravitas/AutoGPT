from typing import Optional

from prisma.models import User

from autogpt_server.data.db import prisma

DEFAULT_USER_ID = "3e53486c-cf57-477e-ba2a-cb02dc828e1a"


async def get_or_create_user(user_data: dict) -> User:
    user = await prisma.user.find_unique(where={"id": user_data["sub"]})
    if not user:
        user = await prisma.user.create(
            data={
                "id": user_data["sub"],
                "email": user_data["email"],
                "name": user_data.get("user_metadata", {}).get("name"),
            }
        )
    return User.model_validate(user)


async def get_user_by_id(user_id: str) -> Optional[User]:
    user = await prisma.user.find_unique(where={"id": user_id})
    return User.model_validate(user) if user else None


async def create_default_user(enable_auth: str) -> Optional[User]:
    if not enable_auth.lower() == "true":
        user = await prisma.user.find_unique(where={"id": DEFAULT_USER_ID})
        if not user:
            user = await prisma.user.create(
                data={
                    "id": DEFAULT_USER_ID,
                    "email": "default@example.com",
                    "name": "Default User",
                }
            )
        return User.model_validate(user)
    return None
