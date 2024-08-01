from typing import Optional

from prisma.models import User

from autogpt_server.data.db import prisma


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
