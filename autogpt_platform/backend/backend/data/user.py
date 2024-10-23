from typing import Optional

from autogpt_libs.supabase_integration_credentials_store.types import UserMetadataRaw
from fastapi import HTTPException
from prisma import Json
from prisma.models import User

from backend.data.db import prisma

DEFAULT_USER_ID = "3e53486c-cf57-477e-ba2a-cb02dc828e1a"
DEFAULT_EMAIL = "default@example.com"


async def get_or_create_user(user_data: dict) -> User:
    user_id = user_data.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")

    user_email = user_data.get("email")
    if not user_email:
        raise HTTPException(status_code=401, detail="Email not found in token")

    user = await prisma.user.find_unique(where={"id": user_id})
    if not user:
        user = await prisma.user.create(
            data={
                "id": user_id,
                "email": user_email,
                "name": user_data.get("user_metadata", {}).get("name"),
            }
        )
    return User.model_validate(user)


async def get_user_by_id(user_id: str) -> Optional[User]:
    user = await prisma.user.find_unique(where={"id": user_id})
    return User.model_validate(user) if user else None


async def create_default_user() -> Optional[User]:
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


async def get_user_metadata(user_id: str) -> UserMetadataRaw:
    user = await User.prisma().find_unique_or_raise(
        where={"id": user_id},
    )
    return (
        UserMetadataRaw.model_validate(user.metadata)
        if user.metadata
        else UserMetadataRaw()
    )


async def update_user_metadata(user_id: str, metadata: UserMetadataRaw):
    await User.prisma().update(
        where={"id": user_id},
        data={"metadata": Json(metadata.model_dump())},
    )
