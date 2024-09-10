# Analytics API

import typing
from typing import Annotated, Optional

import fastapi
import prisma
import prisma.enums
import pydantic

from autogpt_server.server.utils import get_user_id

router = fastapi.APIRouter()


class UserData(pydantic.BaseModel):
    user_id: str
    email: str
    name: str
    username: str


@router.post(path="/log_new_user")
async def log_create_user(
    user_id: Annotated[str, fastapi.Depends(get_user_id)],
    user_data: Annotated[UserData, fastapi.Body(..., embed=True)],
):
    """
    Log the user ID for analytics purposes.
    """
    id = await prisma.models.AnalyticsDetails.prisma().create(
        data={
            "userId": user_id,
            "type": prisma.enums.AnalyticsType.CREATE_USER,
            "data": prisma.Json(user_data.model_dump_json()),
        }
    )
    return id.id


@router.post(path="/log_tutorial_step")
async def log_tutorial_step(
    user_id: Annotated[str, fastapi.Depends(get_user_id)],
    step: Annotated[int, fastapi.Body(..., embed=True)],
    data: Annotated[Optional[dict], fastapi.Body(..., embed=True)],
):
    """
    Log the tutorial step completed by the user for analytics purposes.
    """
    id = await prisma.models.AnalyticsDetails.prisma().create(
        data={
            "userId": user_id,
            "type": prisma.enums.AnalyticsType.TUTORIAL_STEP,
            "data": prisma.Json(data),
            "dataIndex": step,
        }
    )
    return id.id
