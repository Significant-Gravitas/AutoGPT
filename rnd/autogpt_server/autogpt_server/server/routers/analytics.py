"""Analytics API"""

from typing import Annotated, Optional

import fastapi
import prisma
import prisma.enums
import pydantic

import autogpt_server.data.analytics
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
    user_data: Annotated[
        UserData, fastapi.Body(..., embed=True, description="The user data to log")
    ],
):
    """
    Log the user ID for analytics purposes.
    """

    result = await autogpt_server.data.analytics.log_raw_analytics(
        user_id,
        prisma.enums.AnalyticsType.CREATE_USER,
        user_data.model_dump(),
        "",
    )
    return result.id


@router.post(path="/log_tutorial_step")
async def log_tutorial_step(
    user_id: Annotated[str, fastapi.Depends(get_user_id)],
    step: Annotated[str, fastapi.Body(..., embed=True)],
    data: Annotated[
        Optional[dict],
        fastapi.Body(..., embed=True, description="Any additional data to log"),
    ],
):
    """
    Log the tutorial step completed by the user for analytics purposes.
    """
    result = await autogpt_server.data.analytics.log_raw_analytics(
        user_id,
        prisma.enums.AnalyticsType.TUTORIAL_STEP,
        data or {},
        step,
    )
    await autogpt_server.data.analytics.log_raw_metric(
        user_id=user_id,
        metric_name=prisma.enums.AnalyticsMetric.TUTORIAL_STEP_COMPLETION,
        aggregation_type=prisma.enums.AggregationType.COUNT,
        metric_value=1,
        data_string=step,
    )
    return result.id


class PageViewData(pydantic.BaseModel):
    page: str = pydantic.Field(description="The page viewed")
    data: Optional[dict] = pydantic.Field(
        default_factory=dict, description="Any additional data to log"
    )


@router.post(path="/log_page_view")
async def log_page_view(
    user_id: Annotated[str, fastapi.Depends(get_user_id)],
    page_view_data: Annotated[PageViewData, fastapi.Body(..., embed=True)],
):
    """
    Log the page view for analytics purposes.
    """
    await autogpt_server.data.analytics.log_raw_metric(
        user_id=user_id,
        metric_name=prisma.enums.AnalyticsMetric.PAGE_VIEW,
        aggregation_type=prisma.enums.AggregationType.COUNT,
        metric_value=1,
        data_string=page_view_data.page,
    )
    result = await autogpt_server.data.analytics.log_raw_analytics(
        user_id=user_id,
        type=prisma.enums.AnalyticsType.WEB_PAGE,
        data=page_view_data.data or {},
        data_index=page_view_data.page,
    )
    return result.id


@router.post(path="/log_raw_metric")
async def log_raw_metric(
    user_id: Annotated[str, fastapi.Depends(get_user_id)],
    metric_name: Annotated[prisma.enums.AnalyticsMetric, fastapi.Body(..., embed=True)],
    aggregation_type: Annotated[
        prisma.enums.AggregationType, fastapi.Body(..., embed=True)
    ],
    metric_value: Annotated[float, fastapi.Body(..., embed=True)],
    data_string: Annotated[str, fastapi.Body(..., embed=True)],
):
    result = await autogpt_server.data.analytics.log_raw_metric(
        user_id, metric_name, aggregation_type, metric_value, data_string
    )
    return result.id


@router.post("/log_raw_analytics")
async def log_raw_analytics(
    user_id: Annotated[str, fastapi.Depends(get_user_id)],
    type: Annotated[prisma.enums.AnalyticsType, fastapi.Body(..., embed=True)],
    data: Annotated[
        dict,
        fastapi.Body(..., embed=True, description="The data to log"),
    ],
    data_index: Annotated[
        str,
        fastapi.Body(
            ...,
            embed=True,
            description="Indexable field for any count based analytical measures like page order clicking, tutorial step completion, etc.",
        ),
    ],
):
    result = await autogpt_server.data.analytics.log_raw_analytics(
        user_id, type, data, data_index
    )
    return result.id
