# Analytics API

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
    step: Annotated[str, fastapi.Body(..., embed=True)],
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
    await prisma.models.AnalyticsMetrics.prisma().upsert(
        data={
            "update": {"value": {"increment": 1}},
            "create": {
                "value": 1,
                "analyticMetric": prisma.enums.AnalyticsMetric.TUTORIAL_STEP_COMPLETION,
                "userId": user_id,
                "dataString": step,
                "aggregationType": prisma.enums.AggregationType.COUNT,
            },
        },
        where={
            "analyticMetric_userId_dataString_aggregationType": {
                "analyticMetric": prisma.enums.AnalyticsMetric.TUTORIAL_STEP_COMPLETION,
                "userId": user_id,
                "dataString": step,
                "aggregationType": prisma.enums.AggregationType.COUNT,
            }
        },
    )
    return id.id


class PageViewData(pydantic.BaseModel):
    page: str
    data: Optional[dict]


@router.post(path="/log_page_view")
async def log_page_view(
    user_id: Annotated[str, fastapi.Depends(get_user_id)],
    page_view_data: Annotated[PageViewData, fastapi.Body(..., embed=True)],
):
    """
    Log the page view for analytics purposes.
    """
    id = await prisma.models.AnalyticsDetails.prisma().create(
        data={
            "userId": user_id,
            "type": prisma.enums.AnalyticsType.WEB_PAGE,
            "dataIndex": page_view_data.page,
            "data": prisma.Json(page_view_data.data),
        }
    )
    await prisma.models.AnalyticsMetrics.prisma().upsert(
        data={
            "update": {"value": {"increment": 1}},
            "create": {
                "value": 1,
                "analyticMetric": prisma.enums.AnalyticsMetric.PAGE_VIEW,
                "userId": user_id,
                "dataString": page_view_data.page,
                "aggregationType": prisma.enums.AggregationType.COUNT,
            },
        },
        where={
            "analyticMetric_userId_dataString_aggregationType": {
                "analyticMetric": prisma.enums.AnalyticsMetric.PAGE_VIEW,
                "userId": user_id,
                "dataString": page_view_data.page,
                "aggregationType": prisma.enums.AggregationType.COUNT,
            }
        },
    )
    return id.id


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
    if metric_value < 0:
        raise ValueError("metric_value must be non-negative")

    if aggregation_type == prisma.enums.AggregationType.NO_AGGREGATION:
        value_increment = metric_value
        counter_increment = 0
    elif aggregation_type in [
        prisma.enums.AggregationType.COUNT,
        prisma.enums.AggregationType.SUM,
    ]:
        value_increment = metric_value
        counter_increment = 1
    elif aggregation_type in [
        prisma.enums.AggregationType.AVG,
        prisma.enums.AggregationType.MAX,
        prisma.enums.AggregationType.MIN,
    ]:
        value_increment = 0  # These will be handled differently in a separate query
        counter_increment = 1
    else:
        raise ValueError(f"Unsupported aggregation_type: {aggregation_type}")

    await prisma.models.AnalyticsMetrics.prisma().upsert(
        data={
            "update": {
                "value": {"increment": value_increment},
                "aggregationCounter": {"increment": counter_increment},
            },
            "create": {
                "value": metric_value,
                "analyticMetric": metric_name,
                "userId": user_id,
                "dataString": data_string,
                "aggregationType": aggregation_type,
                "aggregationCounter": 1,
            },
        },
        where={
            "analyticMetric_userId_dataString_aggregationType": {
                "analyticMetric": metric_name,
                "userId": user_id,
                "dataString": data_string,
                "aggregationType": aggregation_type,
            }
        },
    )

    # For AVG, MAX, and MIN, we need to perform additional operations
    if aggregation_type in [
        prisma.enums.AggregationType.AVG,
        prisma.enums.AggregationType.MAX,
        prisma.enums.AggregationType.MIN,
    ]:
        existing = await prisma.models.AnalyticsMetrics.prisma().find_unique(
            where={
                "analyticMetric_userId_dataString_aggregationType": {
                    "analyticMetric": metric_name,
                    "userId": user_id,
                    "dataString": data_string,
                    "aggregationType": aggregation_type,
                }
            }
        )
        if existing:
            if aggregation_type == prisma.enums.AggregationType.AVG:
                new_value = (
                    existing.value * existing.aggregationCounter + metric_value
                ) / (existing.aggregationCounter + 1)
            elif aggregation_type == prisma.enums.AggregationType.MAX:
                new_value = max(existing.value, metric_value)
            else:  # MIN
                new_value = min(existing.value, metric_value)

            await prisma.models.AnalyticsMetrics.prisma().update(
                data={"value": new_value}, where={"id": existing.id}
            )
