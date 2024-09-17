"""Analytics API"""

from typing import Annotated

import fastapi
import prisma
import prisma.enums

import autogpt_server.data.analytics
from autogpt_server.server.utils import get_user_id

router = fastapi.APIRouter()


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
