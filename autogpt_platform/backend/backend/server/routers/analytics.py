"""Analytics API"""

import logging
from typing import Annotated

import fastapi

import backend.data.analytics
from backend.server.utils import get_user_id

router = fastapi.APIRouter()
logger = logging.getLogger(__name__)


@router.post(path="/log_raw_metric")
async def log_raw_metric(
    user_id: Annotated[str, fastapi.Depends(get_user_id)],
    metric_name: Annotated[str, fastapi.Body(..., embed=True)],
    metric_value: Annotated[float, fastapi.Body(..., embed=True)],
    data_string: Annotated[str, fastapi.Body(..., embed=True)],
):
    try:
        result = await backend.data.analytics.log_raw_metric(
            user_id=user_id,
            metric_name=metric_name,
            metric_value=metric_value,
            data_string=data_string,
        )
        return result.id
    except Exception as e:
        logger.exception(
            "Failed to log metric %s for user %s: %s", metric_name, user_id, e
        )
        raise fastapi.HTTPException(
            status_code=500,
            detail={
                "message": str(e),
                "hint": "Check analytics service connection and retry.",
            },
        )


@router.post("/log_raw_analytics")
async def log_raw_analytics(
    user_id: Annotated[str, fastapi.Depends(get_user_id)],
    type: Annotated[str, fastapi.Body(..., embed=True)],
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
    try:
        result = await backend.data.analytics.log_raw_analytics(
            user_id, type, data, data_index
        )
        return result.id
    except Exception as e:
        logger.exception("Failed to log analytics for user %s: %s", user_id, e)
        raise fastapi.HTTPException(
            status_code=500,
            detail={"message": str(e), "hint": "Ensure analytics DB is reachable."},
        )
