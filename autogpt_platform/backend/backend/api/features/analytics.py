"""Analytics API"""

import logging
from typing import Annotated

import fastapi
import pydantic
from autogpt_libs.auth import get_user_id
from autogpt_libs.auth.dependencies import requires_user

import backend.data.analytics
from backend.data import attribution as attribution_db
from backend.data.attribution import UserAttribution, UserAttributionInput

router = fastapi.APIRouter(dependencies=[fastapi.Security(requires_user)])
logger = logging.getLogger(__name__)


@router.post(
    path="/attribution",
    summary="Report user attribution",
    response_model=UserAttribution,
)
async def report_user_attribution(
    user_id: Annotated[str, fastapi.Security(get_user_id)],
    request: UserAttributionInput,
    x_datafast_visitor_id: Annotated[
        str | None, fastapi.Header(include_in_schema=False)
    ] = None,
    x_datafast_session_id: Annotated[
        str | None, fastapi.Header(include_in_schema=False)
    ] = None,
) -> UserAttribution:
    """Store where this user came from. Fields are only ever filled once.

    The DataFast ids come from the request headers the frontend attaches to
    every call, so the body never has to carry them.
    """
    data = request.model_copy(
        update={
            "datafast_visitor_id": request.datafast_visitor_id or x_datafast_visitor_id,
            "datafast_session_id": request.datafast_session_id or x_datafast_session_id,
        }
    )
    return await attribution_db.record_user_attribution(user_id, data)


class LogRawMetricRequest(pydantic.BaseModel):
    metric_name: str = pydantic.Field(..., min_length=1)
    metric_value: float = pydantic.Field(..., allow_inf_nan=False)
    data_string: str = pydantic.Field(..., min_length=1)


@router.post(path="/log_raw_metric")
async def log_raw_metric(
    user_id: Annotated[str, fastapi.Security(get_user_id)],
    request: LogRawMetricRequest,
):
    try:
        result = await backend.data.analytics.log_raw_metric(
            user_id=user_id,
            metric_name=request.metric_name,
            metric_value=request.metric_value,
            data_string=request.data_string,
        )
        return result.id
    except Exception as e:
        logger.exception(
            "Failed to log metric %s for user %s: %s", request.metric_name, user_id, e
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
    user_id: Annotated[str, fastapi.Security(get_user_id)],
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
