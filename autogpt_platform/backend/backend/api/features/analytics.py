"""Analytics API"""

from typing import Annotated

import fastapi
import pydantic
from autogpt_libs.auth import get_user_id
from autogpt_libs.auth.dependencies import requires_user

from backend.data import attribution as attribution_db
from backend.data.attribution import UserAttribution, UserAttributionInput

router = fastapi.APIRouter(dependencies=[fastapi.Security(requires_user)])


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
    merged = {
        **request.model_dump(),
        "datafast_visitor_id": request.datafast_visitor_id or x_datafast_visitor_id,
        "datafast_session_id": request.datafast_session_id or x_datafast_session_id,
    }
    try:
        # Re-validate so header-supplied ids meet the same length limits as
        # the body; model_copy(update=...) would skip that check.
        data = UserAttributionInput.model_validate(merged)
    except pydantic.ValidationError as error:
        raise fastapi.HTTPException(status_code=422, detail=error.errors()) from error
    return await attribution_db.record_user_attribution(user_id, data)
