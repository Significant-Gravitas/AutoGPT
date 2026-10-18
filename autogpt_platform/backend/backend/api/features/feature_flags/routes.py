from typing import Annotated

from autogpt_libs.auth import get_optional_user_id
from fastapi import APIRouter, Query, Security

from backend.util.feature_flag.frontend import (
    FrontendFlagValues,
    evaluate_frontend_flags,
)

router = APIRouter(prefix="/feature-flags", tags=["feature-flags"])


@router.get("", operation_id="get_feature_flags")
async def get_feature_flags(
    user_id: Annotated[str | None, Security(get_optional_user_id)] = None,
    visitor_id: Annotated[
        str | None,
        Query(
            description="A signed-out visitor's analytics id, sent only with their "
            "analytics consent; ignored for a signed-in caller or when malformed."
        ),
    ] = None,
) -> FrontendFlagValues:
    """The frontend's flags for the caller, evaluated as the backend's own reads are."""
    return await evaluate_frontend_flags(user_id, visitor_id)
