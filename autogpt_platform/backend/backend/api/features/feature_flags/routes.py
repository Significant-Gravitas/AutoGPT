from typing import Annotated

from autogpt_libs.auth import get_optional_user_id
from fastapi import APIRouter, Security

from backend.util.feature_flag.frontend import (
    FrontendFlagValues,
    evaluate_frontend_flags,
)

router = APIRouter(prefix="/feature-flags", tags=["feature-flags"])


@router.get("", operation_id="get_feature_flags")
async def get_feature_flags(
    user_id: Annotated[str | None, Security(get_optional_user_id)] = None,
) -> FrontendFlagValues:
    """The frontend's flags for the caller, evaluated as the backend's own reads are."""
    return await evaluate_frontend_flags(user_id)
