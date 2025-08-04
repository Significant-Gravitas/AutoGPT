import logging

from autogpt_libs.auth.middleware import auth_middleware
from fastapi import APIRouter, Depends, HTTPException

from backend.server.utils import get_user_id

from .models import ApiResponse, ChatRequest
from .service import OttoService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/ask",
    response_model=ApiResponse,
    dependencies=[Depends(auth_middleware)],
    summary="Proxy Otto Chat Request",
)
async def proxy_otto_request(
    request: ChatRequest, user_id: str = Depends(get_user_id)
) -> ApiResponse:
    """
    Proxy requests to Otto API while adding necessary security headers and logging.
    Requires an authenticated user.
    """
    logger.debug("Forwarding request to Otto for user %s", user_id)
    try:
        return await OttoService.ask(request, user_id)
    except Exception as e:
        logger.exception("Otto request failed for user %s: %s", user_id, e)
        raise HTTPException(
            status_code=502,
            detail={"message": str(e), "hint": "Check Otto service status."},
        )
