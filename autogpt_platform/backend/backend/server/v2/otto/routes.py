import logging

from autogpt_libs.auth.middleware import auth_middleware
from fastapi import APIRouter, Depends

from backend.server.utils import get_user_id

from .models import ApiResponse, ChatRequest
from .service import OttoService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/ask", response_model=ApiResponse, dependencies=[Depends(auth_middleware)]
)
async def proxy_otto_request(
    request: ChatRequest, user_id: str = Depends(get_user_id)
) -> ApiResponse:
    """
    Proxy requests to Otto API while adding necessary security headers and logging.
    Requires an authenticated user.
    """
    return await OttoService.ask(request, user_id)
