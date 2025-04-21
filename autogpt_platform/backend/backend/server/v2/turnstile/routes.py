import logging

from fastapi import APIRouter

from .models import TurnstileVerifyRequest, TurnstileVerifyResponse
from .service import TurnstileService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/verify", response_model=TurnstileVerifyResponse)
async def verify_turnstile_token(
    request: TurnstileVerifyRequest,
) -> TurnstileVerifyResponse:
    """
    Verify a Cloudflare Turnstile token.
    This endpoint verifies a token returned by the Cloudflare Turnstile challenge
    on the client side. It returns whether the verification was successful.
    """
    logger.info(f"Verifying Turnstile token for action: {request.action}")
    return await TurnstileService.verify_token(request)
