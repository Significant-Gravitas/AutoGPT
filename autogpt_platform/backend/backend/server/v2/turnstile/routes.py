import logging

import aiohttp
from fastapi import APIRouter

from backend.util.settings import Settings

from .models import TurnstileVerifyRequest, TurnstileVerifyResponse

logger = logging.getLogger(__name__)

router = APIRouter()
settings = Settings()


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
    return await verify_token(request)


async def verify_token(request: TurnstileVerifyRequest) -> TurnstileVerifyResponse:
    """
    Verify a Cloudflare Turnstile token by making a request to the Cloudflare API.
    """
    # Get the secret key from settings
    turnstile_secret_key = settings.secrets.turnstile_secret_key
    turnstile_verify_url = settings.secrets.turnstile_verify_url

    if not turnstile_secret_key:
        logger.error("Turnstile secret key is not configured")
        return TurnstileVerifyResponse(
            success=False,
            error="CONFIGURATION_ERROR",
            challenge_timestamp=None,
            hostname=None,
            action=None,
        )

    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "secret": turnstile_secret_key,
                "response": request.token,
            }

            if request.action:
                payload["action"] = request.action

            logger.debug(f"Verifying Turnstile token with action: {request.action}")

            async with session.post(
                turnstile_verify_url,
                data=payload,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Turnstile API error: {error_text}")
                    return TurnstileVerifyResponse(
                        success=False,
                        error=f"API_ERROR: {response.status}",
                        challenge_timestamp=None,
                        hostname=None,
                        action=None,
                    )

                data = await response.json()
                logger.debug(f"Turnstile API response: {data}")

                # Parse the response and return a structured object
                return TurnstileVerifyResponse(
                    success=data.get("success", False),
                    error=(
                        data.get("error-codes", None)[0]
                        if data.get("error-codes")
                        else None
                    ),
                    challenge_timestamp=data.get("challenge_timestamp"),
                    hostname=data.get("hostname"),
                    action=data.get("action"),
                )

    except aiohttp.ClientError as e:
        logger.error(f"Connection error to Turnstile API: {str(e)}")
        return TurnstileVerifyResponse(
            success=False,
            error=f"CONNECTION_ERROR: {str(e)}",
            challenge_timestamp=None,
            hostname=None,
            action=None,
        )
    except Exception as e:
        logger.error(f"Unexpected error in Turnstile verification: {str(e)}")
        return TurnstileVerifyResponse(
            success=False,
            error=f"UNEXPECTED_ERROR: {str(e)}",
            challenge_timestamp=None,
            hostname=None,
            action=None,
        )
