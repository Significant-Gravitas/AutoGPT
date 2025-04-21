import logging

import aiohttp

from backend.util.settings import Settings

from .models import TurnstileVerifyRequest, TurnstileVerifyResponse

logger = logging.getLogger(__name__)
settings = Settings()

# Cloudflare Turnstile API endpoint
TURNSTILE_VERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"


class TurnstileService:
    @staticmethod
    async def verify_token(request: TurnstileVerifyRequest) -> TurnstileVerifyResponse:
        """
        Verify a Cloudflare Turnstile token by making a request to the Cloudflare API.
        """
        # Get the secret key from settings
        secret_key = settings.secrets.turnstile_secret_key

        if not secret_key:
            logger.error("Turnstile secret key is not configured")
            return TurnstileVerifyResponse(
                success=False,
                error="CONFIGURATION_ERROR",
                challenge_ts=None,
                hostname=None,
                action=None,
            )

        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "secret": secret_key,
                    "response": request.token,
                }

                if request.action:
                    payload["action"] = request.action

                logger.debug(f"Verifying Turnstile token with action: {request.action}")

                async with session.post(
                    TURNSTILE_VERIFY_URL,
                    data=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Turnstile API error: {error_text}")
                        return TurnstileVerifyResponse(
                            success=False,
                            error=f"API_ERROR: {response.status}",
                            challenge_ts=None,
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
                        challenge_ts=data.get("challenge_ts"),
                        hostname=data.get("hostname"),
                        action=data.get("action"),
                    )

        except aiohttp.ClientError as e:
            logger.error(f"Connection error to Turnstile API: {str(e)}")
            return TurnstileVerifyResponse(
                success=False,
                error=f"CONNECTION_ERROR: {str(e)}",
                challenge_ts=None,
                hostname=None,
                action=None,
            )
        except Exception as e:
            logger.error(f"Unexpected error in Turnstile verification: {str(e)}")
            return TurnstileVerifyResponse(
                success=False,
                error=f"UNEXPECTED_ERROR: {str(e)}",
                challenge_ts=None,
                hostname=None,
                action=None,
            )
