"""Bot API key authentication for platform linking endpoints."""

import hmac
import os

from fastapi import HTTPException, Request

from backend.util.settings import Settings


async def get_bot_api_key(request: Request) -> str | None:
    """Extract the bot API key from the X-Bot-API-Key header."""
    return request.headers.get("x-bot-api-key")


def check_bot_api_key(api_key: str | None) -> None:
    """Validate the bot API key. Uses constant-time comparison.

    Reads the key from env on each call so rotated secrets take effect
    without restarting the process.
    """
    configured_key = os.getenv("PLATFORM_BOT_API_KEY", "")

    if not configured_key:
        settings = Settings()
        if settings.config.enable_auth:
            raise HTTPException(
                status_code=503,
                detail="Bot API key not configured.",
            )
        # Auth disabled (local dev) — allow without key
        return

    if not api_key or not hmac.compare_digest(api_key, configured_key):
        raise HTTPException(status_code=401, detail="Invalid bot API key.")
