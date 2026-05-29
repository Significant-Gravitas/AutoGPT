"""
Discord API helper functions for making authenticated requests.
"""

import logging
from typing import Optional

from pydantic import BaseModel

from backend.data.model import OAuth2Credentials
from backend.util.request import Requests

logger = logging.getLogger(__name__)


class DiscordAPIException(Exception):
    """Exception raised for Discord API errors."""

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


class DiscordOAuthUser(BaseModel):
    """Model for Discord OAuth user response."""

    user_id: str
    username: str
    avatar_url: str
    banner: Optional[str] = None
    accent_color: Optional[int] = None


def get_api(credentials: OAuth2Credentials) -> Requests:
    """
    Create a Requests instance configured for Discord API calls with OAuth2 credentials.

    Args:
        credentials: The OAuth2 credentials containing the access token.

    Returns:
        A configured Requests instance for Discord API calls.
    """
    return Requests(
        trusted_origins=[],
        extra_headers={
            "Authorization": f"Bearer {credentials.access_token.get_secret_value()}",
            "Content-Type": "application/json",
        },
        raise_for_status=False,
    )


async def get_current_user(credentials: OAuth2Credentials) -> DiscordOAuthUser:
    """
    Fetch the current user's information using Discord OAuth2 API.

    Reference: https://discord.com/developers/docs/resources/user#get-current-user

    Args:
        credentials: The OAuth2 credentials.

    Returns:
        A model containing user data with avatar URL.

    Raises:
        DiscordAPIException: If the API request fails.
    """
    api = get_api(credentials)
    response = await api.get("https://discord.com/api/oauth2/@me")

    if not response.ok:
        error_text = response.text()
        raise DiscordAPIException(
            f"Failed to fetch user info: {response.status} - {error_text}",
            response.status,
        )

    data = response.json()
    logger.info(f"Discord OAuth2 API Response: {data}")

    # The /api/oauth2/@me endpoint returns a user object nested in the response
    user_info = data.get("user", {})
    logger.info(f"User info extracted: {user_info}")

    # Build avatar URL
    user_id = user_info.get("id")
    avatar_hash = user_info.get("avatar")
    if avatar_hash:
        # Custom avatar
        avatar_ext = "gif" if avatar_hash.startswith("a_") else "png"
        avatar_url = (
            f"https://cdn.discordapp.com/avatars/{user_id}/{avatar_hash}.{avatar_ext}"
        )
    else:
        # Default avatar based on discriminator or user ID
        discriminator = user_info.get("discriminator", "0")
        if discriminator == "0":
            # New username system - use user ID for default avatar
            default_avatar_index = (int(user_id) >> 22) % 6
        else:
            # Legacy discriminator system
            default_avatar_index = int(discriminator) % 5
        avatar_url = (
            f"https://cdn.discordapp.com/embed/avatars/{default_avatar_index}.png"
        )

    result = DiscordOAuthUser(
        user_id=user_id,
        username=user_info.get("username", ""),
        avatar_url=avatar_url,
        banner=user_info.get("banner"),
        accent_color=user_info.get("accent_color"),
    )

    logger.info(f"Returning user data: {result.model_dump()}")
    return result
