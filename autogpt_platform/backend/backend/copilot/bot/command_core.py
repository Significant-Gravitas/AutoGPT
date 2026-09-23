"""Shared slash-command policy — /setup and /unlink, platform-neutral.

Each adapter owns its transport (Discord interactions, Slack form POSTs) and
its rendering (View buttons vs block kit), but the *policy* — how a link token
is minted, what happens when the server is already linked, where unlink points
— must not drift between platforms. Adapters call these and render the
returned ``CommandReply`` in their own markup.
"""

import logging

from pydantic import BaseModel

from backend.util.exceptions import LinkAlreadyExistsError
from backend.util.settings import Settings

from .bot_backend import BotBackend

logger = logging.getLogger(__name__)


class CommandReply(BaseModel):
    """A transport-neutral ephemeral reply: text plus an optional link button."""

    text: str
    button_label: str | None = None
    button_url: str | None = None


async def setup_reply(
    api: BotBackend,
    *,
    platform: str,
    server_noun: str,
    platform_server_id: str,
    platform_user_id: str,
    platform_username: str,
    server_name: str,
    channel_id: str,
) -> CommandReply:
    """Mint a server link token and describe the outcome.

    ``server_noun`` is the platform's word for a server ("server",
    "workspace") so the shared copy reads natively everywhere.
    """
    try:
        result = await api.create_link_token(
            platform=platform,
            platform_server_id=platform_server_id,
            platform_user_id=platform_user_id,
            platform_username=platform_username,
            server_name=server_name,
            channel_id=channel_id,
        )
    except LinkAlreadyExistsError:
        return CommandReply(
            text=(
                f"This {server_noun} is already linked to an AutoGPT account — "
                "just mention me or DM me to chat. Run /unlink to manage the "
                "link."
            )
        )
    except Exception:
        logger.exception("%s /setup link token creation failed", platform)
        return CommandReply(
            text=(
                "Something went wrong creating the setup link. Try again in a "
                "moment."
            )
        )

    display_name = server_name or f"this {server_noun}"
    return CommandReply(
        text=(
            f"**Set up AutoGPT for {display_name}**\n\n"
            f"Click the button below to connect this {server_noun} to your "
            "AutoGPT account. Once confirmed, everyone here can mention me to "
            "use AutoGPT.\n\n"
            "All usage will be billed to your account.\n"
            "This link expires in 30 minutes."
        ),
        button_label=f"Link {server_noun.capitalize()}",
        button_url=result.link_url,
    )


def unlink_reply() -> CommandReply:
    """Point the user at the Bots settings page — unlinking needs web auth."""
    message = (
        "Unlinking requires authentication, so it has to be done from the "
        "web. Click below to manage your linked accounts."
    )
    base_url = _base_url()
    if not base_url:
        return CommandReply(
            text=f"{message}\n\nOpen AutoGPT on the web and go to Settings → Bots."
        )
    return CommandReply(
        text=message,
        button_label="Open Settings",
        button_url=f"{base_url}/settings/bots",
    )


def _base_url() -> str | None:
    config = Settings().config
    base_url = (config.frontend_base_url or config.platform_base_url).rstrip("/")
    return base_url or None
