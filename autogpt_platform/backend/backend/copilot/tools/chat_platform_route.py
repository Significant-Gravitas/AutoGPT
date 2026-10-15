"""Send a bot-token send block's missing credential to the AutoGPT bot.

``SendDiscordMessageBlock`` and its kin log in with a bot token the user must
bring, post, and log out. A user who linked that platform to the AutoGPT bot
never needs one: ``post_to_chat_platform`` posts there as the bot with no
credential. A setup card asking for the token was a dead end the model kept
walking back into, and on a scheduled turn, with nobody to answer it, the
step was skipped outright although the bot could have posted.
"""

import asyncio
import logging

from pydantic import BaseModel

from backend.copilot.context import get_current_permissions
from backend.data.db_accessors import platform_linking_db
from backend.util.clients import get_copilot_chat_bridge_client

from .chat_platform import _any_chat_platform_configured
from .models import ErrorResponse

logger = logging.getLogger(__name__)

POST_TOOL = "post_to_chat_platform"

# Blocks whose whole job post_to_chat_platform does, by the platform it posts
# to. Embeds, files, reads and lookups are not posts the tool can make, and
# neither is SendDiscordDMBlock: it DMs any user, the bot only the user.
BOT_TOKEN_SEND_BLOCKS: dict[str, str] = {
    "SendDiscordMessageBlock": "discord",
    "CreateDiscordThreadBlock": "discord",
    "SendSlackMessageBlock": "slack",
    "SendTelegramMessageBlock": "telegram",
}


async def bot_post_redirect(
    user_id: str, block_name: str, session_id: str
) -> ErrorResponse | None:
    """The answer instead of a setup card, when the user linked a server on
    the block's platform to the bot; ``None`` when the card stays.  Every
    block here posts into a channel, which the bot does only on a linked
    server: a DM link alone lets it DM the user and nothing more."""
    platform = _bot_platform(block_name)
    if platform is None:
        return None
    if platform not in (await chat_platform_links(user_id)).servers:
        return None
    name = platform.title()
    return ErrorResponse(
        message=(
            f"Do not set up {block_name}: it needs a {name} bot token the user "
            f"has not connected, and they never need one, because {name} is "
            "linked to the AutoGPT bot. Post with "
            f"tool:{POST_TOOL} (platform='{platform}') instead, which needs no "
            "credential."
        ),
        error="use_post_to_chat_platform",
        session_id=session_id,
    ).from_outside()


def bot_link_hint(block_name: str) -> str:
    """The other way to post, appended to the card for a user with no link."""
    platform = _bot_platform(block_name)
    if platform is None:
        return ""
    return (
        f" Or, with no token: once the user links {platform.title()} to the "
        f"AutoGPT bot (its setup command), tool:{POST_TOOL} posts there as the "
        "bot."
    )


def _bot_platform(block_name: str) -> str | None:
    """The platform the bot could post to in this block's place, when this
    deployment runs the bot and this turn may call the tool."""
    platform = BOT_TOKEN_SEND_BLOCKS.get(block_name)
    if platform is None or not _any_chat_platform_configured():
        return None
    permissions = get_current_permissions()
    if permissions is not None and POST_TOOL not in (
        permissions.effective_allowed_tools(frozenset({POST_TOOL}))
    ):
        return None
    return platform


class ChatPlatformLinks(BaseModel):
    """Platforms the user linked to the bot: a server, where it can post into
    channels, or their own DMs, where it can DM them."""

    servers: frozenset[str] = frozenset()
    dms: frozenset[str] = frozenset()


async def chat_platform_links(user_id: str) -> ChatPlatformLinks:
    try:
        db = platform_linking_db()
        servers, users = await asyncio.gather(
            db.list_server_links(user_id), db.list_user_links(user_id)
        )
    except Exception:
        logger.warning("Could not read chat platform links", exc_info=True)
        return ChatPlatformLinks()
    return ChatPlatformLinks(
        servers=frozenset(link.platform.lower() for link in servers),
        dms=frozenset(link.platform.lower() for link in users),
    )
