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

from backend.copilot.context import get_current_permissions
from backend.data.db_accessors import platform_linking_db

from .chat_platform import _any_chat_platform_configured
from .models import ErrorResponse

logger = logging.getLogger(__name__)

POST_TOOL = "post_to_chat_platform"

# Blocks whose whole job post_to_chat_platform does, by the platform it posts
# to. Embeds, files, reads and lookups are not posts the tool can make.
BOT_TOKEN_SEND_BLOCKS: dict[str, str] = {
    "SendDiscordMessageBlock": "discord",
    "SendDiscordDMBlock": "discord",
    "CreateDiscordThreadBlock": "discord",
    "SendSlackMessageBlock": "slack",
    "SendTelegramMessageBlock": "telegram",
}


async def bot_post_redirect(
    user_id: str, block_name: str, session_id: str
) -> ErrorResponse | None:
    """The answer instead of a setup card, when the user linked the block's
    platform to the bot; ``None`` when the card stays."""
    platform = _bot_platform(block_name)
    if platform is None or platform not in await linked_chat_platforms(user_id):
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


async def linked_chat_platforms(user_id: str) -> frozenset[str]:
    """Platforms the user linked to the bot, by a server or by their DMs."""
    try:
        db = platform_linking_db()
        servers, users = await asyncio.gather(
            db.list_server_links(user_id), db.list_user_links(user_id)
        )
    except Exception:
        logger.warning("Could not read chat platform links", exc_info=True)
        return frozenset()
    return frozenset(link.platform.lower() for link in [*servers, *users])
