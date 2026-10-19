"""Point the model at the AutoGPT bot where a send block wants a bot token.

``SendDiscordMessageBlock`` and its kin log in with a bot token the user must
bring. ``post_to_chat_platform`` posts as the AutoGPT bot with no credential
wherever the user linked that platform to it, which no credential records:
the ``find_capability`` listing says where that is, and a send block's setup
card names the bot as the other way to post.
"""

import asyncio
import logging
import re
from typing import Any

from pydantic import BaseModel

from backend.copilot.context import get_current_permissions
from backend.data.db_accessors import platform_linking_db

from .chat_platform import _configured_chat_platforms

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


def bot_link_hint(block_name: str, input_data: dict[str, Any]) -> str:
    """The other way to post, appended to the card when the bot could post
    this once the user links where it goes."""
    platform = _bot_platform(block_name)
    if platform is None or _beyond_the_bot(block_name, input_data):
        return ""
    where = _PLACE[platform]
    if platform != "telegram":
        where += " that channel is in"
    return (
        f" Or, with no token: if the user links the {platform.title()} {where} "
        f"to the AutoGPT bot (its setup command), tool:{POST_TOOL} can post "
        "there as the bot."
    )


# What the user links to the bot on each platform.
_PLACE = {"discord": "server", "slack": "workspace", "telegram": "group"}

# A Slack member (U…, W…) or DM conversation (D…): the bot DMs only the user.
_SLACK_DM = re.compile(r"^(@|[UWD][A-Z0-9]{7,}$)")


def _beyond_the_bot(block_name: str, input_data: dict[str, Any]) -> bool:
    """Whether the block was asked for something the bot never does, linked
    or not: a DM to someone else, a reply into a thread, a private thread."""
    if block_name == "SendSlackMessageBlock":
        channel = str(input_data.get("channel") or "").strip()
        return bool(input_data.get("thread_ts")) or bool(_SLACK_DM.match(channel))
    if block_name == "CreateDiscordThreadBlock":
        return bool(input_data.get("is_private"))
    if block_name == "SendTelegramMessageBlock":
        if input_data.get("reply_to_message_id"):
            return True
        # A positive chat id is a private chat with a person, not a group.
        chat_id = _telegram_chat_id(input_data)
        return chat_id is not None and int(chat_id) > 0
    return False


def _telegram_chat_id(input_data: dict[str, Any]) -> str | None:
    chat_id = input_data.get("chat_id")
    if chat_id is None:
        return None
    try:
        return str(int(chat_id))
    except (TypeError, ValueError):
        return None


def _bot_platform(block_name: str) -> str | None:
    """The platform the bot could post to in this block's place, when this
    deployment runs the bot and this turn may call the tool."""
    platform = BOT_TOKEN_SEND_BLOCKS.get(block_name)
    if platform is None or platform not in _configured_chat_platforms():
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
