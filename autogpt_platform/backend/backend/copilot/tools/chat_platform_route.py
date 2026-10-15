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
import re
from typing import Any

from pydantic import BaseModel

from backend.copilot.context import get_current_permissions
from backend.data.db_accessors import platform_linking_db
from backend.platform_linking.models import Platform
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
    user_id: str, block_name: str, input_data: dict[str, Any], session_id: str
) -> ErrorResponse | None:
    """The answer instead of a setup card, when the bot can post what the
    block was asked to post; ``None`` when the card stays.  That is a channel
    in a server the user linked, read from the block's own inputs: a DM link
    alone lets the bot DM the user and nothing more, and a target the inputs
    don't pin down is never assumed to be reachable."""
    platform = _bot_platform(block_name)
    if platform is None or _beyond_the_bot(block_name, input_data):
        return None
    ref = _target_ref(block_name, input_data)
    if not ref:
        return None
    server_ids = await _linked_server_ids(user_id, platform)
    if not server_ids:
        return None
    if platform == "telegram":
        # A linked group's server id is its chat id; the Bot API can't list
        # chats, so the link itself is the whole allowlist.
        channel = ref if ref in server_ids else None
    else:
        channel = await _linked_channel(
            user_id, platform, ref, str(input_data.get("server_name") or "")
        )
    if channel is None:
        return None
    name = platform.title()
    args = f"platform='{platform}', channel='{channel}'"
    if block_name == "CreateDiscordThreadBlock":
        args += ", mode='thread'"
    return ErrorResponse(
        message=(
            f"Do not set up {block_name}: it needs a {name} bot token the user "
            f"has not connected, and they never need one, because that "
            f"{_PLACE[platform]} is linked to the AutoGPT bot. Post with "
            f"tool:{POST_TOOL}({args}) instead, which needs no credential."
        ),
        error="use_post_to_chat_platform",
        session_id=session_id,
    ).from_outside()


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


def _target_ref(block_name: str, input_data: dict[str, Any]) -> str:
    if block_name == "SendTelegramMessageBlock":
        return _telegram_chat_id(input_data) or ""
    field = "channel" if block_name == "SendSlackMessageBlock" else "channel_name"
    return str(input_data.get(field) or "").strip()


def _telegram_chat_id(input_data: dict[str, Any]) -> str | None:
    try:
        return str(int(input_data.get("chat_id")))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


async def _linked_server_ids(user_id: str, platform: str) -> frozenset[str]:
    try:
        links = await platform_linking_db().list_server_links(user_id)
    except Exception:
        logger.warning("Could not read chat platform links", exc_info=True)
        return frozenset()
    return frozenset(
        str(link.platform_server_id)
        for link in links
        if link.platform.lower() == platform
    )


async def _linked_channel(
    user_id: str, platform: str, ref: str, server_name: str
) -> str | None:
    """The block's channel as post_to_chat_platform should name it, when it
    is one channel the bot can post to in the user's linked servers."""
    try:
        channels = await get_copilot_chat_bridge_client().list_channels(
            platform=Platform(platform.upper()), user_id=user_id
        )
    except Exception:
        logger.warning("Could not list chat platform channels", exc_info=True)
        return None
    wanted = ref.lstrip("#").strip()
    # Slack ids come back as team|channel|thread; the block names the channel.
    matches = [
        c
        for c in channels
        if (c.name.lower() == wanted.lower() or wanted in c.id.split("|"))
        and (not server_name or (c.server_name or "").lower() == server_name.lower())
    ]
    if len(matches) != 1:
        return None
    match = matches[0]
    # The tool takes no server, so a name another linked channel shares goes
    # over as the id, which the tool resolves only in Discord's plain form.
    if sum(c.name.lower() == match.name.lower() for c in channels) == 1:
        return f"#{match.name}"
    return match.id if "|" not in match.id else None


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
