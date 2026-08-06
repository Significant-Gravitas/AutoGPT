"""Telegram bot-command handlers — /setup, /help, /unlink.

Policy comes from the shared ``command_core``; this module owns only
Telegram's transport: commands arrive as ordinary messages with a
``bot_command`` entity, and replies render as HTML messages with an inline
URL button.
"""

import html
import logging
from typing import Any, Optional

from backend.copilot.bot import sessions
from backend.copilot.bot.bot_backend import BotBackend
from backend.copilot.bot.command_core import CommandReply, setup_reply, unlink_reply

from .api_client import TelegramClient
from .targets import encode_target
from .text import to_html

logger = logging.getLogger(__name__)

# Published to Telegram's command menu via setMyCommands on startup.
COMMAND_MENU = [
    {"command": "setup", "description": "Link this group to an AutoGPT account"},
    {"command": "new", "description": "Start a fresh AutoGPT conversation"},
    {"command": "help", "description": "Show AutoGPT bot usage info"},
    {"command": "unlink", "description": "Manage linked chats from your settings"},
]

_HELP_TEXT = (
    "**AutoGPT for Telegram**\n"
    "• Add me to a group and run /setup to link it to an AutoGPT account.\n"
    "• Mention me (or reply to my messages) in a group to chat.\n"
    "• Message me directly to chat with your personal AutoGPT account.\n"
    "• Run /new to start a fresh conversation.\n"
    "• Run /unlink to manage your linked groups and DM."
)


def parse_command(message: dict[str, Any], bot_username: str) -> Optional[str]:
    """Return the command name ("setup") when ``message`` is a bot command
    addressed to us (bare ``/setup`` or the disambiguated ``/setup@OurBot``
    form Telegram uses in groups), else None."""
    text = message.get("text") or ""
    entities = message.get("entities") or []
    is_command = any(
        e.get("type") == "bot_command" and e.get("offset") == 0 for e in entities
    )
    if not is_command or not text.startswith("/"):
        return None
    command = text.split()[0][1:]
    if "@" in command:
        command, _, target = command.partition("@")
        if bot_username and target.casefold() != bot_username.casefold():
            return None
    return command.casefold()


async def handle(
    api: BotBackend, client: TelegramClient, message: dict[str, Any], command: str
) -> None:
    chat = message.get("chat") or {}
    sender = message.get("from") or {}
    chat_id = str(chat.get("id", ""))
    is_private = chat.get("type") == "private"
    api.track_event(
        platform="telegram",
        event_type="command_used",
        server_id=None if is_private else chat_id,
        command_name=command,
    )

    if command == "help":
        await _send(client, chat_id, CommandReply(text=_HELP_TEXT), message)
        return
    if command == "unlink":
        await _send(client, chat_id, unlink_reply(), message)
        return
    if command == "new":
        target = encode_target(chat_id, message.get("message_thread_id"))
        try:
            await sessions.clear_session("telegram", target)
        except Exception:
            logger.exception("Failed to clear telegram session for /new")
            await _send(
                client,
                chat_id,
                CommandReply(
                    text=(
                        "Couldn't reset the conversation right now. Please try "
                        "again in a moment."
                    )
                ),
                message,
            )
            return
        await _send(
            client,
            chat_id,
            CommandReply(
                text="Started a fresh conversation — send a message to begin."
            ),
            message,
        )
        return
    if command in ("setup", "start"):
        if is_private:
            await _send(
                client,
                chat_id,
                CommandReply(
                    text=(
                        "/setup links a group. To link your own DMs, just send "
                        "me a message and I'll walk you through it."
                    )
                ),
                message,
            )
            return
        reply = await setup_reply(
            api,
            platform="telegram",
            server_noun="group",
            platform_server_id=chat_id,
            platform_user_id=str(sender.get("id", "")),
            platform_username=sender.get("username")
            or sender.get("first_name")
            or "unknown",
            server_name=chat.get("title") or "",
            channel_id=chat_id,
        )
        await _send(client, chat_id, reply, message)
        return
    logger.debug("Ignoring unknown Telegram command /%s", command)


async def _send(
    client: TelegramClient,
    chat_id: str,
    reply: CommandReply,
    message: dict[str, Any],
) -> None:
    text = to_html(reply.text)
    params: dict[str, Any] = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "message_thread_id": message.get("message_thread_id"),
    }
    if reply.button_label and reply.button_url:
        params["reply_markup"] = {
            "inline_keyboard": [[{"text": reply.button_label, "url": reply.button_url}]]
        }
    try:
        await client.call("sendMessage", **params)
    except Exception:
        if "reply_markup" not in params:
            logger.exception("Failed to send Telegram command reply")
            return
        # Telegram rejects some button URLs (e.g. localhost in local dev);
        # degrade to the URL as plain text so the command still answers.
        params.pop("reply_markup")
        # Escaped because parse_mode stays HTML — a bare & in the URL would
        # read as a malformed entity and kill the fallback too.
        params["text"] += html.escape(f"\n\n{reply.button_label}: {reply.button_url}")
        try:
            await client.call("sendMessage", **params)
        except Exception:
            logger.exception("Failed to send Telegram command reply")
