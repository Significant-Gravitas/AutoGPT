"""Tools for the copilot to post to Discord on the user's behalf.

``post_to_discord`` lets AutoPilot send a standalone message or open a new
thread in a channel of a Discord server the user has linked. The headline use
case is *scheduled* output: the user asks the bot "every Monday post an update
in #standup", AutoPilot schedules a follow-up turn (via ``schedule_followup``)
whose message instructs it to post, and at fire time it calls this tool.

Delivery and authorization live in the bot bridge (``CoPilotChatBridge``): the
tool only forwards the request over RPC. A post is only allowed into channels
belonging to a server the calling user has linked — enforced bridge-side, so a
scheduled turn running in a fresh session (no Discord context) is still safe.

``list_discord_channels`` backs name disambiguation and the "pick a channel"
flow when the user refers to a channel the model can't resolve on its own.
"""

import logging
from typing import Any

from backend.copilot.model import ChatSession
from backend.platform_linking.models import Platform
from backend.util.clients import get_copilot_chat_bridge_client
from backend.util.settings import Settings

from .base import BaseTool
from .models import (
    DiscordChannelListResponse,
    DiscordChannelSummary,
    DiscordPostedResponse,
    ErrorResponse,
    ToolResponseBase,
)

logger = logging.getLogger(__name__)

# Maps the bridge's stable DeliveryResult error codes to user-facing text the
# model can relay or act on. Anything unmapped falls back to the raw code.
_ERROR_MESSAGES: dict[str, str] = {
    "no_linked_servers": (
        "No Discord server is linked to this AutoGPT account yet. Link one via "
        "the bot's /setup command before posting."
    ),
    "channel_not_found": (
        "No channel matching that reference was found in your linked Discord "
        "server(s). Call list_discord_channels to see valid options."
    ),
    "not_authorized": (
        "That channel belongs to a Discord server that isn't linked to your "
        "account, so the bot won't post there."
    ),
    "ambiguous_channel": (
        "More than one channel matches that name across your linked servers. "
        "Use the numeric channel ID, or call list_discord_channels to pick one."
    ),
    "send_failed": (
        "Discord rejected the message — the bot likely lacks permission to post "
        "in that channel."
    ),
    "thread_failed": (
        "Couldn't create the thread — the bot likely lacks the 'Create Public "
        "Threads' permission in that channel."
    ),
}


def _discord_bot_configured() -> bool:
    """Whether a Discord bot is configured, used to gate tool availability.

    Assumes the copilot/executor process shares the bot token env with the
    bridge pod (true for ``poetry run app`` and the standard deployment). If
    they ever diverge, gate on bridge reachability instead — actual delivery
    still fails safe via the RPC; this only controls whether the LLM is
    offered the tool at all.
    """
    return bool(Settings().secrets.autopilot_bot_discord_token)


def _error_message(code: str | None) -> str:
    if not code:
        return "The Discord post could not be completed."
    return _ERROR_MESSAGES.get(code, f"The Discord post failed ({code}).")


class PostToDiscordTool(BaseTool):
    """Post a message or open a thread in a linked Discord server's channel."""

    @property
    def name(self) -> str:
        return "post_to_discord"

    @property
    def description(self) -> str:
        return (
            "Post to Discord on the user's behalf, into a channel of a Discord "
            "server they've linked. Set 'mode' to 'message' to send a standalone "
            "message, or 'thread' to open a NEW thread (provide 'thread_name'). "
            "'channel' accepts a channel name ('#standup' or 'standup') or a "
            "numeric channel ID. Pair this with schedule_followup for recurring "
            "posts (e.g. 'every Monday post the standup prompt in #standup'): "
            "schedule a follow-up whose message tells you to call this tool. If "
            "the channel can't be resolved, call list_discord_channels first."
        )

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def is_available(self) -> bool:
        return _discord_bot_configured()

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "channel": {
                    "type": "string",
                    "description": (
                        "Target channel: a name ('#standup' or 'standup') or a "
                        "numeric channel ID."
                    ),
                },
                "content": {
                    "type": "string",
                    "description": "The message body to post (or the thread's first message).",
                },
                "mode": {
                    "type": "string",
                    "enum": ["message", "thread"],
                    "description": (
                        "'message' posts a standalone message; 'thread' opens a "
                        "new thread named 'thread_name' and posts 'content' in it. "
                        "Defaults to 'message'."
                    ),
                },
                "thread_name": {
                    "type": "string",
                    "description": "Title for the new thread. Required when mode='thread'.",
                },
            },
            "required": ["channel", "content"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id if session else None
        if not user_id:
            return ErrorResponse(
                message="Authentication required.",
                error="auth_required",
                session_id=session_id,
            )

        channel: str | None = kwargs.get("channel")
        content: str | None = kwargs.get("content")
        mode: str = kwargs.get("mode") or "message"

        if not channel or not channel.strip():
            return ErrorResponse(
                message="`channel` is required.",
                error="missing_channel",
                session_id=session_id,
            )
        if not content or not content.strip():
            return ErrorResponse(
                message="`content` is required.",
                error="missing_content",
                session_id=session_id,
            )
        if mode not in ("message", "thread"):
            return ErrorResponse(
                message="`mode` must be 'message' or 'thread'.",
                error="invalid_mode",
                session_id=session_id,
            )

        client = get_copilot_chat_bridge_client()
        if mode == "thread":
            thread_name: str | None = kwargs.get("thread_name")
            if not thread_name or not thread_name.strip():
                return ErrorResponse(
                    message="`thread_name` is required when mode='thread'.",
                    error="missing_thread_name",
                    session_id=session_id,
                )
            result = await client.create_thread_in_channel(
                platform=Platform.DISCORD,
                user_id=user_id,
                channel=channel,
                thread_name=thread_name,
                content=content,
            )
        else:
            result = await client.send_message_to_channel(
                platform=Platform.DISCORD,
                user_id=user_id,
                channel=channel,
                content=content,
            )

        if not result.ok:
            return ErrorResponse(
                message=_error_message(result.error),
                error=result.error or "discord_post_failed",
                session_id=session_id,
            )

        where = "thread" if result.kind == "thread" else "message"
        link_note = f" ({result.url})" if result.url else ""
        return DiscordPostedResponse(
            message=f"Posted {where} to Discord{link_note}.",
            kind=result.kind,
            channel_id=result.channel_id or channel,
            ref_id=result.ref_id,
            url=result.url,
            session_id=session_id,
        )


class ListDiscordChannelsTool(BaseTool):
    """List channels the bot can post to across the user's linked servers."""

    @property
    def name(self) -> str:
        return "list_discord_channels"

    @property
    def description(self) -> str:
        return (
            "List the Discord channels the bot can post to, across every Discord "
            "server the user has linked. Use this to resolve a channel name to an "
            "ID, disambiguate duplicate names, or show the user a picker before "
            "calling post_to_discord."
        )

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def is_available(self) -> bool:
        return _discord_bot_configured()

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id if session else None
        if not user_id:
            return ErrorResponse(
                message="Authentication required.",
                error="auth_required",
                session_id=session_id,
            )

        channels = await get_copilot_chat_bridge_client().list_channels(
            platform=Platform.DISCORD,
            user_id=user_id,
        )
        summaries = [
            DiscordChannelSummary(
                id=c.id,
                name=c.name,
                server_id=c.server_id,
                server_name=c.server_name,
            )
            for c in channels
        ]
        if summaries:
            message = f"Found {len(summaries)} channel(s) you can post to."
        else:
            message = (
                "No postable Discord channels found. Link a server via the bot's "
                "/setup command, or check the bot's channel permissions."
            )
        return DiscordChannelListResponse(
            message=message,
            channels=summaries,
            count=len(summaries),
            session_id=session_id,
        )
