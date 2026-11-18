"""Slack slash-command handlers — /setup, /help, /unlink.

Policy (link-token minting, already-linked handling, unlink destination) is
shared with every platform via ``command_core``; this module owns only Slack's
transport: form parsing and block-kit rendering.
"""

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import Response
from fastapi.responses import JSONResponse
from slack_sdk.web.async_client import AsyncWebClient

from backend.copilot.bot.bot_backend import BotBackend
from backend.copilot.bot.command_core import (
    AdminCheck,
    CommandReply,
    setup_reply,
    unlink_reply,
)

from .text import to_mrkdwn

logger = logging.getLogger(__name__)

# The adapter's per-workspace client lookup: None when the workspace has no
# usable bot token.
ClientFor = Callable[[str], Awaitable[AsyncWebClient | None]]

# users.info flags that make a member an administrator of the workspace.
_ADMIN_FLAGS = ("is_admin", "is_owner", "is_primary_owner")


async def handle(
    api: BotBackend, form: dict[str, str], client_for: ClientFor
) -> Response:
    command = form.get("command", "")
    if command == "/setup":
        return await _setup(api, form, client_for)
    if command == "/help":
        return _help()
    if command == "/unlink":
        return _render(unlink_reply())
    return _ephemeral(f"Unknown command: {command}")


async def _setup(
    api: BotBackend, form: dict[str, str], client_for: ClientFor
) -> Response:
    team_id = form.get("team_id", "")
    user_id = form.get("user_id", "")

    if not team_id or not user_id:
        return _ephemeral(
            "Slack didn't send the workspace/user info. Try the command again."
        )

    reply = await setup_reply(
        api,
        platform="slack",
        server_noun="workspace",
        admin_check=await _workspace_admin_check(client_for, team_id, user_id),
        admin_role="workspace admin or owner",
        platform_server_id=team_id,
        platform_user_id=user_id,
        platform_username=form.get("user_name", ""),
        server_name=form.get("team_domain", ""),
        channel_id=form.get("channel_id", ""),
    )
    return _render(reply)


async def _workspace_admin_check(
    client_for: ClientFor, team_id: str, user_id: str
) -> AdminCheck:
    """Ask Slack whether ``user_id`` administers the workspace.

    The slash-command form says who ran it but not their role, so this reads
    their member record with the workspace's own bot token (``users:read``).
    """
    try:
        client = await client_for(team_id)
        if client is None:
            return AdminCheck.UNVERIFIED
        response = await client.users_info(user=user_id)
    except Exception:
        logger.warning("Couldn't read the Slack member running /setup", exc_info=True)
        return AdminCheck.UNVERIFIED
    user = response.get("user") or {}
    if any(user.get(flag) for flag in _ADMIN_FLAGS):
        return AdminCheck.ADMIN
    return AdminCheck.NOT_ADMIN


def _help() -> Response:
    return _ephemeral(
        "*AutoGPT for Slack*\n"
        "• Run `/setup` to link this workspace to an AutoGPT account "
        "(workspace admins and owners only).\n"
        "• Mention `@AutoGPT` in a channel to start a conversation in a thread.\n"
        "• DM `@AutoGPT` to chat with your personal AutoGPT account.\n"
        "• Run `/unlink` to manage your linked workspace and DM."
    )


def _render(reply: CommandReply) -> Response:
    text = to_mrkdwn(reply.text)
    if reply.button_label and reply.button_url:
        return _link_response(text=text, label=reply.button_label, url=reply.button_url)
    return _ephemeral(text)


def _ephemeral(text: str) -> Response:
    return JSONResponse({"response_type": "ephemeral", "text": text})


def _link_response(text: str, label: str, url: str) -> Response:
    blocks: list[dict[str, Any]] = [
        {"type": "section", "text": {"type": "mrkdwn", "text": text}},
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": label},
                    "url": url,
                }
            ],
        },
    ]
    return JSONResponse({"response_type": "ephemeral", "text": text, "blocks": blocks})
