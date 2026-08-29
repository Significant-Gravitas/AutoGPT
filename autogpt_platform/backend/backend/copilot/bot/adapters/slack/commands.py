"""Slack slash-command handlers — /setup, /help, /unlink.

Policy (link-token minting, already-linked handling, unlink destination) is
shared with every platform via ``command_core``; this module owns only Slack's
transport: form parsing and block-kit rendering.
"""

import logging
from typing import Any

from fastapi import Response
from fastapi.responses import JSONResponse

from backend.copilot.bot.bot_backend import BotBackend
from backend.copilot.bot.command_core import CommandReply, setup_reply, unlink_reply

from .text import to_mrkdwn

logger = logging.getLogger(__name__)


async def handle(api: BotBackend, form: dict[str, str]) -> Response:
    command = form.get("command", "")
    if command == "/setup":
        return await _setup(api, form)
    if command == "/help":
        return _help()
    if command == "/unlink":
        return _render(unlink_reply())
    return _ephemeral(f"Unknown command: {command}")


async def _setup(api: BotBackend, form: dict[str, str]) -> Response:
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
        platform_server_id=team_id,
        platform_user_id=user_id,
        platform_username=form.get("user_name", ""),
        server_name=form.get("team_domain", ""),
        channel_id=form.get("channel_id", ""),
    )
    return _render(reply)


def _help() -> Response:
    return _ephemeral(
        "*AutoGPT for Slack*\n"
        "• Run `/setup` in this workspace to link it to an AutoGPT account.\n"
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
