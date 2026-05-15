"""Slack slash command handlers — /setup, /help, /unlink."""

import logging
from typing import Any

from fastapi import Response
from fastapi.responses import JSONResponse

from backend.copilot.bot.bot_backend import BotBackend
from backend.util.settings import Settings

logger = logging.getLogger(__name__)


async def handle(api: BotBackend, form: dict[str, str]) -> Response:
    command = form.get("command", "")
    if command == "/setup":
        return await _setup(api, form)
    if command == "/help":
        return _help()
    if command == "/unlink":
        return _unlink()
    return _ephemeral(f"Unknown command: {command}")


async def _setup(api: BotBackend, form: dict[str, str]) -> Response:
    team_id = form.get("team_id", "")
    user_id = form.get("user_id", "")
    user_name = form.get("user_name", "")
    team_domain = form.get("team_domain", "")
    channel_id = form.get("channel_id", "")

    if not team_id or not user_id:
        return _ephemeral(
            "Slack didn't send the workspace/user info. Try the command again."
        )

    try:
        result = await api.create_link_token(
            platform="slack",
            platform_server_id=team_id,
            platform_user_id=user_id,
            platform_username=user_name,
            server_name=team_domain,
            channel_id=channel_id,
        )
    except Exception:
        logger.exception("Slack /setup link token creation failed")
        return _ephemeral(
            "Something went wrong creating the setup link. Try again in a moment."
        )

    return _link_response(
        text=(
            "*Set up AutoPilot for this workspace*\n\n"
            "Click the button below to link this workspace to your AutoGPT "
            "account. The link expires in 30 minutes."
        ),
        label="Link Workspace",
        url=result.link_url,
    )


def _help() -> Response:
    return _ephemeral(
        "*AutoPilot for Slack*\n"
        "• Run `/setup` in this workspace to link it to an AutoGPT account.\n"
        "• Mention `@AutoPilot` in a channel to start a conversation in a thread.\n"
        "• DM `@AutoPilot` to chat with your personal AutoPilot account.\n"
        "• Run `/unlink` to manage your linked workspace and DM."
    )


def _unlink() -> Response:
    config = Settings().config
    base_url = (config.frontend_base_url or config.platform_base_url).rstrip("/")
    if not base_url:
        return _ephemeral(
            "Settings page isn't configured — ask an admin to set FRONTEND_BASE_URL."
        )
    return _link_response(
        text="Manage your workspace and DM links from your AutoGPT settings.",
        label="Open Settings",
        url=f"{base_url}/profile/settings",
    )


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
