"""Tests for Slack slash command handlers."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from .commands import handle


def _form(**overrides: str) -> dict[str, str]:
    base = {
        "command": "/setup",
        "team_id": "T1",
        "team_domain": "myteam",
        "user_id": "U42",
        "user_name": "bently",
        "channel_id": "C1",
    }
    base.update(overrides)
    return base


def _body(response) -> dict:
    return json.loads(response.body.decode("utf-8"))


@pytest.mark.asyncio
async def test_setup_success_returns_link_button():
    api = MagicMock()
    api.create_link_token = AsyncMock(
        return_value=MagicMock(link_url="https://example.com/link/abc")
    )

    response = await handle(api, _form())

    payload = _body(response)
    assert payload["response_type"] == "ephemeral"
    assert "Set up AutoPilot" in payload["text"]
    button = payload["blocks"][1]["elements"][0]
    assert button["url"] == "https://example.com/link/abc"
    assert button["text"]["text"] == "Link Workspace"
    api.create_link_token.assert_awaited_once_with(
        platform="slack",
        platform_server_id="T1",
        platform_user_id="U42",
        platform_username="bently",
        server_name="myteam",
        channel_id="C1",
    )


@pytest.mark.asyncio
async def test_setup_missing_team_id_returns_ephemeral_error():
    api = MagicMock()
    api.create_link_token = AsyncMock()
    response = await handle(api, _form(team_id=""))
    payload = _body(response)
    assert payload["response_type"] == "ephemeral"
    assert "workspace/user info" in payload["text"]
    api.create_link_token.assert_not_awaited()


@pytest.mark.asyncio
async def test_setup_backend_failure_returns_ephemeral_error():
    api = MagicMock()
    api.create_link_token = AsyncMock(side_effect=RuntimeError("boom"))
    response = await handle(api, _form())
    payload = _body(response)
    assert payload["response_type"] == "ephemeral"
    assert "Something went wrong" in payload["text"]


@pytest.mark.asyncio
async def test_help_returns_usage():
    response = await handle(MagicMock(), _form(command="/help"))
    payload = _body(response)
    assert payload["response_type"] == "ephemeral"
    assert "/setup" in payload["text"]
    assert "/unlink" in payload["text"]


@pytest.mark.asyncio
async def test_unlink_with_frontend_url_returns_settings_button():
    settings = MagicMock()
    settings.config.frontend_base_url = "https://app.example.com"
    settings.config.platform_base_url = ""
    with patch(
        "backend.copilot.bot.adapters.slack.commands.Settings",
        return_value=settings,
    ):
        response = await handle(MagicMock(), _form(command="/unlink"))
    payload = _body(response)
    button = payload["blocks"][1]["elements"][0]
    assert button["url"] == "https://app.example.com/profile/settings"


@pytest.mark.asyncio
async def test_unlink_without_frontend_url_returns_error():
    settings = MagicMock()
    settings.config.frontend_base_url = ""
    settings.config.platform_base_url = ""
    with patch(
        "backend.copilot.bot.adapters.slack.commands.Settings",
        return_value=settings,
    ):
        response = await handle(MagicMock(), _form(command="/unlink"))
    payload = _body(response)
    assert payload["response_type"] == "ephemeral"
    assert "isn't configured" in payload["text"]


@pytest.mark.asyncio
async def test_unknown_command_returns_ephemeral_error():
    response = await handle(MagicMock(), _form(command="/whatever"))
    payload = _body(response)
    assert payload["response_type"] == "ephemeral"
    assert "Unknown command" in payload["text"]
