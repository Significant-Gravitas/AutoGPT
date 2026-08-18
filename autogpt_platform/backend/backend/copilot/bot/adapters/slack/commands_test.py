"""Tests for Slack slash-command handlers."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from . import commands


def _body(resp) -> dict:
    return json.loads(bytes(resp.body).decode())


@pytest.mark.asyncio
async def test_setup_returns_link_button():
    api = MagicMock()
    api.create_link_token = AsyncMock(
        return_value=MagicMock(link_url="https://example.com/link/abc")
    )
    resp = await commands.handle(
        api, {"command": "/setup", "team_id": "T1", "user_id": "U1"}
    )
    body = _body(resp)
    assert body["response_type"] == "ephemeral"
    assert body["blocks"][1]["elements"][0]["url"] == "https://example.com/link/abc"
    api.create_link_token.assert_awaited_once()
    assert api.create_link_token.await_args.kwargs["platform"] == "slack"


@pytest.mark.asyncio
async def test_setup_without_team_is_rejected():
    api = MagicMock()
    api.create_link_token = AsyncMock()
    resp = await commands.handle(api, {"command": "/setup", "user_id": "U1"})
    assert "workspace/user info" in _body(resp)["text"]
    api.create_link_token.assert_not_awaited()


@pytest.mark.asyncio
async def test_setup_already_linked_gets_friendly_message():
    from backend.util.exceptions import LinkAlreadyExistsError

    api = MagicMock()
    api.create_link_token = AsyncMock(side_effect=LinkAlreadyExistsError("linked"))
    resp = await commands.handle(
        api, {"command": "/setup", "team_id": "T1", "user_id": "U1"}
    )
    assert "already linked" in _body(resp)["text"]


@pytest.mark.asyncio
async def test_setup_link_failure_is_graceful():
    api = MagicMock()
    api.create_link_token = AsyncMock(side_effect=RuntimeError("boom"))
    resp = await commands.handle(
        api, {"command": "/setup", "team_id": "T1", "user_id": "U1"}
    )
    assert "went wrong" in _body(resp)["text"]


@pytest.mark.asyncio
async def test_help_is_ephemeral():
    resp = await commands.handle(MagicMock(), {"command": "/help"})
    assert _body(resp)["response_type"] == "ephemeral"
    assert "/setup" in _body(resp)["text"]


@pytest.mark.asyncio
async def test_unlink_points_at_settings():
    settings = MagicMock()
    settings.config.frontend_base_url = "https://app.example"
    settings.config.platform_base_url = ""
    with patch("backend.copilot.bot.command_core.Settings", return_value=settings):
        resp = await commands.handle(MagicMock(), {"command": "/unlink"})
    assert (
        _body(resp)["blocks"][1]["elements"][0]["url"]
        == "https://app.example/settings/bots"
    )


@pytest.mark.asyncio
async def test_unlink_without_base_url_returns_error():
    settings = MagicMock()
    settings.config.frontend_base_url = ""
    settings.config.platform_base_url = ""
    with patch("backend.copilot.bot.command_core.Settings", return_value=settings):
        resp = await commands.handle(MagicMock(), {"command": "/unlink"})
    assert "Settings → Bots" in _body(resp)["text"]


@pytest.mark.asyncio
async def test_unknown_command():
    resp = await commands.handle(MagicMock(), {"command": "/wat"})
    assert "Unknown command" in _body(resp)["text"]
