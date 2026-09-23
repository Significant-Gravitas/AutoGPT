"""Tests for the shared slash-command policy."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.util.exceptions import LinkAlreadyExistsError

from .command_core import setup_reply, unlink_reply

_CORE = "backend.copilot.bot.command_core"


def _api(**overrides) -> MagicMock:
    api = MagicMock()
    api.create_link_token = AsyncMock(
        return_value=MagicMock(link_url="https://x/link/tok"), **overrides
    )
    return api


async def _setup(api) -> object:
    return await setup_reply(
        api,
        platform="slack",
        server_noun="workspace",
        platform_server_id="T1",
        platform_user_id="U1",
        platform_username="bently",
        server_name="acme",
        channel_id="C1",
    )


@pytest.mark.asyncio
async def test_setup_success_returns_link_button():
    reply = await _setup(_api())
    assert reply.button_url == "https://x/link/tok"
    assert reply.button_label == "Link Workspace"
    assert "Set up AutoGPT for acme" in reply.text
    assert "expires in 30 minutes" in reply.text


@pytest.mark.asyncio
async def test_setup_already_linked_is_friendly_not_error():
    api = _api()
    api.create_link_token = AsyncMock(side_effect=LinkAlreadyExistsError("dup"))
    reply = await _setup(api)
    assert reply.button_url is None
    assert "already linked" in reply.text
    assert "workspace" in reply.text  # platform's own noun


@pytest.mark.asyncio
async def test_setup_failure_returns_generic_message():
    api = _api()
    api.create_link_token = AsyncMock(side_effect=RuntimeError("boom"))
    reply = await _setup(api)
    assert reply.button_url is None
    assert "went wrong" in reply.text.lower()


def test_unlink_points_at_settings_bots():
    fake = MagicMock()
    fake.config.frontend_base_url = "https://app.example"
    fake.config.platform_base_url = ""
    with patch(f"{_CORE}.Settings", return_value=fake):
        reply = unlink_reply()
    assert reply.button_url == "https://app.example/settings/bots"
    assert reply.button_label == "Open Settings"


def test_unlink_without_base_url_falls_back_to_text():
    fake = MagicMock()
    fake.config.frontend_base_url = ""
    fake.config.platform_base_url = ""
    with patch(f"{_CORE}.Settings", return_value=fake):
        reply = unlink_reply()
    assert reply.button_url is None
    assert "Settings → Bots" in reply.text
