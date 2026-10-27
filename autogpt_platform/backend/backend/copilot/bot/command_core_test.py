"""Tests for the shared slash-command policy."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.util.exceptions import LinkAlreadyExistsError

from .command_core import dm_link_reply, setup_reply, unlink_reply

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


def _dm_api(*, linked: bool = False, account_hint: str | None = None) -> MagicMock:
    api = MagicMock()
    api.resolve_user = AsyncMock(
        return_value=MagicMock(linked=linked, account_hint=account_hint)
    )
    api.create_user_link_token = AsyncMock(
        return_value=MagicMock(link_url="https://x/link/dm-tok")
    )
    return api


async def _dm_link(api) -> object:
    return await dm_link_reply(
        api,
        platform="telegram",
        platform_display="Telegram",
        platform_user_id="42",
        platform_username="bently",
    )


@pytest.mark.asyncio
async def test_dm_link_unlinked_user_gets_link_button():
    api = _dm_api()
    reply = await _dm_link(api)
    assert reply.button_label == "Link Account"
    assert reply.button_url == "https://x/link/dm-tok"
    assert "expires in 30 minutes" in reply.text
    api.resolve_user.assert_awaited_once_with("telegram", "42", include_account=True)
    api.create_user_link_token.assert_awaited_once_with(
        platform="telegram", platform_user_id="42", platform_username="bently"
    )


@pytest.mark.asyncio
async def test_dm_link_linked_user_is_told_to_just_chat():
    api = _dm_api(linked=True, account_hint="b***@agpt.co")
    reply = await _dm_link(api)
    assert reply.button_url is None
    assert "Telegram DMs are linked" in reply.text
    api.create_user_link_token.assert_not_called()


@pytest.mark.asyncio
async def test_dm_link_linked_reply_names_the_account_and_the_way_out():
    api = _dm_api(linked=True, account_hint="b***@agpt.co")
    reply = await _dm_link(api)
    assert reply.text == (
        "Your Telegram DMs are linked to the AutoGPT account b***@agpt.co. "
        "Send me a message to start chatting.\n\n"
        "Using a different AutoGPT account? Send /unlink, sign in as "
        "b***@agpt.co and unlink these DMs, then message me again to link "
        "the right one."
    )


@pytest.mark.asyncio
async def test_dm_link_linked_reply_without_a_hint_still_offers_unlink():
    api = _dm_api(linked=True, account_hint=None)
    reply = await _dm_link(api)
    assert "linked to an AutoGPT account." in reply.text
    assert "Send /unlink, sign in as that account" in reply.text


@pytest.mark.asyncio
async def test_dm_link_race_with_existing_link_reads_as_linked():
    api = _dm_api()
    api.create_user_link_token = AsyncMock(side_effect=LinkAlreadyExistsError("dup"))
    reply = await _dm_link(api)
    assert reply.button_url is None
    assert "linked" in reply.text
    assert "/unlink" in reply.text


@pytest.mark.asyncio
async def test_dm_link_failure_returns_generic_message():
    api = _dm_api()
    api.resolve_user = AsyncMock(side_effect=RuntimeError("boom"))
    reply = await _dm_link(api)
    assert reply.button_url is None
    assert "went wrong" in reply.text.lower()
