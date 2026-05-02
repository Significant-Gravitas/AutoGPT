"""Tests for Discord slash command handlers.

Targets the ``_handle_*`` functions directly — sidesteps ``CommandTree``
registration since it requires a live ``discord.Client``.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.util.exceptions import LinkAlreadyExistsError

from ...bot_backend import LinkTokenResult
from .commands import _handle_help, _handle_setup, _handle_unlink


def _interaction(*, guild: bool = True) -> MagicMock:
    interaction = MagicMock()
    interaction.response.send_message = AsyncMock()
    interaction.response.defer = AsyncMock()
    interaction.followup.send = AsyncMock()
    if guild:
        # MagicMock treats `name` as a constructor kwarg for the mock's repr,
        # not as an attribute — so set it after construction.
        interaction.guild = MagicMock(id=123)
        interaction.guild.name = "Test Guild"
        interaction.user = MagicMock(id=456, display_name="Bently")
        interaction.channel_id = 789
    else:
        interaction.guild = None
        interaction.user = MagicMock(id=456, display_name="Bently")
        interaction.channel_id = None
    return interaction


def _api_with_token() -> MagicMock:
    api = MagicMock()
    api.create_link_token = AsyncMock(
        return_value=LinkTokenResult(
            token="abc",
            link_url="https://example.com/link/abc",
            expires_at="2099-01-01T00:00:00Z",
        )
    )
    return api


class TestHandleSetup:
    @pytest.mark.asyncio
    async def test_dm_invocation_rejects_early(self):
        interaction = _interaction(guild=False)
        api = _api_with_token()
        await _handle_setup(interaction, api)

        interaction.response.send_message.assert_awaited_once()
        api.create_link_token.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_guild_invocation_creates_token_and_posts_button(self):
        interaction = _interaction()
        api = _api_with_token()
        await _handle_setup(interaction, api)

        interaction.response.defer.assert_awaited_once_with(ephemeral=True)
        api.create_link_token.assert_awaited_once()
        call_kwargs = api.create_link_token.await_args.kwargs
        assert call_kwargs["platform"] == "discord"
        assert call_kwargs["platform_server_id"] == "123"
        assert call_kwargs["server_name"] == "Test Guild"

        interaction.followup.send.assert_awaited_once()
        sent = interaction.followup.send.await_args
        assert "Set up AutoPilot for Test Guild" in sent.args[0]
        assert sent.kwargs["view"] is not None

    @pytest.mark.asyncio
    async def test_already_linked_gets_friendly_message(self):
        interaction = _interaction()
        api = _api_with_token()
        api.create_link_token = AsyncMock(side_effect=LinkAlreadyExistsError("already"))

        await _handle_setup(interaction, api)

        interaction.followup.send.assert_awaited_once()
        msg = interaction.followup.send.await_args.args[0]
        assert "already linked" in msg

    @pytest.mark.asyncio
    async def test_backend_error_surfaces_generic_message(self):
        interaction = _interaction()
        api = _api_with_token()
        api.create_link_token = AsyncMock(side_effect=RuntimeError("boom"))

        await _handle_setup(interaction, api)

        interaction.followup.send.assert_awaited_once()
        msg = interaction.followup.send.await_args.args[0]
        assert "went wrong" in msg.lower()


class TestHandleHelp:
    @pytest.mark.asyncio
    async def test_help_sends_ephemeral_message(self):
        interaction = _interaction()
        await _handle_help(interaction)
        interaction.response.send_message.assert_awaited_once()
        assert interaction.response.send_message.await_args.kwargs["ephemeral"] is True
        body = interaction.response.send_message.await_args.args[0]
        assert "/setup" in body
        assert "/help" in body
        assert "/unlink" in body


class TestHandleUnlink:
    @pytest.mark.asyncio
    async def test_with_frontend_url_posts_button(self):
        interaction = _interaction()
        fake_settings = MagicMock()
        fake_settings.config.frontend_base_url = "http://localhost:3000"
        fake_settings.config.platform_base_url = ""
        with patch(
            "backend.copilot.bot.adapters.discord.commands.Settings",
            return_value=fake_settings,
        ):
            await _handle_unlink(interaction)

        interaction.response.send_message.assert_awaited_once()
        sent = interaction.response.send_message.await_args
        assert sent.kwargs["view"] is not None
        assert sent.kwargs["ephemeral"] is True

    @pytest.mark.asyncio
    async def test_falls_back_to_platform_base_url(self):
        interaction = _interaction()
        fake_settings = MagicMock()
        fake_settings.config.frontend_base_url = ""
        fake_settings.config.platform_base_url = "http://other"
        with patch(
            "backend.copilot.bot.adapters.discord.commands.Settings",
            return_value=fake_settings,
        ):
            await _handle_unlink(interaction)

        # Button uses the fallback URL.
        sent = interaction.response.send_message.await_args
        view = sent.kwargs["view"]
        assert any(
            "http://other" in getattr(child, "url", "") for child in view.children
        )

    @pytest.mark.asyncio
    async def test_no_urls_configured_sends_plain_text(self):
        interaction = _interaction()
        fake_settings = MagicMock()
        fake_settings.config.frontend_base_url = ""
        fake_settings.config.platform_base_url = ""
        with patch(
            "backend.copilot.bot.adapters.discord.commands.Settings",
            return_value=fake_settings,
        ):
            await _handle_unlink(interaction)

        sent = interaction.response.send_message.await_args
        assert "view" not in sent.kwargs or sent.kwargs.get("view") is None
        assert "Profile" in sent.args[0]
