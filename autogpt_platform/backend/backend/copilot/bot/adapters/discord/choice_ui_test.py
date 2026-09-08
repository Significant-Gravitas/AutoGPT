"""Tests for Discord native choice buttons (SECRT-2604)."""

from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest

from .choice_ui import build_choice_view

_CHOICES = "backend.copilot.bot.adapters.discord.choice_ui.choices"


def _interaction(
    *, channel_id: int = 111, guild_id: int | None = 222, user_id: int = 9
) -> MagicMock:
    interaction = MagicMock()
    interaction.channel_id = channel_id
    interaction.guild_id = guild_id
    interaction.channel = MagicMock(spec=discord.TextChannel)
    interaction.id = 555
    interaction.user = MagicMock()
    interaction.user.id = user_id
    interaction.user.display_name = "Bently"
    interaction.response = MagicMock()
    interaction.response.send_message = AsyncMock()
    interaction.response.edit_message = AsyncMock()
    return interaction


class TestBuildChoiceView:
    @pytest.mark.asyncio
    async def test_one_button_per_option(self):
        adapter = MagicMock()
        on_message = AsyncMock()
        view = build_choice_view(adapter, on_message, "tok", ["US", "EU", "AP"])
        assert len(view.children) == 3
        assert [b.label for b in view.children] == ["US", "EU", "AP"]

    @pytest.mark.asyncio
    async def test_labels_truncate_to_discord_button_cap(self):
        adapter = MagicMock()
        on_message = AsyncMock()
        long_label = "x" * 200
        view = build_choice_view(adapter, on_message, "tok", [long_label])
        assert len(view.children[0].label) == 80


class TestChoiceButtonCallback:
    @pytest.mark.asyncio
    async def test_click_resolves_and_dispatches_as_message(self):
        adapter = MagicMock()
        on_message = AsyncMock()
        interaction = _interaction()

        with patch(f"{_CHOICES}.resolve_choice", new=AsyncMock(return_value="EU")):
            view = build_choice_view(adapter, on_message, "tok", ["US", "EU"])
            await view.children[1].callback(interaction)

        interaction.response.edit_message.assert_awaited_once_with(
            content="✅ You answered: EU", view=None
        )
        on_message.assert_awaited_once()
        ctx, dispatched_adapter = on_message.await_args.args
        assert dispatched_adapter is adapter
        assert ctx.text == "EU"
        assert ctx.platform == "discord"
        assert ctx.channel_id == "111"
        assert ctx.user_id == "9"
        assert ctx.username == "Bently"
        assert ctx.channel_type == "channel"
        assert ctx.bot_mentioned is True

    @pytest.mark.asyncio
    async def test_dm_interaction_maps_to_dm_channel_type(self):
        adapter = MagicMock()
        on_message = AsyncMock()
        interaction = _interaction(guild_id=None)

        with patch(f"{_CHOICES}.resolve_choice", new=AsyncMock(return_value="US")):
            view = build_choice_view(adapter, on_message, "tok", ["US"])
            await view.children[0].callback(interaction)

        ctx, _ = on_message.await_args.args
        assert ctx.channel_type == "dm"
        assert ctx.server_id is None

    @pytest.mark.asyncio
    async def test_expired_token_shows_ephemeral_notice_and_does_not_dispatch(self):
        adapter = MagicMock()
        on_message = AsyncMock()
        interaction = _interaction()

        with patch(f"{_CHOICES}.resolve_choice", new=AsyncMock(return_value=None)):
            view = build_choice_view(adapter, on_message, "tok", ["US", "EU"])
            await view.children[0].callback(interaction)

        interaction.response.send_message.assert_awaited_once()
        assert "expired" in interaction.response.send_message.await_args.args[0]
        interaction.response.edit_message.assert_not_awaited()
        on_message.assert_not_awaited()
