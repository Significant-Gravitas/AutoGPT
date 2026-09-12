"""Tests for Discord native choice buttons (SECRT-2604)."""

from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest

from backend.copilot.bot.choices import ResolvedChoice

from . import choice_ui
from .choice_ui import _ChoiceButton, build_choice_view

_CHOICES = "backend.copilot.bot.adapters.discord.choice_ui.choices"


def _buttons(view: discord.ui.View) -> list[_ChoiceButton]:
    """``View.children`` is typed as the base ``Item``; ours are all ours."""
    return [cast(_ChoiceButton, child) for child in view.children]


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


@pytest.fixture
def handler():
    """Register a click handler, as the adapter does once at startup.

    The buttons are stateless, so the adapter and callback are module-level
    rather than captured per view — a click has to resolve in a process that
    may never have sent the message.
    """
    adapter = MagicMock()
    on_message = AsyncMock()
    choice_ui.register_choice_handler(MagicMock(), adapter, on_message)
    yield adapter, on_message
    choice_ui._adapter = None
    choice_ui._on_message = None


class TestBuildChoiceView:
    @pytest.mark.asyncio
    async def test_one_button_per_option(self):
        view = build_choice_view("tok", ["US", "EU", "AP"])
        assert len(view.children) == 3
        assert [b.item.label for b in _buttons(view)] == ["US", "EU", "AP"]

    @pytest.mark.asyncio
    async def test_labels_truncate_to_discord_button_cap(self):
        view = build_choice_view("tok", ["x" * 200])
        label = _buttons(view)[0].item.label
        assert label is not None and len(label) == 80

    @pytest.mark.asyncio
    async def test_buttons_are_stateless_so_they_survive_a_restart(self):
        # Everything needed to resolve a click rides in the custom_id, and
        # the view never times out — an in-memory view with generated ids
        # went dead on every deploy and on View.timeout, giving the user
        # "This interaction failed" while the Redis token stayed valid.
        view = build_choice_view("abc123", ["US", "EU"])
        assert [b.custom_id for b in _buttons(view)] == [
            "qans:abc123:0",
            "qans:abc123:1",
        ]
        assert view.timeout is None


class TestChoiceButtonCallback:
    @pytest.mark.asyncio
    async def test_click_resolves_and_dispatches_as_message(self, handler):
        adapter, on_message = handler
        interaction = _interaction()

        with patch(
            f"{_CHOICES}.resolve_choice",
            new=AsyncMock(return_value=ResolvedChoice(text="EU")),
        ):
            view = build_choice_view("tok", ["US", "EU"])
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
    async def test_dm_interaction_maps_to_dm_channel_type(self, handler):
        _, on_message = handler
        interaction = _interaction(guild_id=None)

        with patch(
            f"{_CHOICES}.resolve_choice",
            new=AsyncMock(return_value=ResolvedChoice(text="US")),
        ):
            view = build_choice_view("tok", ["US"])
            await view.children[0].callback(interaction)

        ctx, _ = on_message.await_args.args
        assert ctx.channel_type == "dm"
        assert ctx.server_id is None

    @pytest.mark.asyncio
    async def test_expired_token_shows_ephemeral_notice_and_does_not_dispatch(
        self, handler
    ):
        _, on_message = handler
        interaction = _interaction()

        with patch(
            f"{_CHOICES}.resolve_choice",
            new=AsyncMock(return_value=ResolvedChoice(text=None)),
        ):
            view = build_choice_view("tok", ["US", "EU"])
            await view.children[0].callback(interaction)

        interaction.response.send_message.assert_awaited_once()
        assert "expired" in interaction.response.send_message.await_args.args[0]
        interaction.response.edit_message.assert_not_awaited()
        on_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dispatches_even_if_the_ack_edit_fails(self, handler):
        # resolve_choice has already consumed the token, so the answer exists
        # only in this call: a 404 on a deleted message (or 40060 on a fast
        # double-click) must not cost the user their turn.
        _, on_message = handler
        interaction = _interaction()
        interaction.response.edit_message = AsyncMock(
            side_effect=discord.HTTPException(MagicMock(status=404), "gone")
        )

        with patch(
            f"{_CHOICES}.resolve_choice",
            new=AsyncMock(return_value=ResolvedChoice(text="EU")),
        ):
            view = build_choice_view("tok", ["US", "EU"])
            await view.children[1].callback(interaction)

        on_message.assert_awaited_once()
