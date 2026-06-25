"""Tests for Discord slash command handlers.

Targets the ``_handle_*`` functions directly — sidesteps ``CommandTree``
registration since it requires a live ``discord.Client``.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest

from backend.util.exceptions import LinkAlreadyExistsError

from ...bot_backend import ChatSummary, LinkTokenResult
from .commands import (
    _handle_help,
    _handle_leave,
    _handle_new,
    _handle_resume,
    _handle_setup,
    _handle_unlink,
    _resume_to_session,
)


def _interaction(*, guild: bool = True, manage_guild: bool = True) -> MagicMock:
    interaction = MagicMock()
    interaction.response.send_message = AsyncMock()
    interaction.response.defer = AsyncMock()
    interaction.followup.send = AsyncMock()
    interaction.permissions = MagicMock(manage_guild=manage_guild)
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
    async def test_non_admin_guild_invocation_is_rejected(self):
        interaction = _interaction(manage_guild=False)
        api = _api_with_token()
        await _handle_setup(interaction, api)

        interaction.response.send_message.assert_awaited_once()
        assert interaction.response.send_message.await_args.kwargs["ephemeral"] is True
        assert "Manage Server" in interaction.response.send_message.await_args.args[0]
        interaction.response.defer.assert_not_awaited()
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
        assert "/new" in body
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
        view = sent.kwargs["view"]
        assert any(
            getattr(child, "url", "").endswith("/settings/bots")
            for child in view.children
        )

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
        assert "Bots" in sent.args[0]


class TestHandleNew:
    @pytest.mark.asyncio
    async def test_dm_clears_session(self):
        interaction = _interaction(guild=False)
        interaction.channel = MagicMock()
        interaction.channel_id = 555
        with patch(
            "backend.copilot.bot.sessions.clear_session",
            new=AsyncMock(),
        ) as mock_clear:
            await _handle_new(interaction)

        mock_clear.assert_awaited_once_with("discord", "555")
        interaction.response.send_message.assert_awaited_once()
        assert interaction.response.send_message.await_args.kwargs["ephemeral"] is True

    @pytest.mark.asyncio
    async def test_thread_clears_session(self):
        interaction = _interaction()
        interaction.channel = MagicMock(spec=discord.Thread)
        interaction.channel_id = 777
        with patch(
            "backend.copilot.bot.sessions.clear_session",
            new=AsyncMock(),
        ) as mock_clear:
            await _handle_new(interaction)

        mock_clear.assert_awaited_once_with("discord", "777")

    @pytest.mark.asyncio
    async def test_plain_channel_is_rejected(self):
        interaction = _interaction()
        interaction.channel = MagicMock()
        with patch(
            "backend.copilot.bot.sessions.clear_session",
            new=AsyncMock(),
        ) as mock_clear:
            await _handle_new(interaction)

        mock_clear.assert_not_awaited()
        interaction.response.send_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_redis_failure_sends_fallback_message(self):
        """A Redis outage during /new must never leave the interaction
        hanging. We must send an ephemeral fallback instead of bubbling up."""
        interaction = _interaction(guild=False)
        interaction.channel = MagicMock()
        interaction.channel_id = 999
        with patch(
            "backend.copilot.bot.sessions.clear_session",
            new=AsyncMock(side_effect=RuntimeError("redis down")),
        ):
            await _handle_new(interaction)

        interaction.response.send_message.assert_awaited_once()
        sent = interaction.response.send_message.await_args
        assert sent.kwargs["ephemeral"] is True
        assert "try again" in sent.args[0].lower()


class TestHandleLeave:
    @pytest.mark.asyncio
    async def test_thread_unsubscribes(self):
        interaction = _interaction()
        interaction.channel = MagicMock(spec=discord.Thread)
        interaction.channel.id = 777
        with patch(
            "backend.copilot.bot.threads.unsubscribe",
            new=AsyncMock(),
        ) as mock_unsub:
            await _handle_leave(interaction)

        mock_unsub.assert_awaited_once_with("discord", "777")
        interaction.response.send_message.assert_awaited_once()
        sent = interaction.response.send_message.await_args
        assert sent.kwargs["ephemeral"] is True
        assert "quiet" in sent.args[0].lower()

    @pytest.mark.asyncio
    async def test_plain_channel_is_rejected(self):
        # Channels aren't subscribed in the first place — /leave only makes
        # sense in a thread we'd otherwise auto-reply in.
        interaction = _interaction()
        interaction.channel = MagicMock()
        with patch(
            "backend.copilot.bot.threads.unsubscribe",
            new=AsyncMock(),
        ) as mock_unsub:
            await _handle_leave(interaction)

        mock_unsub.assert_not_awaited()
        interaction.response.send_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_dm_is_rejected(self):
        interaction = _interaction(guild=False)
        interaction.channel = MagicMock()
        with patch(
            "backend.copilot.bot.threads.unsubscribe",
            new=AsyncMock(),
        ) as mock_unsub:
            await _handle_leave(interaction)

        mock_unsub.assert_not_awaited()
        interaction.response.send_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_redis_failure_sends_fallback_message(self):
        interaction = _interaction()
        interaction.channel = MagicMock(spec=discord.Thread)
        interaction.channel.id = 777
        with patch(
            "backend.copilot.bot.threads.unsubscribe",
            new=AsyncMock(side_effect=RuntimeError("redis down")),
        ):
            await _handle_leave(interaction)

        interaction.response.send_message.assert_awaited_once()
        sent = interaction.response.send_message.await_args
        assert sent.kwargs["ephemeral"] is True
        assert "try again" in sent.args[0].lower()


def _chat(session_id: str, title: str | None = "A chat") -> ChatSummary:
    return ChatSummary(
        session_id=session_id,
        title=title,
        updated_at=datetime.now(timezone.utc),
    )


class TestHandleResume:
    @pytest.mark.asyncio
    async def test_rejected_in_a_server(self):
        interaction = _interaction()
        api = MagicMock()
        api.list_user_chats = AsyncMock()
        await _handle_resume(interaction, api)

        interaction.response.send_message.assert_awaited_once()
        api.list_user_chats.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_chats_shows_message(self):
        interaction = _interaction(guild=False)
        api = MagicMock()
        api.list_user_chats = AsyncMock(return_value=[])
        await _handle_resume(interaction, api)

        interaction.response.defer.assert_awaited_once()
        sent = interaction.followup.send.await_args
        assert "view" not in sent.kwargs or sent.kwargs.get("view") is None
        assert "conversations" in sent.args[0]

    @pytest.mark.asyncio
    async def test_lists_chats_in_a_picker(self):
        interaction = _interaction(guild=False)
        api = MagicMock()
        api.list_user_chats = AsyncMock(return_value=[_chat("s1"), _chat("s2")])
        await _handle_resume(interaction, api)

        interaction.response.defer.assert_awaited_once()
        api.list_user_chats.assert_awaited_once_with("discord", "456")
        sent = interaction.followup.send.await_args
        assert sent.kwargs["view"] is not None
        assert sent.kwargs["ephemeral"] is True

    @pytest.mark.asyncio
    async def test_unlinked_user_gets_guidance(self):
        interaction = _interaction(guild=False)
        api = MagicMock()
        api.list_user_chats = AsyncMock(side_effect=ValueError("not linked"))
        await _handle_resume(interaction, api)

        interaction.response.defer.assert_awaited_once()
        sent = interaction.followup.send.await_args
        assert "linked" in sent.args[0].lower()

    @pytest.mark.asyncio
    async def test_resume_to_session_sets_the_dm_session_cache(self):
        interaction = _interaction(guild=False)
        interaction.channel_id = 999
        with patch(
            "backend.copilot.bot.sessions.set_session",
            new=AsyncMock(),
        ) as mock_set:
            await _resume_to_session(interaction, "sess-pick")

        mock_set.assert_awaited_once_with("discord", "999", "sess-pick")
        interaction.response.send_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_resume_to_session_bails_when_channel_id_missing(self):
        """If discord.py hands us a None channel_id, persisting to
        `copilot-bot:session:discord:None` would silently break resume. We
        must refuse the action and tell the user."""
        interaction = _interaction(guild=False)
        interaction.channel_id = None
        with patch(
            "backend.copilot.bot.sessions.set_session",
            new=AsyncMock(),
        ) as mock_set:
            await _resume_to_session(interaction, "sess-pick")

        mock_set.assert_not_awaited()
        interaction.response.send_message.assert_awaited_once()
        sent = interaction.response.send_message.await_args
        assert sent.kwargs["ephemeral"] is True

    def test_resume_select_caps_options_at_discord_limit(self):
        """Discord's select menu hard-caps at 25 options. _ResumeSelect must
        trim its input so a long chat history doesn't break the picker."""
        from .commands import _ResumeSelect

        too_many = [_chat(f"s{i}", f"Chat {i}") for i in range(40)]
        select = _ResumeSelect(too_many)

        assert len(select.options) == _ResumeSelect.DISCORD_SELECT_OPTION_LIMIT
