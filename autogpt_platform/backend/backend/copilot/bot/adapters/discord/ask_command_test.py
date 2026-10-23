"""Tests for ``/autogpt``: autocomplete, starter message, hand-off."""

from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest
from discord import app_commands

from backend.copilot.bot.adapters.base import MessageHistoryEntry
from backend.copilot.bot.bot_backend import BotBackend
from backend.copilot.bot.expert_choice import SERVER_EXPERTS_REPLY
from backend.copilot.bot.options import TurnOptions
from backend.platform_linking.models import BotExpert

from .adapter import DiscordAdapter
from .ask_command import (
    AUTOCOMPLETE_LIMIT,
    DISPATCH_FAILED_REPLY,
    STARTER_PROMPT_LIMIT,
    expert_choices,
    handle_autogpt,
    register,
    starter_text,
)

BOLT = BotExpert(id="e1", name="Bolt", role="Builder")
SAGE = BotExpert(id="e2", name="Sage", role="Researcher")


def _interaction(*, guild: bool = False, thread: bool = False) -> MagicMock:
    interaction = MagicMock()
    interaction.response.defer = AsyncMock()
    interaction.followup.send = AsyncMock()
    interaction.delete_original_response = AsyncMock()
    interaction.user = MagicMock(id=456, display_name="Bently")
    if guild:
        interaction.guild = MagicMock(id=123)
        spec = discord.Thread if thread else discord.TextChannel
    else:
        interaction.guild = None
        spec = discord.DMChannel
    interaction.channel = MagicMock(spec=spec)
    starter = MagicMock(id=999)
    starter.channel = MagicMock(id=789)
    interaction.channel.send = AsyncMock(return_value=starter)
    return interaction


def _api(experts: list[BotExpert] | None = None) -> MagicMock:
    api = MagicMock()
    api.list_user_experts = AsyncMock(return_value=experts or [])
    return api


class TestExpertChoices:
    @pytest.mark.asyncio
    async def test_dm_offers_matching_experts_labelled_with_role(self):
        api = _api([BOLT, SAGE])
        choices = await expert_choices(api, _interaction(), "bo")
        assert choices == [app_commands.Choice(name="Bolt, Builder", value="e1")]
        api.list_user_experts.assert_awaited_once_with("discord", "456", None)

    @pytest.mark.asyncio
    async def test_caps_at_discord_limit(self):
        many = [BotExpert(id=f"e{i}", name=f"X{i}", role="r") for i in range(40)]
        choices = await expert_choices(_api(many), _interaction(), "")
        assert len(choices) == AUTOCOMPLETE_LIMIT

    @pytest.mark.asyncio
    async def test_server_offers_none(self):
        api = _api([BOLT])
        assert await expert_choices(api, _interaction(guild=True), "") == []
        api.list_user_experts.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_lookup_failure_offers_none(self):
        api = MagicMock(list_user_experts=AsyncMock(side_effect=RuntimeError("x")))
        assert await expert_choices(api, _interaction(), "") == []


class TestHandleAutogpt:
    @pytest.mark.asyncio
    async def test_dm_expert_and_mode_post_starter_and_dispatch(self):
        interaction = _interaction()
        dispatch = AsyncMock()
        await handle_autogpt(_api([BOLT]), interaction, dispatch, "hi", "e1", "smart")

        interaction.response.defer.assert_awaited_once_with(
            ephemeral=True, thinking=True
        )
        interaction.channel.send.assert_awaited_once_with("**Bently** asked Bolt: hi")
        interaction.delete_original_response.assert_awaited_once()
        _, starter, prompt, options = dispatch.await_args.args
        assert starter.id == 999 and prompt == "hi"
        assert options == TurnOptions(
            expert_id="e1", expert_name="Bolt", model="advanced"
        )

    @pytest.mark.asyncio
    async def test_typed_name_resolves_too(self):
        dispatch = AsyncMock()
        await handle_autogpt(
            _api([BOLT, SAGE]), _interaction(), dispatch, "hi", "sage", None
        )
        assert dispatch.await_args.args[3].expert_id == "e2"

    @pytest.mark.asyncio
    async def test_no_expert_asks_otto_in_a_server(self):
        interaction = _interaction(guild=True)
        dispatch = AsyncMock()
        await handle_autogpt(_api(), interaction, dispatch, "hi", None, "fast")
        interaction.channel.send.assert_awaited_once_with("**Bently** asked Otto: hi")
        assert dispatch.await_args.args[3] == TurnOptions(model="standard")

    @pytest.mark.asyncio
    async def test_server_expert_refused_privately(self):
        interaction = _interaction(guild=True)
        dispatch = AsyncMock()
        api = _api([BOLT])
        await handle_autogpt(api, interaction, dispatch, "hi", "e1", None)
        interaction.followup.send.assert_awaited_once_with(
            SERVER_EXPERTS_REPLY, ephemeral=True
        )
        interaction.channel.send.assert_not_awaited()
        dispatch.assert_not_awaited()
        api.list_user_experts.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unknown_expert_refused_privately(self):
        interaction = _interaction()
        dispatch = AsyncMock()
        await handle_autogpt(_api([BOLT]), interaction, dispatch, "hi", "Nova", None)
        text = interaction.followup.send.await_args.args[0]
        assert 'No expert called "Nova"' in text
        dispatch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unlinked_dm_gets_link_hint(self):
        interaction = _interaction()
        api = MagicMock(list_user_experts=AsyncMock(side_effect=ValueError("nope")))
        await handle_autogpt(api, interaction, AsyncMock(), "hi", "e1", None)
        assert "aren't linked" in interaction.followup.send.await_args.args[0]

    @pytest.mark.asyncio
    async def test_starter_post_failure_tells_the_user(self):
        interaction = _interaction(guild=True)
        interaction.channel.send = AsyncMock(
            side_effect=discord.HTTPException(MagicMock(status=403), "no")
        )
        dispatch = AsyncMock()
        await handle_autogpt(_api(), interaction, dispatch, "hi", None, None)
        assert "can't post" in interaction.followup.send.await_args.args[0]
        dispatch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dispatch_failure_tells_the_user_privately(self):
        interaction = _interaction(guild=True)
        dispatch = AsyncMock(side_effect=RuntimeError("boom"))
        await handle_autogpt(_api(), interaction, dispatch, "hi", None, None)
        interaction.followup.send.assert_awaited_once_with(
            DISPATCH_FAILED_REPLY, ephemeral=True
        )


def test_starter_text_clamps_long_prompts():
    text = starter_text("Bently", "x" * (STARTER_PROMPT_LIMIT + 50), None)
    assert text.startswith("**Bently** asked Otto: ")
    assert len(text) < 2000


def test_register_adds_the_autogpt_command():
    client = MagicMock()
    client._connection._command_tree = None
    tree = app_commands.CommandTree(client)
    register(tree, cast(BotBackend, MagicMock()), AsyncMock())
    command = tree.get_command("autogpt")
    assert command is not None
    assert [p.name for p in command.parameters] == ["prompt", "expert", "mode"]
    assert [p.required for p in command.parameters] == [True, False, False]


class TestDispatchCommand:
    def _adapter(self) -> tuple[DiscordAdapter, AsyncMock]:
        adapter = DiscordAdapter.__new__(DiscordAdapter)
        callback = AsyncMock()
        adapter._on_message_callback = callback
        return adapter, callback

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "guild,thread,channel_type",
        [(False, False, "dm"), (True, False, "channel"), (True, True, "thread")],
    )
    async def test_builds_context_from_the_starter(self, guild, thread, channel_type):
        adapter, callback = self._adapter()
        interaction = _interaction(guild=guild, thread=thread)
        starter = await interaction.channel.send("x")
        options = TurnOptions(model="advanced")

        await adapter._dispatch_command(interaction, starter, "hi", options)

        ctx = callback.await_args.args[0]
        assert ctx.channel_type == channel_type
        assert ctx.message_id == "999" and ctx.channel_id == "789"
        assert ctx.server_id == ("123" if guild else None)
        assert ctx.text == "hi" and ctx.options == options
        assert ctx.bot_mentioned is True
        assert ctx.user_id == "456" and ctx.username == "Bently"

    @pytest.mark.asyncio
    async def test_a_command_in_a_thread_keeps_the_location(self):
        adapter, callback = self._adapter()
        interaction = _interaction(guild=True, thread=True)
        interaction.guild.name = "SG"
        thread = MagicMock(spec=discord.Thread)
        thread.id = 789
        thread.name = "Bug triage"
        thread.parent_id = 222
        thread.parent = MagicMock(spec=discord.TextChannel)
        thread.parent.name = "eng-general"
        starter = MagicMock(id=999)
        starter.channel = thread

        with patch.object(adapter, "_thread_history", new=AsyncMock(return_value=())):
            await adapter._dispatch_command(interaction, starter, "hi", TurnOptions())

        ctx = callback.await_args.args[0]
        assert ctx.server_name == "SG"
        assert ctx.channel_name == "eng-general"
        assert ctx.public_channel_id == "222"
        assert ctx.thread_name == "Bug triage"
        assert ctx.public_thread_id == "789"
        assert ctx.post_channel_id == "789"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("thread", [True, False])
    async def test_a_command_in_a_thread_carries_its_history(self, thread):
        adapter, callback = self._adapter()
        interaction = _interaction(guild=True, thread=thread)
        starter = await interaction.channel.send("x")
        earlier = (MessageHistoryEntry(username="Ana", user_id="1", text="the bug"),)
        fetch = AsyncMock(return_value=earlier)

        with patch.object(adapter, "_thread_history", new=fetch):
            await adapter._dispatch_command(interaction, starter, "hi", TurnOptions())

        ctx = callback.await_args.args[0]
        if thread:
            fetch.assert_awaited_once_with(starter)
            assert ctx.thread_history == earlier
        else:
            fetch.assert_not_awaited()
            assert ctx.thread_history == ()
