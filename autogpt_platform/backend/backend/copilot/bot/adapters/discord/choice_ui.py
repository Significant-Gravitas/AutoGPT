"""Native Discord buttons for ask_question, and the click handler that turns
a press into an ordinary inbound message.

Kept out of ``adapter.py`` (already large) — the only thing the adapter needs
from here is ``build_choice_view``.
"""

import logging

import discord

from backend.copilot.bot import choices
from backend.copilot.bot.adapters.base import (
    ChannelType,
    MessageCallback,
    MessageContext,
    PlatformAdapter,
)

logger = logging.getLogger(__name__)

_EXPIRED_NOTICE = "This question has expired — type your answer instead."


def build_choice_view(
    adapter: PlatformAdapter,
    on_message: MessageCallback,
    token: str,
    options: list[str],
) -> discord.ui.View:
    """One button per option (View auto-wraps into rows of 5); a click
    resolves the answer via ``bot.choices`` and feeds it through
    ``on_message``, exactly like a normal typed reply."""
    view = discord.ui.View(timeout=choices.CHOICE_TTL)
    for index, option in enumerate(options):
        view.add_item(_ChoiceButton(adapter, on_message, token, index, option))
    return view


class _ChoiceButton(discord.ui.Button):
    def __init__(
        self,
        adapter: PlatformAdapter,
        on_message: MessageCallback,
        token: str,
        index: int,
        label: str,
    ) -> None:
        super().__init__(style=discord.ButtonStyle.secondary, label=label[:80])
        self._adapter = adapter
        self._on_message = on_message
        self._token = token
        self._index = index

    async def callback(self, interaction: discord.Interaction) -> None:
        option = await choices.resolve_choice("discord", self._token, self._index)
        if option is None:
            await interaction.response.send_message(_EXPIRED_NOTICE, ephemeral=True)
            return
        await interaction.response.edit_message(
            content=f"✅ You answered: {option}", view=None
        )
        ctx = _context_from_interaction(interaction, option)
        if ctx is not None:
            await self._on_message(ctx, self._adapter)


def _context_from_interaction(
    interaction: discord.Interaction, option: str
) -> MessageContext | None:
    if interaction.channel_id is None or interaction.user is None:
        return None
    channel_type: ChannelType = "channel"
    if interaction.guild_id is None:
        channel_type = "dm"
    elif isinstance(interaction.channel, discord.Thread):
        channel_type = "thread"
    return MessageContext(
        platform="discord",
        channel_type=channel_type,
        server_id=str(interaction.guild_id) if interaction.guild_id else None,
        channel_id=str(interaction.channel_id),
        message_id=str(interaction.id),
        user_id=str(interaction.user.id),
        username=interaction.user.display_name,
        text=option,
        bot_mentioned=True,
    )
