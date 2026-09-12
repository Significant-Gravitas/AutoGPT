"""Native Discord buttons for ask_question, and the click handler that turns
a press into an ordinary inbound message.

Kept out of ``adapter.py`` (already large) — the only thing the adapter needs
from here is ``build_choice_view`` and ``register_choice_handler``.

The buttons are **stateless**, like Slack's, Telegram's and Teams'. Everything
needed to resolve a click — the ``bot.choices`` token and the option index —
is encoded in each button's ``custom_id`` and parsed back out on click, so no
per-message state lives in this process. A view held in memory instead would
go dead on every deploy and once discord.py's own view timeout fired, and a
click on those dead buttons produces Discord's generic "This interaction
failed" while the Redis token stays valid for the rest of its hour.
"""

import logging
import re

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
_NOT_YOUR_QUESTION = (
    "This question was for someone else — they still need to answer it."
)

# Set once by `register_choice_handler`. The click handler is reconstructed by
# discord.py from the custom_id alone (possibly in a process that never sent
# the message), so it has no closure to read these from.
_adapter: PlatformAdapter | None = None
_on_message: MessageCallback | None = None


def register_choice_handler(
    client: discord.Client, adapter: PlatformAdapter, on_message: MessageCallback
) -> None:
    """Wire clicks on choice buttons to ``on_message`` for this process.

    Registered once at startup rather than per message: that is what lets a
    button posted before a restart still work afterwards.
    """
    global _adapter, _on_message
    _adapter = adapter
    _on_message = on_message
    client.add_dynamic_items(_ChoiceButton)


def build_choice_view(token: str, options: list[str]) -> discord.ui.View:
    """One button per option (View auto-wraps into rows of 5).

    ``timeout=None`` because the buttons carry their own state: expiry is the
    Redis token's job, and an expired token gives the user the notice above
    rather than Discord's generic failure.
    """
    view = discord.ui.View(timeout=None)
    for index, option in enumerate(options):
        view.add_item(_ChoiceButton.for_option(token, index, option))
    return view


class _ChoiceButton(
    discord.ui.DynamicItem[discord.ui.Button],
    # Deliberately wider than today's token grammar (`uuid4().hex[:12]`):
    # this pattern is what routes a click on a button that may have been
    # posted weeks ago, so a future change to how tokens are minted must not
    # silently orphan every live button.
    template=r"qans:(?P<token>[0-9A-Za-z_-]{1,64}):(?P<index>[0-9]{1,3})",
):
    def __init__(self, token: str, index: int, label: str = "") -> None:
        self._token = token
        self._index = index
        super().__init__(
            discord.ui.Button(
                style=discord.ButtonStyle.secondary,
                label=label[:80],
                custom_id=f"qans:{token}:{index}",
            )
        )

    @classmethod
    def for_option(cls, token: str, index: int, label: str) -> "_ChoiceButton":
        return cls(token, index, label)

    @classmethod
    async def from_custom_id(
        cls,
        interaction: discord.Interaction,
        item: discord.ui.Item[discord.ui.View],
        match: re.Match[str],
        /,
    ) -> "_ChoiceButton":
        # Rebuilt from the click alone — the label is whatever Discord still
        # renders on the message, so it is not needed here.
        return cls(match["token"], int(match["index"]))

    async def callback(self, interaction: discord.Interaction) -> None:
        if _adapter is None or _on_message is None:
            logger.error("Choice button clicked before the handler was registered")
            return
        resolved = await choices.resolve_choice(
            "discord", self._token, self._index, str(interaction.user.id)
        )
        if resolved.text is None:
            await interaction.response.send_message(
                _NOT_YOUR_QUESTION if resolved.refused else _EXPIRED_NOTICE,
                ephemeral=True,
            )
            return
        option = resolved.text
        # The token is already consumed, so the answer exists only in this
        # call. The ack is cosmetic (a 404 on a deleted message, or 40060 on
        # a fast double-click) and must never cost the user their answer.
        try:
            await interaction.response.edit_message(
                content=f"✅ You answered: {option}", view=None
            )
        except discord.HTTPException:
            logger.exception("Failed to acknowledge choice click; continuing the turn")
        ctx = _context_from_interaction(interaction, option)
        if ctx is not None:
            await _on_message(ctx, _adapter)


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
