"""Native Discord buttons for ask_question, and the click handler that turns
a press into an ordinary inbound message.

Kept out of ``adapter.py`` (already large) — the only thing the adapter needs
from here is ``build_choice_view`` and ``register_choice_handler``.

The buttons are **stateless**, like Slack's, Telegram's and Teams'. Everything
needed to resolve a click — the button's kind, its token and the option index —
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
from backend.copilot.bot.bot_backend import BotBackend
from backend.copilot.bot.choices import QUESTION_KIND, ButtonKind

logger = logging.getLogger(__name__)

# Set once by `register_choice_handler`. The click handler is reconstructed by
# discord.py from the custom_id alone (possibly in a process that never sent
# the message), so it has no closure to read these from.
_adapter: PlatformAdapter | None = None
_on_message: MessageCallback | None = None
_api: BotBackend | None = None


def register_choice_handler(
    client: discord.Client,
    adapter: PlatformAdapter,
    on_message: MessageCallback,
    api: BotBackend,
) -> None:
    """Wire clicks on choice buttons to ``on_message`` for this process.

    Registered once at startup rather than per message: that is what lets a
    button posted before a restart still work afterwards.
    """
    global _adapter, _on_message, _api
    _adapter = adapter
    _on_message = on_message
    _api = api
    client.add_dynamic_items(_ChoiceButton)


def build_choice_view(
    token: str, options: list[str], kind: ButtonKind = QUESTION_KIND
) -> discord.ui.View:
    """One button per option (View auto-wraps into rows of 5).

    ``timeout=None`` because the buttons carry their own state: expiry is the
    Redis token's job, and an expired token gives the user a notice rather
    than Discord's generic failure.
    """
    view = discord.ui.View(timeout=None)
    for index, option in enumerate(options):
        view.add_item(_ChoiceButton(kind, token, index, option))
    return view


class _ChoiceButton(
    discord.ui.DynamicItem[discord.ui.Button],
    # Deliberately wider than today's token grammar (`uuid4().hex[:12]`):
    # this pattern is what routes a click on a button that may have been
    # posted weeks ago, so a future change to how tokens are minted must not
    # silently orphan every live button.
    template=r"(?P<kind>qans|appr):(?P<token>[0-9A-Za-z_-]{1,64}):(?P<index>[0-9]{1,3})",
):
    def __init__(
        self, kind: ButtonKind, token: str, index: int, label: str = ""
    ) -> None:
        self._kind: ButtonKind = kind
        self._token = token
        self._index = index
        primary = kind == choices.CARD_KIND and index == 0
        super().__init__(
            discord.ui.Button(
                style=(
                    discord.ButtonStyle.primary
                    if primary
                    else discord.ButtonStyle.secondary
                ),
                label=label[:80],
                custom_id=f"{kind}:{token}:{index}",
            )
        )

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
        return cls(
            choices.BUTTON_KINDS[match["kind"]], match["token"], int(match["index"])
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        if _adapter is None or _on_message is None or _api is None:
            logger.error("Choice button clicked before the handler was registered")
            return
        # Answering a card crosses two services and can outlast Discord's
        # three seconds to acknowledge a click.
        await interaction.response.defer()
        answer = await choices.answer_button(
            _api,
            "discord",
            self._kind,
            self._token,
            self._index,
            str(interaction.user.id),
            _server_id(interaction),
        )
        if not answer.answered:
            await interaction.followup.send(answer.text, ephemeral=True)
            return
        # The token is already consumed, so the answer exists only in this
        # call. The ack is cosmetic (a 404 on a deleted message, or 40060 on
        # a fast double-click) and must never cost the user their answer.
        try:
            await interaction.edit_original_response(content=answer.text, view=None)
        except discord.HTTPException:
            logger.exception("Failed to acknowledge choice click; continuing the turn")
        ctx = _context_from_interaction(interaction, answer.reply or "")
        if ctx is not None:
            ctx.follow = answer.follow
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
        server_id=_server_id(interaction),
        channel_id=str(interaction.channel_id),
        message_id=str(interaction.id),
        user_id=str(interaction.user.id),
        username=interaction.user.display_name,
        text=option,
        bot_mentioned=True,
    )


def _server_id(interaction: discord.Interaction) -> str | None:
    return str(interaction.guild_id) if interaction.guild_id else None
