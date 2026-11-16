"""``/autogpt`` — ask Otto or one of your experts, with an optional mode.

An interaction carries no message to thread off, so the command posts a
visible starter message ("**Bently** asked Bolt: ...") and hands that to the
core handler like any other message. Expert picks are DM-only.
"""

import logging
from typing import Awaitable, Callable

import discord
from discord import app_commands

from backend.copilot.bot.bot_backend import BotBackend
from backend.copilot.bot.expert_choice import (
    SERVER_EXPERTS_REPLY,
    match_experts,
    no_single_match_reply,
)
from backend.copilot.bot.options import MODE_TIERS, TurnOptions
from backend.platform_linking.models import BotExpert

logger = logging.getLogger(__name__)

# Discord caps autocomplete at 25 choices and a choice label at 100 chars.
AUTOCOMPLETE_LIMIT = 25
CHOICE_LABEL_LIMIT = 100
# Room for the "**name** asked X: " prefix under Discord's 2000-char cap.
STARTER_PROMPT_LIMIT = 1800
DISPATCH_FAILED_REPLY = "Otto couldn't start on that. Try again in a moment."

MODE_CHOICES = [
    app_commands.Choice(name="fast", value="fast"),
    app_commands.Choice(name="smart", value="smart"),
]

CommandDispatch = Callable[
    [discord.Interaction, discord.Message, str, TurnOptions], Awaitable[None]
]


def register(
    tree: app_commands.CommandTree, api: BotBackend, dispatch: CommandDispatch
) -> None:
    """Register ``/autogpt``; ``dispatch`` runs the turn off the starter."""

    async def expert_autocomplete(
        interaction: discord.Interaction, current: str
    ) -> list[app_commands.Choice[str]]:
        return await expert_choices(api, interaction, current)

    @tree.command(name="autogpt", description="Ask AutoGPT, or one of your experts")
    @app_commands.describe(
        prompt="What you want to ask",
        expert="One of your experts (DMs only); leave empty for Otto",
        mode="fast (default) or smart (the most capable model)",
    )
    @app_commands.choices(mode=MODE_CHOICES)
    @app_commands.autocomplete(expert=expert_autocomplete)
    async def autogpt_command(
        interaction: discord.Interaction,
        prompt: str,
        expert: str | None = None,
        mode: app_commands.Choice[str] | None = None,
    ) -> None:
        api.track_event(
            platform="discord",
            event_type="command_used",
            server_id=str(interaction.guild.id) if interaction.guild else None,
            command_name="autogpt",
        )
        await handle_autogpt(
            api, interaction, dispatch, prompt, expert, mode.value if mode else None
        )


async def expert_choices(
    api: BotBackend, interaction: discord.Interaction, current: str
) -> list[app_commands.Choice[str]]:
    """The invoker's experts whose name contains ``current``; none in a
    server, where experts can't be used."""
    if interaction.guild is not None:
        return []
    try:
        experts = await api.list_user_experts("discord", str(interaction.user.id), None)
    except Exception:
        logger.warning("Expert autocomplete failed", exc_info=True)
        return []
    wanted = current.casefold()
    return [
        app_commands.Choice(name=f"{e.name}, {e.role}"[:CHOICE_LABEL_LIMIT], value=e.id)
        for e in experts
        if wanted in e.name.casefold()
    ][:AUTOCOMPLETE_LIMIT]


async def handle_autogpt(
    api: BotBackend,
    interaction: discord.Interaction,
    dispatch: CommandDispatch,
    prompt: str,
    expert: str | None,
    mode: str | None,
) -> None:
    # Defer first: the expert lookup is an RPC that can outlast Discord's
    # 3-second ack window. Ephemeral, so refusals stay private.
    await interaction.response.defer(ephemeral=True, thinking=True)
    chosen: BotExpert | None = None
    if expert:
        if interaction.guild is not None:
            await interaction.followup.send(SERVER_EXPERTS_REPLY, ephemeral=True)
            return
        chosen, problem = await _pick_expert(api, interaction, expert)
        if chosen is None:
            await interaction.followup.send(problem, ephemeral=True)
            return
    starter = await _post_starter(interaction, prompt, chosen)
    if starter is None:
        await interaction.followup.send(
            "I can't post in this channel. Check my permissions and try again.",
            ephemeral=True,
        )
        return
    await _drop_thinking(interaction)
    options = TurnOptions(
        expert_id=chosen.id if chosen else None,
        expert_name=chosen.name if chosen else None,
        model=MODE_TIERS.get(mode) if mode else None,
    )
    try:
        await dispatch(interaction, starter, prompt, options)
    except Exception:
        logger.exception("Dispatching /autogpt failed")
        await interaction.followup.send(DISPATCH_FAILED_REPLY, ephemeral=True)


async def _pick_expert(
    api: BotBackend, interaction: discord.Interaction, value: str
) -> tuple[BotExpert | None, str]:
    """The expert an autocomplete id, or a typed name, refers to."""
    try:
        experts = await api.list_user_experts("discord", str(interaction.user.id), None)
    except ValueError:
        return None, (
            "Your DMs aren't linked to an AutoGPT account yet. "
            "Send me a message to get started."
        )
    except Exception:
        logger.exception("Failed to list experts for /autogpt")
        return None, "Couldn't load your experts right now. Try again soon."
    by_id = [e for e in experts if e.id == value]
    matches = by_id or match_experts(experts, value)
    if len(matches) == 1:
        return matches[0], ""
    return None, no_single_match_reply(value, experts, matches)


def starter_text(username: str, prompt: str, expert: BotExpert | None) -> str:
    shown = (
        prompt
        if len(prompt) <= STARTER_PROMPT_LIMIT
        else (prompt[:STARTER_PROMPT_LIMIT] + "…")
    )
    return f"**{username}** asked {expert.name if expert else 'Otto'}: {shown}"


async def _post_starter(
    interaction: discord.Interaction, prompt: str, expert: BotExpert | None
) -> discord.Message | None:
    channel = interaction.channel
    if not isinstance(channel, discord.abc.Messageable):
        return None
    try:
        return await channel.send(
            starter_text(interaction.user.display_name, prompt, expert)
        )
    except discord.HTTPException:
        logger.warning("Couldn't post the /autogpt starter message", exc_info=True)
        return None


async def _drop_thinking(interaction: discord.Interaction) -> None:
    """The starter message is the visible answer; clear the private
    'thinking...' placeholder so it doesn't linger."""
    try:
        await interaction.delete_original_response()
    except discord.HTTPException:
        logger.debug("Couldn't delete the /autogpt placeholder", exc_info=True)
