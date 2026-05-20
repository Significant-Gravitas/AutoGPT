"""Discord slash command handlers.

Registered on the bot's CommandTree at startup. All responses are ephemeral
(visible only to the invoking user) to keep channels clean and to keep link
URLs private.
"""

import logging

import discord
from discord import app_commands

from backend.copilot.bot.bot_backend import BotBackend
from backend.util.exceptions import LinkAlreadyExistsError
from backend.util.settings import Settings

logger = logging.getLogger(__name__)


def register(tree: app_commands.CommandTree, api: BotBackend) -> None:
    """Register all slash commands on the given CommandTree."""

    @tree.command(
        name="setup",
        description="Link this server to an AutoGPT account for AutoPilot",
    )
    @app_commands.default_permissions(manage_guild=True)
    async def setup_command(interaction: discord.Interaction) -> None:
        await _handle_setup(interaction, api)

    @tree.command(name="help", description="Show AutoPilot bot usage info")
    async def help_command(interaction: discord.Interaction) -> None:
        await _handle_help(interaction)

    @tree.command(
        name="unlink",
        description="Manage linked servers from your AutoGPT settings",
    )
    async def unlink_command(interaction: discord.Interaction) -> None:
        await _handle_unlink(interaction)


async def _handle_setup(interaction: discord.Interaction, api: BotBackend) -> None:
    if interaction.guild is None:
        await interaction.response.send_message(
            "This command can only be used in a server. "
            "To link your DMs, just send me a direct message.",
            ephemeral=True,
        )
        return

    if not interaction.permissions.manage_guild:
        await interaction.response.send_message(
            "Only members with the Manage Server permission can link this "
            "server to an AutoGPT account.",
            ephemeral=True,
        )
        return

    await interaction.response.defer(ephemeral=True)
    try:
        result = await api.create_link_token(
            platform="discord",
            platform_server_id=str(interaction.guild.id),
            platform_user_id=str(interaction.user.id),
            platform_username=interaction.user.display_name,
            server_name=interaction.guild.name,
            channel_id=str(interaction.channel_id or ""),
        )
    except LinkAlreadyExistsError:
        await interaction.followup.send(
            "This server is already linked — just mention me!",
            ephemeral=True,
        )
        return
    except Exception:
        logger.exception("Failed to create link token")
        await interaction.followup.send(
            "Something went wrong. Try again later.",
            ephemeral=True,
        )
        return

    view = discord.ui.View()
    view.add_item(
        discord.ui.Button(
            style=discord.ButtonStyle.link,
            label="Link Server",
            url=result.link_url,
        )
    )
    await interaction.followup.send(
        f"**Set up AutoPilot for {interaction.guild.name}**\n\n"
        "Click the button below to connect this server to your AutoGPT "
        "account. Once confirmed, everyone here can mention me to use "
        "AutoPilot.\n\n"
        "All usage will be billed to your account.\n"
        "This link expires in 30 minutes.",
        ephemeral=True,
        view=view,
    )


async def _handle_help(interaction: discord.Interaction) -> None:
    await interaction.response.send_message(
        "**AutoPilot Bot**\n\n"
        "Mention me in a server or DM me directly to chat.\n\n"
        "**Commands:**\n"
        "- `/setup` — Link this server to your AutoGPT account\n"
        "- `/help` — Show this message\n"
        "- `/unlink` — Manage linked accounts\n\n"
        "**How it works:**\n"
        "- In a server: the person who runs `/setup` pays for usage\n"
        "- In DMs: you link and pay for your own usage\n",
        ephemeral=True,
    )


async def _handle_unlink(interaction: discord.Interaction) -> None:
    config = Settings().config
    base_url = config.frontend_base_url or config.platform_base_url
    message = (
        "Unlinking requires authentication, so it has to be done "
        "from the web. Click below to manage your linked accounts."
    )

    if not base_url:
        await interaction.response.send_message(
            f"{message}\n\nOpen your AutoGPT settings and visit "
            "Profile → Linked accounts.",
            ephemeral=True,
        )
        return

    view = discord.ui.View()
    view.add_item(
        discord.ui.Button(
            style=discord.ButtonStyle.link,
            label="Open Settings",
            url=f"{base_url}/profile/settings",
        )
    )
    await interaction.response.send_message(message, ephemeral=True, view=view)
