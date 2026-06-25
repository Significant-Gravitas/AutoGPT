"""Discord slash command handlers.

Registered on the bot's CommandTree at startup. All responses are ephemeral
(visible only to the invoking user) to keep channels clean and to keep link
URLs private.
"""

import logging
from datetime import datetime

import discord
from discord import app_commands

from backend.copilot.bot import sessions, threads
from backend.copilot.bot.bot_backend import BotBackend, ChatSummary
from backend.util.exceptions import LinkAlreadyExistsError
from backend.util.settings import Settings

logger = logging.getLogger(__name__)


def register(tree: app_commands.CommandTree, api: BotBackend) -> None:
    """Register all slash commands on the given CommandTree."""

    def _track(interaction: discord.Interaction, command_name: str) -> None:
        api.track_event(
            platform="discord",
            event_type="command_used",
            server_id=str(interaction.guild.id) if interaction.guild else None,
            command_name=command_name,
        )

    @tree.command(
        name="setup",
        description="Link this server to an AutoGPT account for AutoPilot",
    )
    @app_commands.default_permissions(manage_guild=True)
    async def setup_command(interaction: discord.Interaction) -> None:
        _track(interaction, "setup")
        await _handle_setup(interaction, api)

    @tree.command(name="help", description="Show AutoPilot bot usage info")
    async def help_command(interaction: discord.Interaction) -> None:
        _track(interaction, "help")
        await _handle_help(interaction)

    @tree.command(
        name="unlink",
        description="Manage linked servers from your AutoGPT settings",
    )
    async def unlink_command(interaction: discord.Interaction) -> None:
        _track(interaction, "unlink")
        await _handle_unlink(interaction)

    @tree.command(
        name="new",
        description="Start a fresh AutoPilot conversation in this DM or thread",
    )
    async def new_command(interaction: discord.Interaction) -> None:
        _track(interaction, "new")
        await _handle_new(interaction)

    @tree.command(
        name="resume",
        description="Resume one of your past AutoPilot conversations (DMs only)",
    )
    async def resume_command(interaction: discord.Interaction) -> None:
        _track(interaction, "resume")
        await _handle_resume(interaction, api)

    @tree.command(
        name="leave",
        description="Stop AutoPilot from auto-replying in this thread",
    )
    async def leave_command(interaction: discord.Interaction) -> None:
        _track(interaction, "leave")
        await _handle_leave(interaction)


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
        "- `/new` — Start a fresh conversation (DMs & threads)\n"
        "- `/resume` — Resume a past conversation (DMs only)\n"
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
            f"{message}\n\nOpen AutoGPT on the web and go to " "Settings → Bots.",
            ephemeral=True,
        )
        return

    view = discord.ui.View()
    view.add_item(
        discord.ui.Button(
            style=discord.ButtonStyle.link,
            label="Open Settings",
            url=f"{base_url}/settings/bots",
        )
    )
    await interaction.response.send_message(message, ephemeral=True, view=view)


async def _handle_new(interaction: discord.Interaction) -> None:
    is_dm = interaction.guild is None
    is_thread = isinstance(interaction.channel, discord.Thread)
    if not is_dm and not is_thread:
        await interaction.response.send_message(
            "Use `/new` in a DM or a thread. Mentioning me in a channel "
            "already starts a fresh thread.",
            ephemeral=True,
        )
        return

    try:
        await sessions.clear_session("discord", str(interaction.channel_id))
    except Exception:
        logger.exception("Failed to clear copilot session for /new")
        await interaction.response.send_message(
            "Couldn't reset the conversation right now. Please try again in a moment.",
            ephemeral=True,
        )
        return

    await interaction.response.send_message(
        "Started a fresh conversation — send a message to begin.",
        ephemeral=True,
    )


async def _handle_leave(interaction: discord.Interaction) -> None:
    if not isinstance(interaction.channel, discord.Thread):
        await interaction.response.send_message(
            "Use `/leave` inside a thread. I only auto-reply in threads I'm "
            "subscribed to.",
            ephemeral=True,
        )
        return

    try:
        await threads.unsubscribe("discord", str(interaction.channel.id))
    except Exception:
        logger.exception("Failed to unsubscribe thread for /leave")
        await interaction.response.send_message(
            "Couldn't step out of this thread right now. Please try again in a moment.",
            ephemeral=True,
        )
        return

    await interaction.response.send_message(
        "Going quiet in this thread. @mention me to bring me back.",
        ephemeral=True,
    )


async def _handle_resume(interaction: discord.Interaction, api: BotBackend) -> None:
    if interaction.guild is not None:
        await interaction.response.send_message(
            "`/resume` only works in my DMs — your conversations are private "
            "to you. Send me a direct message to use it.",
            ephemeral=True,
        )
        return

    # Defer immediately — the linking-manager RPC can blow past Discord's
    # 3-second ack window on a cold cache, which 404s the interaction.
    await interaction.response.defer(ephemeral=True)

    try:
        chats = await api.list_user_chats("discord", str(interaction.user.id))
    except ValueError:
        # NotFoundError (DMs not linked) crosses RPC as its ValueError base.
        await interaction.followup.send(
            "Your DMs aren't linked to an AutoGPT account yet — send me a "
            "message to get started.",
            ephemeral=True,
        )
        return
    except Exception:
        logger.exception("Failed to list chats for /resume")
        await interaction.followup.send(
            "Something went wrong loading your conversations. Try again later.",
            ephemeral=True,
        )
        return

    if not chats:
        await interaction.followup.send(
            "You don't have any conversations yet — send me a message to start one.",
            ephemeral=True,
        )
        return

    await interaction.followup.send(
        "Pick a conversation to resume:",
        view=_ResumeView(chats),
        ephemeral=True,
    )


class _ResumeView(discord.ui.View):
    def __init__(self, chats: list[ChatSummary]) -> None:
        super().__init__(timeout=180)
        self.add_item(_ResumeSelect(chats))


class _ResumeSelect(discord.ui.Select):
    # Discord caps select menus at 25 options — silently breaks the picker
    # past that. _handle_resume already requests at most that many chats, but
    # we re-slice here so any future caller is safe.
    DISCORD_SELECT_OPTION_LIMIT = 25

    def __init__(self, chats: list[ChatSummary]) -> None:
        capped_chats = chats[: self.DISCORD_SELECT_OPTION_LIMIT]
        options = [
            discord.SelectOption(
                label=(chat.title or "Untitled conversation")[:100],
                value=chat.session_id,
                description=f"Last active {_relative_time(chat.updated_at)}"[:100],
            )
            for chat in capped_chats
        ]
        super().__init__(placeholder="Choose a conversation to resume", options=options)

    async def callback(self, interaction: discord.Interaction) -> None:
        await _resume_to_session(interaction, self.values[0])


async def _resume_to_session(interaction: discord.Interaction, session_id: str) -> None:
    # In a DM, discord.py types interaction.channel_id as Optional[int]; an
    # unset value would silently route the session into the wrong cache key.
    # Bail with an ephemeral error instead of corrupting state.
    if interaction.channel_id is None:
        logger.warning("/resume callback fired without channel_id; cannot resume")
        await interaction.response.send_message(
            "Couldn't resume that conversation right now. Please try `/resume` again.",
            ephemeral=True,
        )
        return

    await sessions.set_session("discord", str(interaction.channel_id), session_id)
    await interaction.response.send_message(
        "Resumed that conversation — send a message to continue it.",
        ephemeral=True,
    )


def _relative_time(when: datetime) -> str:
    seconds = max(0, int((datetime.now(when.tzinfo) - when).total_seconds()))
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"
