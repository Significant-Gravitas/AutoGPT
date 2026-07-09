"""Welcome message posted when AutoPilot is added to a Discord server."""

import discord


def pick_intro_channel(guild: discord.Guild) -> discord.TextChannel | None:
    """Return the best channel to post the welcome message in.

    Prefer the server's system channel; fall back to the first text channel
    we can send in. ``None`` if there's nowhere we can post.
    """
    me = guild.me
    if me is None:
        return None
    system_channel = guild.system_channel
    if system_channel is not None and system_channel.permissions_for(me).send_messages:
        return system_channel
    for channel in guild.text_channels:
        if channel.permissions_for(me).send_messages:
            return channel
    return None


def intro_message() -> str:
    return (
        "**Hey, I'm AutoPilot — AutoGPT in your Discord.** Thanks for adding me!\n\n"
        "To get started, a server admin runs `/setup` and follows the link "
        "to connect this server to an AutoGPT account.\n\n"
        "Once linked, mention me anywhere I can read messages "
        "(e.g. `@AutoPilot summarise this channel`) and I'll spin up a "
        "thread for our chat.\n\n"
        "You can also **DM me** to chat 1:1 — send any message and I'll "
        "walk you through linking your own DMs.\n\n"
        "Commands: `/help` `/setup` `/new`"
    )
