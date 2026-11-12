"""Where a Discord message or button click was sent, for the model's header."""

import discord
from pydantic import BaseModel


class Location(BaseModel):
    """Names and public ids for the location header the model sees."""

    server_name: str | None = None
    channel_name: str | None = None
    public_channel_id: str | None = None
    thread_name: str | None = None
    public_thread_id: str | None = None
    post_channel_id: str | None = None


def location_of(guild: discord.Guild | None, channel: object) -> Location:
    if guild is None or channel is None:
        return Location()
    channel_id = str(getattr(channel, "id"))
    location = Location(
        server_name=str(guild.name),
        public_channel_id=channel_id,
        post_channel_id=channel_id,
    )
    if isinstance(channel, discord.Thread):
        location.thread_name = str(channel.name)
        location.public_thread_id = channel_id
        location.public_channel_id = (
            str(channel.parent_id) if channel.parent_id else None
        )
        location.channel_name = str(channel.parent.name) if channel.parent else None
    elif isinstance(channel, discord.abc.GuildChannel):
        location.channel_name = str(channel.name)
    return location
