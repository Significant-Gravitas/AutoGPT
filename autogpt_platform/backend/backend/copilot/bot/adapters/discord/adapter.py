"""Discord adapter — connects to the Discord Gateway and forwards messages.

Platform-specific machinery only: Gateway connection, message event handling,
thread creation, typing, button rendering. All platform-agnostic logic lives
in the core handler. Slash commands live in commands.py.
"""

import logging
from typing import Optional

import discord
from discord import app_commands

from backend.copilot.bot.bot_backend import BotBackend

from ..base import ChannelType, MessageCallback, MessageContext, PlatformAdapter
from . import commands, config

logger = logging.getLogger(__name__)


class DiscordAdapter(PlatformAdapter):
    def __init__(self, api: BotBackend):
        intents = discord.Intents.default()
        intents.message_content = True
        # AutoPilot output is untrusted w.r.t. mentions — suppress @everyone,
        # role, and user pings the LLM might produce. Client-level default
        # applies to every send() + reply() below.
        self._client = discord.Client(
            intents=intents,
            allowed_mentions=discord.AllowedMentions.none(),
        )
        self._tree = app_commands.CommandTree(self._client)
        self._api = api
        self._on_message_callback: Optional[MessageCallback] = None
        self._commands_synced = False

        self._register_events()
        commands.register(self._tree, self._api)

    @property
    def platform_name(self) -> str:
        return "discord"

    @property
    def max_message_length(self) -> int:
        return config.MAX_MESSAGE_LENGTH

    @property
    def chunk_flush_at(self) -> int:
        return config.CHUNK_FLUSH_AT

    def on_message(self, callback: MessageCallback) -> None:
        self._on_message_callback = callback

    async def start(self) -> None:
        await self._client.start(config.get_bot_token())

    async def stop(self) -> None:
        if not self._client.is_closed():
            await self._client.close()

    async def _resolve_channel(self, channel_id: str):
        """Return the channel for ``channel_id``, falling back to a REST fetch.

        ``Client.get_channel`` only reads the in-memory cache, so it misses
        threads the bot hasn't seen since its last restart. Fall back to
        ``fetch_channel`` (REST) so long-lived threads keep working.
        """
        channel = self._client.get_channel(int(channel_id))
        if channel is not None:
            return channel
        try:
            return await self._client.fetch_channel(int(channel_id))
        except (discord.NotFound, discord.Forbidden, discord.HTTPException):
            logger.warning("Channel %s not found or inaccessible", channel_id)
            return None

    async def send_message(self, channel_id: str, text: str) -> None:
        channel = await self._resolve_channel(channel_id)
        if channel and isinstance(channel, discord.abc.Messageable):
            # tts=False is the default but we pin it explicitly — AutoPilot
            # output is untrusted and should never blast through voice.
            await channel.send(text, tts=False)

    async def send_link(
        self, channel_id: str, text: str, link_label: str, link_url: str
    ) -> None:
        channel = await self._resolve_channel(channel_id)
        if channel is None or not isinstance(channel, discord.abc.Messageable):
            return
        view = discord.ui.View()
        view.add_item(
            discord.ui.Button(
                style=discord.ButtonStyle.link,
                label=link_label[:80],  # Discord button label max
                url=link_url,
            )
        )
        await channel.send(text, view=view, tts=False)

    async def send_reply(
        self, channel_id: str, text: str, reply_to_message_id: str
    ) -> None:
        channel = await self._resolve_channel(channel_id)
        if not channel or not isinstance(channel, discord.abc.Messageable):
            return
        try:
            msg = await channel.fetch_message(int(reply_to_message_id))
            await msg.reply(text, tts=False)
        except discord.NotFound:
            await channel.send(text, tts=False)

    async def send_ephemeral(self, channel_id: str, user_id: str, text: str) -> None:
        # Ephemeral messages are only possible via interaction responses.
        # Fall back to a normal message for non-interaction contexts.
        await self.send_message(channel_id, text)

    async def start_typing(self, channel_id: str) -> None:
        channel = await self._resolve_channel(channel_id)
        if channel and isinstance(channel, discord.abc.Messageable):
            await channel.typing()

    async def stop_typing(self, channel_id: str) -> None:
        pass  # Discord typing auto-expires after ~10s

    async def create_thread(
        self, channel_id: str, message_id: str, name: str
    ) -> Optional[str]:
        channel = await self._resolve_channel(channel_id)
        if channel is None or not isinstance(channel, discord.TextChannel):
            logger.warning("Cannot create thread in non-text channel %s", channel_id)
            return None
        try:
            msg = await channel.fetch_message(int(message_id))
            thread = await msg.create_thread(name=name[:100])
            return str(thread.id)
        except discord.HTTPException:
            logger.exception("Failed to create thread in channel %s", channel_id)
            return None

    # -- Internal --

    def _register_events(self) -> None:
        @self._client.event
        async def on_ready() -> None:
            logger.info(f"Discord bot connected as {self._client.user}")
            # Sync slash commands once per process — on_ready fires on every
            # gateway reconnect, but the command tree only needs uploading once.
            if self._commands_synced:
                return
            try:
                synced = await self._tree.sync()
                self._commands_synced = True
                logger.info(f"Synced {len(synced)} slash commands")
            except Exception:
                logger.exception("Failed to sync slash commands")

        @self._client.event
        async def on_message(message: discord.Message) -> None:
            if message.author.bot:
                return
            if self._on_message_callback is None:
                return

            channel_type = self._channel_type(message)

            # Channels require an explicit @mention; DMs and threads always forward
            # (handler checks thread subscription).
            if channel_type == "channel" and not self._is_mentioned(message):
                return

            ctx = MessageContext(
                platform="discord",
                channel_type=channel_type,
                server_id=str(message.guild.id) if message.guild else None,
                channel_id=str(message.channel.id),
                message_id=str(message.id),
                user_id=str(message.author.id),
                username=message.author.display_name,
                text=self._strip_mentions(message),
            )
            await self._on_message_callback(ctx, self)

    def _is_mentioned(self, message: discord.Message) -> bool:
        if message.guild is None:
            return True  # DMs always count
        return bool(self._client.user and self._client.user.mentioned_in(message))

    @staticmethod
    def _channel_type(message: discord.Message) -> ChannelType:
        if message.guild is None:
            return "dm"
        if isinstance(message.channel, discord.Thread):
            return "thread"
        return "channel"

    def _strip_mentions(self, message: discord.Message) -> str:
        """Strip the bot's own mention; replace other users' raw mention
        tokens with `@displayname` so the LLM keeps the context.
        """
        text = message.content
        bot_id = self._client.user.id if self._client.user else None
        for user in message.mentions:
            raw_tokens = (f"<@{user.id}>", f"<@!{user.id}>")
            replacement = "" if user.id == bot_id else f"@{user.display_name}"
            for token in raw_tokens:
                text = text.replace(token, replacement)
        return text.strip()
