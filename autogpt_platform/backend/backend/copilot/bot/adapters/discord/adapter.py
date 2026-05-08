"""Discord adapter — connects to the Discord Gateway and forwards messages.

Platform-specific machinery only: Gateway connection, message event handling,
thread creation, typing, button rendering. All platform-agnostic logic lives
in the core handler. Slash commands live in commands.py.
"""

import logging
import re
from typing import Optional

import discord
from discord import app_commands

from backend.copilot.bot.bot_backend import BotBackend

from ..base import (
    ChannelType,
    MessageCallback,
    MessageContext,
    MessageHistoryEntry,
    PlatformAdapter,
)
from . import commands, config

logger = logging.getLogger(__name__)
THREAD_HISTORY_LIMIT = 20


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

    async def send_message(
        self,
        channel_id: str,
        text: str,
        mentionable_users: tuple[tuple[str, str], ...] = (),
    ) -> None:
        channel = await self._resolve_channel(channel_id)
        if channel and isinstance(channel, discord.abc.Messageable):
            rendered, allowed = _resolve_mentions(text, mentionable_users)
            # tts=False is the default but we pin it explicitly — AutoPilot
            # output is untrusted and should never blast through voice.
            await channel.send(rendered, tts=False, allowed_mentions=allowed)

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
        self,
        channel_id: str,
        text: str,
        reply_to_message_id: str,
        mentionable_users: tuple[tuple[str, str], ...] = (),
    ) -> None:
        channel = await self._resolve_channel(channel_id)
        if not channel or not isinstance(channel, discord.abc.Messageable):
            return
        rendered, allowed = _resolve_mentions(text, mentionable_users)
        try:
            msg = await channel.fetch_message(int(reply_to_message_id))
            await msg.reply(rendered, tts=False, allowed_mentions=allowed)
        except discord.NotFound:
            await channel.send(rendered, tts=False, allowed_mentions=allowed)

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

    async def rename_thread(self, thread_id: str, name: str) -> bool:
        channel = await self._resolve_channel(thread_id)
        if channel is None or not isinstance(channel, discord.Thread):
            logger.warning("Cannot rename non-thread channel %s", thread_id)
            return False
        try:
            await channel.edit(name=name[:100])
            return True
        except discord.HTTPException:
            logger.exception("Failed to rename thread %s", thread_id)
            return False

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
            # Always skip our own messages — without this we'd loop forever
            # answering our own replies. Other bots are fine: in channels
            # we still require an @mention, in threads we still require an
            # explicit subscription, so the existing gates already bound the
            # blast radius of bot-to-bot interactions.
            if (
                self._client.user is not None
                and message.author.id == self._client.user.id
            ):
                return
            if self._on_message_callback is None:
                return

            channel_type = self._channel_type(message)
            bot_mentioned = self._is_mentioned(message)

            # Channels require an explicit @mention; DMs and threads always forward
            # (handler checks thread subscription).
            if channel_type == "channel" and not bot_mentioned:
                return

            thread_history = ()
            if channel_type == "thread" and bot_mentioned:
                thread_history = await self._thread_history(message)

            ctx = MessageContext(
                platform="discord",
                channel_type=channel_type,
                server_id=str(message.guild.id) if message.guild else None,
                channel_id=str(message.channel.id),
                message_id=str(message.id),
                user_id=str(message.author.id),
                username=message.author.display_name,
                text=self._strip_mentions(message),
                bot_mentioned=bot_mentioned,
                thread_history=thread_history,
                mentionable_users=self._collect_mentionable_users(message),
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

    def _collect_mentionable_users(
        self, message: discord.Message
    ) -> tuple[tuple[str, str], ...]:
        """Users from the inbound message the bot may ping back this turn."""
        bot_id = self._client.user.id if self._client.user else None
        return tuple(
            (user.display_name, str(user.id))
            for user in message.mentions
            if user.id != bot_id
        )

    async def _thread_history(
        self, message: discord.Message
    ) -> tuple[MessageHistoryEntry, ...]:
        if not isinstance(message.channel, discord.Thread):
            return ()

        entries: list[MessageHistoryEntry] = []
        bot_user_id = self._client.user.id if self._client.user else None
        try:
            async for prior in message.channel.history(
                limit=THREAD_HISTORY_LIMIT,
                before=message,
                oldest_first=True,
            ):
                # Skip our own outputs — copilot has its own transcript for
                # that side. Other bots' messages are kept as context.
                if bot_user_id is not None and prior.author.id == bot_user_id:
                    continue
                text = self._strip_mentions(prior)
                if not text:
                    continue
                entries.append(
                    MessageHistoryEntry(
                        username=prior.author.display_name,
                        user_id=str(prior.author.id),
                        text=text,
                    )
                )
        except (discord.Forbidden, discord.HTTPException):
            logger.warning("Could not fetch Discord thread history", exc_info=True)
            return ()

        return tuple(entries)


def _resolve_mentions(
    text: str,
    mentionable_users: tuple[tuple[str, str], ...],
) -> tuple[str, discord.AllowedMentions]:
    """Substitute `@DisplayName` in `text` with `<@id>` markup for users on
    the allowlist, and return an AllowedMentions that pings exactly those
    users (and nobody else).

    Anyone not on the allowlist stays as plain text — even if the LLM produces
    `@everyone`, `@here`, or `@SomeRandomUser`. This keeps the bot from
    pinging users it learned about elsewhere or hallucinated entirely.
    """
    if not mentionable_users:
        return text, discord.AllowedMentions.none()

    rendered = text
    pinged_ids: list[int] = []
    # Longest names first so e.g. "@John Smith" matches before "@John".
    for display_name, user_id in sorted(
        mentionable_users, key=lambda pair: -len(pair[0])
    ):
        # Word-bounded so "@Name" inside emails/URLs is left alone.
        pattern = re.compile(
            rf"(?<![\w@]){re.escape(f'@{display_name}')}(?!\w)",
            re.IGNORECASE,
        )
        if not pattern.search(rendered):
            continue
        # Callable replacement avoids backref interpretation of user_id.
        rendered = pattern.sub(lambda _m, uid=user_id: f"<@{uid}>", rendered)
        try:
            pinged_ids.append(int(user_id))
        except ValueError:
            continue

    if not pinged_ids:
        return rendered, discord.AllowedMentions.none()

    return rendered, discord.AllowedMentions(
        everyone=False,
        users=[discord.Object(id=uid) for uid in pinged_ids],
        roles=False,
        replied_user=False,
    )
