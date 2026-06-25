"""Discord adapter — connects to the Discord Gateway and forwards messages.

Platform-specific machinery only: Gateway connection, message event handling,
thread creation, typing, button rendering. All platform-agnostic logic lives
in the core handler. Slash commands live in commands.py.
"""

import io
import logging
import re
from typing import Optional

import discord
from discord import app_commands

from backend.copilot.bot import threads
from backend.copilot.bot.bot_backend import BotBackend
from backend.copilot.bot.text import split_at_boundary

from ..base import (
    ChannelInfo,
    ChannelType,
    FileAttachment,
    MessageCallback,
    MessageContext,
    MessageHistoryEntry,
    PlatformAdapter,
    PostedRef,
    ReferencedConversation,
)
from . import commands, config, intro
from .references import (
    ReferenceTarget,
    extract_referenced_targets,
    replace_referenced_links,
)

logger = logging.getLogger(__name__)

# When the bot is @-ed into an existing thread it pulls the prior messages in as
# context. Read the whole thread (up to this hard cap, ~10 Discord API pages)
# rather than just the last handful — skipping older messages makes the bot act
# on a partial conversation. THREAD_HISTORY_CHAR_BUDGET then keeps the assembled
# context within the copilot request size limit, preferring the most recent
# messages when a thread is very long.
THREAD_HISTORY_LIMIT = 1000
THREAD_HISTORY_CHAR_BUDGET = 24000
_HISTORY_TRUNCATION_MARKER = "\n… [message truncated]"

# When a message links or @-mentions other threads/channels, the bot fetches
# their recent content up-front (same guild only) so the model can read it
# instead of trying to web-fetch a JS-rendered Discord page. Bounded so a
# link-heavy message can't fan out into many large reads.
MAX_REFERENCED_CONVERSATIONS = 3
REFERENCED_HISTORY_LIMIT = 200
REFERENCED_CHAR_BUDGET = 8000
# When a link names a specific message, fetch that message plus a little of the
# conversation leading up to it (rather than the channel's latest activity).
REFERENCED_MESSAGE_CONTEXT = 15


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

    @property
    def max_attachment_bytes(self) -> int:
        return config.MAX_ATTACHMENT_BYTES

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

    async def send_file(self, channel_id: str, text: str, file: FileAttachment) -> None:
        channel = await self._resolve_channel(channel_id)
        if channel is None or not isinstance(channel, discord.abc.Messageable):
            return
        # spoiler=False — AutoPilot output is untrusted but spoilering every
        # generated file would be noisy; the workspace fetcher already
        # validated user ownership before we got bytes.
        attachment = discord.File(
            io.BytesIO(file.content),
            filename=file.filename or "file",
            spoiler=False,
        )
        await channel.send(text or None, file=attachment, tts=False)

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

    # -- Proactive output --

    async def list_text_channels(
        self, server_ids: tuple[str, ...]
    ) -> list[ChannelInfo]:
        wanted = set(server_ids)
        channels: list[ChannelInfo] = []
        for guild in self._client.guilds:
            if str(guild.id) not in wanted:
                continue
            me = guild.me
            for channel in guild.text_channels:
                # Only surface channels the bot can actually post in, so the
                # picker upstream never offers a target that would 403 on send.
                # If the bot member isn't cached (`me is None`) we can't check,
                # so we surface the channel and let the eventual send fail
                # loudly rather than hide everything.
                if me is not None and not channel.permissions_for(me).send_messages:
                    continue
                channels.append(
                    ChannelInfo(
                        id=str(channel.id),
                        name=channel.name,
                        server_id=str(guild.id),
                        server_name=guild.name,
                    )
                )
        return channels

    async def get_channel_server_id(self, channel_id: str) -> Optional[str]:
        channel = await self._resolve_channel(channel_id)
        # Guild channels and threads carry `.guild`; DM/group channels don't,
        # so they resolve to None (never authorized for a server post).
        if isinstance(channel, (discord.abc.GuildChannel, discord.Thread)):
            return str(channel.guild.id)
        return None

    async def post_channel_message(
        self, channel_id: str, text: str
    ) -> Optional[PostedRef]:
        channel = await self._resolve_channel(channel_id)
        if channel is None or not isinstance(channel, discord.abc.Messageable):
            logger.warning("Cannot post to non-messageable channel %s", channel_id)
            return None
        try:
            first = await self._send_chunked(channel, text)
        except discord.HTTPException:
            logger.exception("Failed to post message to channel %s", channel_id)
            return None
        if first is None:
            return None
        return PostedRef(id=str(first.id), url=first.jump_url)

    async def create_channel_thread(
        self, channel_id: str, name: str, text: str
    ) -> Optional[PostedRef]:
        channel = await self._resolve_channel(channel_id)
        if channel is None or not isinstance(channel, discord.TextChannel):
            logger.warning("Cannot create thread in non-text channel %s", channel_id)
            return None
        try:
            thread = await channel.create_thread(
                name=name[:100],
                type=discord.ChannelType.public_thread,
            )
        except discord.HTTPException:
            logger.exception("Failed to create thread in channel %s", channel_id)
            return None
        # The thread now exists on Discord. Surface its ref even if posting the
        # body fails, so the caller reports partial success and doesn't retry
        # into a duplicate thread.
        try:
            await self._send_chunked(thread, text)
        except discord.HTTPException:
            logger.exception(
                "Thread %s created but posting its content failed", thread.id
            )
        return PostedRef(id=str(thread.id), url=thread.jump_url)

    async def _send_chunked(
        self, channel: discord.abc.Messageable, text: str
    ) -> Optional[discord.Message]:
        """Send ``text`` to ``channel``, splitting at natural boundaries to stay
        under Discord's per-message cap. Returns the first message sent (the one
        callers permalink to), or ``None`` if there was nothing to send.

        Raises only if the *first* chunk fails — once anything is delivered, a
        later-chunk failure stops the send and keeps the partial result rather
        than discarding what already posted (a retry would duplicate it).
        """
        remaining = text.strip()
        first: Optional[discord.Message] = None
        while remaining:
            chunk, remaining = split_at_boundary(remaining, config.CHUNK_FLUSH_AT)
            if not chunk:
                break
            try:
                msg = await channel.send(chunk, tts=False)
            except discord.HTTPException:
                if first is None:
                    raise
                logger.exception("Dropping trailing chunk after partial send")
                break
            if first is None:
                first = msg
        return first

    # -- Internal --

    def _register_events(self) -> None:
        @self._client.event
        async def on_ready() -> None:
            logger.info(f"Discord bot connected as {self._client.user}")
            # Refresh display names for every guild we're currently in — keeps
            # the Bots settings page in sync with renames and backfills any
            # rows that pre-date name capture. Cheap: in-memory cache only,
            # no Discord API calls.
            await self._refresh_known_server_names()
            # Reconcile presence analytics: record every server we're currently
            # in and mark any we've left while disconnected. Drives the admin
            # server-count / sharding-prediction charts.
            self._api.sync_guilds(
                "discord",
                [(str(guild.id), guild.name) for guild in self._client.guilds],
            )
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
        async def on_guild_join(guild: discord.Guild) -> None:
            await self._refresh_server_name(guild)
            self._api.track_guild_joined("discord", str(guild.id), guild.name)
            channel = intro.pick_intro_channel(guild)
            if channel is None:
                logger.info(
                    "No sendable channel in guild %s for intro message", guild.id
                )
                return
            try:
                await channel.send(
                    intro.intro_message(),
                    tts=False,
                    allowed_mentions=discord.AllowedMentions.none(),
                )
            except discord.HTTPException:
                logger.exception("Failed to post intro message in guild %s", guild.id)

        @self._client.event
        async def on_guild_update(before: discord.Guild, after: discord.Guild) -> None:
            if before.name != after.name:
                await self._refresh_server_name(after)

        @self._client.event
        async def on_guild_remove(guild: discord.Guild) -> None:
            # Bot was kicked or the server was deleted — mark presence as left
            # so the live server count and sharding prediction stay accurate.
            self._api.track_guild_left("discord", str(guild.id))

        @self._client.event
        async def on_thread_remove(thread: discord.Thread) -> None:
            # Fires when a thread is removed from the client's cache — the
            # typical path is the bot being kicked or the thread being made
            # private without us in it. Drop the auto-reply subscription so a
            # subsequent re-add by @mention starts in the @-only mode again.
            # Unlike on_thread_member_remove, this event only requires the
            # default Intents.guilds and not the privileged members intent.
            try:
                await threads.unsubscribe("discord", str(thread.id))
            except Exception:
                logger.exception(
                    f"Failed to unsubscribe thread {thread.id} after removal"
                )

        @self._client.event
        async def on_message(message: discord.Message) -> None:
            if self._should_ignore_message(message):
                return
            if self._on_message_callback is None:
                return

            # A locked thread rejects bot sends — processing the message would
            # just burn a turn and error on every reply. Skip until it's
            # unlocked. (`locked` is set on archive-locked threads too.)
            channel = message.channel
            if isinstance(channel, discord.Thread) and channel.locked:
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

            own_text = self._strip_mentions(message)
            # Scan ONLY the user's own typed message for references. Links inside
            # a forwarded or replied-to message are quoted context, not the
            # user's request, so they must not be auto-fetched or rewritten.
            referenced = await self._fetch_referenced_conversations(message, own_text)
            if referenced:
                # Swap the raw channel links/mentions for readable "#name"
                # tokens — the fetched content is supplied under those names, so
                # this stops the model from treating the paste as a URL it must
                # open (and then claiming it can't access Discord).
                own_text = replace_referenced_links(
                    own_text, {c.channel_id: c.title for c in referenced}
                )
            # Fold in forwarded content and the replied-to message as quoted
            # context — both verbatim, with their links left untouched.
            message_text = self._compose_with_forward(message, own_text)
            message_text = await self._with_reply_context(message, message_text)
            ctx = MessageContext(
                platform="discord",
                channel_type=channel_type,
                server_id=str(message.guild.id) if message.guild else None,
                channel_id=str(message.channel.id),
                message_id=str(message.id),
                user_id=str(message.author.id),
                username=message.author.display_name,
                text=message_text,
                bot_mentioned=bot_mentioned,
                thread_history=thread_history,
                mentionable_users=self._collect_mentionable_users(message),
                referenced_conversations=referenced,
            )
            await self._on_message_callback(ctx, self)

    async def _refresh_known_server_names(self) -> None:
        """Push current display names for every guild the bot is in."""
        for guild in self._client.guilds:
            await self._refresh_server_name(guild)

    async def _refresh_server_name(self, guild: discord.Guild) -> None:
        if not guild.name:
            return
        try:
            await self._api.refresh_server_name(
                platform="discord",
                platform_server_id=str(guild.id),
                server_name=guild.name,
            )
        except Exception:
            logger.exception("Failed to refresh display name for guild %s", guild.id)

    def _should_ignore_message(self, message: discord.Message) -> bool:
        if self._client.user is not None and message.author.id == self._client.user.id:
            return True
        # Other bots reach us only by @mentioning us; without this gate two
        # bots in a shared thread (our own dev + prod included) loop forever.
        if message.author.bot:
            return not self._is_mentioned(message)
        return False

    def _is_mentioned(self, message: discord.Message) -> bool:
        if message.guild is None:
            return True  # DMs always count
        bot_user = self._client.user
        if bot_user is None:
            return False
        return any(user.id == bot_user.id for user in message.mentions)

    @staticmethod
    def _channel_type(message: discord.Message) -> ChannelType:
        if message.guild is None:
            return "dm"
        if isinstance(message.channel, discord.Thread):
            return "thread"
        return "channel"

    def _message_text(self, message: discord.Message) -> str:
        """The full message body the LLM should see.

        A Discord *forward* carries the forwarded message in
        ``message.message_snapshots`` — separate from ``content``, which only
        holds the comment the user typed alongside the forward. Reading
        ``content`` alone (as ``_strip_mentions`` does) drops the forwarded
        message entirely, so the LLM acts on just the comment and can badly
        misread the request. Stitch the forwarded content back in here.
        """
        return self._compose_with_forward(message, self._strip_mentions(message))

    def _compose_with_forward(self, message: discord.Message, own: str) -> str:
        """Fold any forwarded snapshot content onto the user's own text.

        Kept separate from reference scanning so forwarded content (quoted, not
        the user's request) is never mined for channel links to auto-fetch.
        """
        forwarded = self._forwarded_text(message)
        if not forwarded:
            return own
        if own:
            return f"{own}\n\n[Forwarded message]\n{forwarded}"
        return f"[Forwarded message]\n{forwarded}"

    async def _with_reply_context(
        self, message: discord.Message, message_text: str
    ) -> str:
        """Prepend the replied-to message so the bot sees what's being answered.

        A Discord reply (``message.reference``) only carries the short reply
        text; the message it replies to holds the actual intent. It's always in
        the same channel the user is already posting in, so surfacing it leaks
        nothing they can't already see.
        """
        replied = await self._resolve_reply(message)
        if replied is None:
            return message_text
        quoted = self._message_text(replied)
        if not quoted:
            return message_text
        prefix = f"[Replying to {replied.author.display_name}]\n{quoted}"
        return f"{prefix}\n\n{message_text}" if message_text else prefix

    async def _resolve_reply(
        self, message: discord.Message
    ) -> Optional[discord.Message]:
        ref = message.reference
        if ref is None:
            return None
        # Forwards also use ``reference`` but carry their body in snapshots,
        # which ``_message_text`` already stitches in — don't re-resolve those.
        if getattr(message, "message_snapshots", None):
            return None
        if isinstance(ref.resolved, discord.Message):
            return ref.resolved
        if ref.message_id is None or not isinstance(
            message.channel, discord.abc.Messageable
        ):
            return None
        try:
            return await message.channel.fetch_message(ref.message_id)
        except (discord.NotFound, discord.Forbidden, discord.HTTPException):
            return None

    @staticmethod
    def _forwarded_text(message: discord.Message) -> str:
        """Flatten any forwarded message snapshots into labelled text."""
        # ``message_snapshots`` is a real runtime attribute (in ``Message``'s
        # slots) but isn't in discord.py 2.6's type stubs yet; read it through
        # getattr so the type checker is happy and older builds without message
        # forwarding degrade to "no snapshots".
        snapshots = getattr(message, "message_snapshots", [])
        parts: list[str] = []
        for snapshot in snapshots:
            content = (snapshot.content or "").strip()
            if content:
                parts.append(content)
            # Surface forwarded attachments by name so the LLM knows files came
            # with the forward, even though we can't inline their bytes here.
            for attachment in snapshot.attachments:
                parts.append(f"[Attached file: {attachment.filename}]")
        return "\n\n".join(parts).strip()

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
        try:
            return await self._budgeted_history(
                message.channel.history(
                    limit=THREAD_HISTORY_LIMIT,
                    before=message,
                    oldest_first=False,
                ),
                THREAD_HISTORY_CHAR_BUDGET,
            )
        except (discord.Forbidden, discord.HTTPException):
            logger.warning("Could not fetch Discord thread history", exc_info=True)
            return ()

    async def _budgeted_history(
        self, history, char_budget: int
    ) -> tuple[MessageHistoryEntry, ...]:
        """Drain a newest-first message iterator into chronological entries,
        capped at ``char_budget`` chars (most-recent kept when over budget).

        Skips the bot's own messages (copilot has its own transcript for those)
        but keeps other bots' messages as context.
        """
        entries: list[MessageHistoryEntry] = []
        used_chars = 0
        bot_user_id = self._client.user.id if self._client.user else None
        async for prior in history:
            if bot_user_id is not None and prior.author.id == bot_user_id:
                continue
            text = self._strip_mentions(prior)
            if not text:
                continue
            remaining = char_budget - used_chars
            if remaining <= 0:
                break
            oversized = len(text) > remaining
            if oversized and entries:
                # No room for another whole message — keep what we have.
                break
            if oversized:
                # Lone most-recent message is itself over budget: keep a head.
                text = _truncate_to_budget(text, remaining)
            used_chars += len(text)
            entries.append(
                MessageHistoryEntry(
                    username=prior.author.display_name,
                    user_id=str(prior.author.id),
                    text=text,
                )
            )
            if oversized:
                break
        entries.reverse()  # chronological order for the prompt
        return tuple(entries)

    async def _fetch_referenced_conversations(
        self, message: discord.Message, text: str
    ) -> tuple[ReferencedConversation, ...]:
        """Fetch the recent content of any thread/channel ``text`` references.

        Read as the *requesting member* would: same-guild only, and only
        channels they can actually see. The bot's gateway can read more than
        the user can, so we must never surface a private channel's history to
        someone who lacks access to it themselves.
        """
        if message.guild is None:
            return ()
        # ``message.author`` IS the guild Member (the gateway attaches it to the
        # event); ``get_member`` only reads the cache, which is empty without the
        # privileged members intent — so prefer the author and fall back to the
        # cache only if it somehow isn't a Member (e.g. a webhook).
        requester = (
            message.author
            if isinstance(message.author, discord.Member)
            else message.guild.get_member(message.author.id)
        )
        if requester is None:
            return ()
        targets = extract_referenced_targets(
            text,
            exclude_channel_id=str(message.channel.id),
            limit=MAX_REFERENCED_CONVERSATIONS,
        )
        conversations: list[ReferencedConversation] = []
        for target in targets:
            convo = await self._fetch_one_referenced(message.guild, requester, target)
            if convo is not None:
                conversations.append(convo)
        return tuple(conversations)

    async def _fetch_one_referenced(
        self, guild: discord.Guild, requester: discord.Member, target: ReferenceTarget
    ) -> Optional[ReferencedConversation]:
        channel = await self._resolve_channel(target.channel_id)
        if not isinstance(channel, (discord.TextChannel, discord.Thread)):
            return None
        if channel.guild.id != guild.id:
            # Cross-guild reference — don't read content from another server.
            return None
        if not await self._can_requester_read(channel, requester):
            return None
        try:
            if target.message_id is not None:
                # A permalink to a specific message: read that message and the
                # turns leading up to it (``before`` the next snowflake includes
                # the message itself), so "what was said here <link>" works even
                # for an older message not in the channel's latest activity.
                history = channel.history(
                    limit=REFERENCED_MESSAGE_CONTEXT,
                    before=discord.Object(id=int(target.message_id) + 1),
                    oldest_first=False,
                )
            else:
                history = channel.history(
                    limit=REFERENCED_HISTORY_LIMIT, oldest_first=False
                )
            messages = await self._budgeted_history(history, REFERENCED_CHAR_BUDGET)
        except (discord.Forbidden, discord.HTTPException):
            logger.warning(
                "Could not fetch referenced channel %s",
                target.channel_id,
                exc_info=True,
            )
            return None
        if not messages:
            return None
        return ReferencedConversation(
            title=channel.name, channel_id=target.channel_id, messages=messages
        )

    async def _can_requester_read(
        self, channel: "discord.TextChannel | discord.Thread", requester: discord.Member
    ) -> bool:
        """Whether ``requester`` may read ``channel`` themselves.

        Channel-level ``view_channel`` + ``read_message_history`` is the base
        gate. For a *private* thread that is not enough — Discord requires the
        member to actually be in the thread (``manage_threads`` bypasses, as it
        does in the client), so check membership explicitly to avoid leaking a
        private thread the bot happens to be in.
        """
        perms = channel.permissions_for(requester)
        if not (perms.view_channel and perms.read_message_history):
            return False
        if isinstance(channel, discord.Thread) and channel.is_private():
            if perms.manage_threads:
                return True
            try:
                await channel.fetch_member(requester.id)
            except (discord.NotFound, discord.Forbidden, discord.HTTPException):
                return False
        return True


def _truncate_to_budget(text: str, limit: int) -> str:
    """Trim ``text`` to at most ``limit`` characters, leaving a visible marker.

    Used only when a single thread message is itself larger than the history
    budget — keep a head of it (with context that it was cut) rather than emit
    an oversized payload or drop the message entirely.
    """
    if len(text) <= limit:
        return text
    keep = max(0, limit - len(_HISTORY_TRUNCATION_MARKER))
    return text[:keep].rstrip() + _HISTORY_TRUNCATION_MARKER


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
