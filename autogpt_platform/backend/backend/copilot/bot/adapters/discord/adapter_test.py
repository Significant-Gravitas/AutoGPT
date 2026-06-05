"""Tests for DiscordAdapter helpers that don't need a live gateway."""

from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest

from backend.copilot.bot.adapters.discord.adapter import (
    THREAD_HISTORY_LIMIT,
    DiscordAdapter,
    _resolve_mentions,
)


def _bare_adapter(bot_id: int | None = 1000) -> tuple[DiscordAdapter, MagicMock]:
    """Build a DiscordAdapter without going through __init__ (which spins up
    discord.py internals). Returns the adapter alongside the MagicMock that
    stands in for ``_client`` — tests reach into the mock directly for
    per-method stubbing.
    """
    adapter = DiscordAdapter.__new__(DiscordAdapter)
    client = MagicMock()
    client.user = MagicMock(id=bot_id) if bot_id is not None else None
    adapter._client = cast(discord.Client, client)
    adapter._on_message_callback = None
    adapter._commands_synced = False
    return adapter, client


def _mention(user_id: int, display_name: str) -> MagicMock:
    user = MagicMock()
    user.id = user_id
    user.display_name = display_name
    return user


def _message(content: str, mentions: list[MagicMock]) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    msg.mentions = mentions
    return msg


class _AsyncHistory:
    def __init__(self, messages: list[MagicMock]):
        self._messages = messages

    def __aiter__(self):
        self._iter = iter(self._messages)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


# ── _strip_mentions ────────────────────────────────────────────────────


class TestStripMentions:
    def test_strips_only_bot_mention(self):
        adapter, _ = _bare_adapter(bot_id=1000)
        bot = _mention(1000, "AutoPilot")
        alice = _mention(2000, "Alice")
        msg = _message(
            "<@1000> please summarise what <@2000> said",
            mentions=[bot, alice],
        )

        assert adapter._strip_mentions(msg) == "please summarise what @Alice said"

    def test_handles_nickname_style_tokens(self):
        adapter, _ = _bare_adapter(bot_id=1000)
        bot = _mention(1000, "AutoPilot")
        alice = _mention(2000, "Alice")
        msg = _message("<@!1000> ping <@!2000>", mentions=[bot, alice])

        assert adapter._strip_mentions(msg) == "ping @Alice"

    def test_no_bot_user_leaves_all_mentions_as_names(self):
        adapter, _ = _bare_adapter(bot_id=None)
        alice = _mention(2000, "Alice")
        msg = _message("hi <@2000>", mentions=[alice])

        assert adapter._strip_mentions(msg) == "hi @Alice"

    def test_message_without_mentions_is_trimmed(self):
        adapter, _ = _bare_adapter(bot_id=1000)
        msg = _message("  hello world  ", mentions=[])

        assert adapter._strip_mentions(msg) == "hello world"

    @pytest.mark.parametrize(
        "content,expected",
        [
            ("<@1000>", ""),
            ("<@!1000>", ""),
            ("<@1000> hi", "hi"),
            ("hi <@1000>", "hi"),
        ],
    )
    def test_bot_only_variants(self, content: str, expected: str):
        adapter, _ = _bare_adapter(bot_id=1000)
        bot = _mention(1000, "AutoPilot")
        msg = _message(content, mentions=[bot])

        assert adapter._strip_mentions(msg) == expected


# ── _channel_type ──────────────────────────────────────────────────────


class TestChannelType:
    def test_dm_has_no_guild(self):
        msg = MagicMock()
        msg.guild = None
        assert DiscordAdapter._channel_type(msg) == "dm"

    def test_thread_inside_guild(self):
        msg = MagicMock()
        msg.guild = MagicMock()
        msg.channel = MagicMock(spec=discord.Thread)
        assert DiscordAdapter._channel_type(msg) == "thread"

    def test_regular_channel_inside_guild(self):
        msg = MagicMock()
        msg.guild = MagicMock()
        msg.channel = MagicMock()
        assert DiscordAdapter._channel_type(msg) == "channel"


# ── _is_mentioned ──────────────────────────────────────────────────────


class TestShouldIgnoreMessage:
    def test_ignores_own_message(self):
        adapter, _ = _bare_adapter(bot_id=1000)
        msg = _message("hi", [])
        msg.author = MagicMock(id=1000, bot=True)

        assert adapter._should_ignore_message(msg) is True

    def test_ignores_unmentioned_bot_message(self):
        # A bot that doesn't @mention us is skipped — otherwise two bots
        # sharing a thread (our own dev + prod included) loop forever.
        adapter, _ = _bare_adapter(bot_id=1000)
        msg = _message("hi", [])
        msg.author = MagicMock(id=2000, bot=True)
        msg.guild = MagicMock()

        assert adapter._should_ignore_message(msg) is True

    def test_allows_mentioned_bot_message(self):
        # Another bot can still reach us by explicitly @mentioning us.
        adapter, _ = _bare_adapter(bot_id=1000)
        msg = _message("hi", [_mention(1000, "AutoPilot")])
        msg.author = MagicMock(id=2000, bot=True)
        msg.guild = MagicMock()

        assert adapter._should_ignore_message(msg) is False

    def test_allows_human_message(self):
        adapter, _ = _bare_adapter(bot_id=1000)
        msg = _message("hi", [])
        msg.author = MagicMock(id=2000, bot=False)

        assert adapter._should_ignore_message(msg) is False


class TestIsMentioned:
    def test_dm_always_counts_as_mentioned(self):
        adapter, _ = _bare_adapter(bot_id=1000)
        msg = MagicMock()
        msg.guild = None
        assert adapter._is_mentioned(msg) is True

    def test_guild_requires_explicit_mention(self):
        adapter, _ = _bare_adapter(bot_id=1000)
        msg = _message("hi", [_mention(2000, "someone-else")])
        msg.guild = MagicMock()
        assert adapter._is_mentioned(msg) is False

    def test_guild_with_mention_passes(self):
        adapter, _ = _bare_adapter(bot_id=1000)
        msg = _message("hi", [_mention(1000, "AutoPilot")])
        msg.guild = MagicMock()
        assert adapter._is_mentioned(msg) is True

    def test_no_bot_user_treats_guild_mention_as_false(self):
        adapter, _ = _bare_adapter(bot_id=None)
        msg = _message("hi", [_mention(1000, "AutoPilot")])
        msg.guild = MagicMock()
        assert adapter._is_mentioned(msg) is False

    def test_everyone_ping_alone_is_not_a_mention(self):
        # `discord.User.mentioned_in` short-circuits to True on @everyone /
        # @here, which used to make the bot reply to every server-wide ping.
        # We now check `message.mentions` so only explicit @bot counts.
        adapter, _ = _bare_adapter(bot_id=1000)
        msg = _message("hey @everyone heads up", [])
        msg.guild = MagicMock()
        msg.mention_everyone = True
        assert adapter._is_mentioned(msg) is False

    def test_everyone_ping_with_explicit_bot_mention_still_counts(self):
        # If they ping the bot AND @everyone, the bot is still in
        # `message.mentions` and we should reply normally.
        adapter, _ = _bare_adapter(bot_id=1000)
        msg = _message("@everyone and @AutoPilot", [_mention(1000, "AutoPilot")])
        msg.guild = MagicMock()
        msg.mention_everyone = True
        assert adapter._is_mentioned(msg) is True


# ── _resolve_channel ───────────────────────────────────────────────────


class TestResolveChannel:
    @pytest.mark.asyncio
    async def test_cache_hit_skips_rest_fetch(self):
        adapter, client = _bare_adapter()
        cached = MagicMock()
        client.get_channel.return_value = cached
        client.fetch_channel = AsyncMock()

        result = await adapter._resolve_channel("123")

        assert result is cached
        client.fetch_channel.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cache_miss_falls_back_to_rest(self):
        adapter, client = _bare_adapter()
        fetched = MagicMock()
        client.get_channel.return_value = None
        client.fetch_channel = AsyncMock(return_value=fetched)

        result = await adapter._resolve_channel("123")

        assert result is fetched
        client.fetch_channel.assert_awaited_once_with(123)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exc",
        [
            discord.NotFound(MagicMock(status=404), "gone"),
            discord.Forbidden(MagicMock(status=403), "nope"),
            discord.HTTPException(MagicMock(status=500), "boom"),
        ],
    )
    async def test_rest_failure_returns_none(self, exc: Exception):
        adapter, client = _bare_adapter()
        client.get_channel.return_value = None
        client.fetch_channel = AsyncMock(side_effect=exc)

        assert await adapter._resolve_channel("123") is None


# ── send_message / send_reply / send_link ──────────────────────────────


class TestSendMethods:
    @pytest.mark.asyncio
    async def test_send_message_pins_tts_false(self):
        adapter, client = _bare_adapter()
        channel = MagicMock(spec=discord.TextChannel)
        channel.send = AsyncMock()
        client.get_channel.return_value = channel

        await adapter.send_message("123", "hi")

        channel.send.assert_awaited_once()
        args, kwargs = channel.send.await_args
        assert args == ("hi",)
        assert kwargs["tts"] is False
        # Default empty mentionable_users → AllowedMentions.none()
        assert isinstance(kwargs["allowed_mentions"], discord.AllowedMentions)

    @pytest.mark.asyncio
    async def test_send_message_silently_drops_when_channel_missing(self):
        adapter, client = _bare_adapter()
        client.get_channel.return_value = None
        client.fetch_channel = AsyncMock(
            side_effect=discord.NotFound(MagicMock(status=404), "gone")
        )
        # Should not raise even though there's nothing to send to.
        await adapter.send_message("123", "hi")

    @pytest.mark.asyncio
    async def test_send_link_attaches_button_and_pins_tts(self):
        adapter, client = _bare_adapter()
        channel = MagicMock(spec=discord.TextChannel)
        channel.send = AsyncMock()
        client.get_channel.return_value = channel

        await adapter.send_link("123", "click me", "Open", "https://example.com")

        assert channel.send.await_count == 1
        kwargs = channel.send.await_args.kwargs
        assert kwargs["tts"] is False
        view = kwargs["view"]
        assert any(
            getattr(c, "url", None) == "https://example.com" for c in view.children
        )

    @pytest.mark.asyncio
    async def test_send_reply_falls_back_to_send_when_message_missing(self):
        adapter, client = _bare_adapter()
        channel = MagicMock(spec=discord.TextChannel)
        channel.send = AsyncMock()
        channel.fetch_message = AsyncMock(
            side_effect=discord.NotFound(MagicMock(status=404), "gone")
        )
        client.get_channel.return_value = channel

        await adapter.send_reply("123", "hello", "999")

        channel.send.assert_awaited_once()
        args, kwargs = channel.send.await_args
        assert args == ("hello",)
        assert kwargs["tts"] is False
        assert isinstance(kwargs["allowed_mentions"], discord.AllowedMentions)


class TestRenameThread:
    @pytest.mark.asyncio
    async def test_renames_resolved_thread(self):
        adapter, client = _bare_adapter()
        thread = MagicMock(spec=discord.Thread)
        thread.edit = AsyncMock()
        client.get_channel.return_value = thread

        assert await adapter.rename_thread("123", "Generated Web Title") is True

        thread.edit.assert_awaited_once_with(name="Generated Web Title")

    @pytest.mark.asyncio
    async def test_refuses_non_thread_channel(self):
        adapter, client = _bare_adapter()
        channel = MagicMock(spec=discord.TextChannel)
        channel.edit = AsyncMock()
        client.get_channel.return_value = channel

        assert await adapter.rename_thread("123", "Generated Web Title") is False

        channel.edit.assert_not_awaited()


# ── properties ─────────────────────────────────────────────────────────


class TestThreadHistory:
    @pytest.mark.asyncio
    async def test_fetches_user_thread_history_oldest_first(self):
        adapter, _ = _bare_adapter(bot_id=1000)
        bot = _mention(1000, "AutoPilot")

        prior_1 = _message("first idea", [])
        prior_1.author = MagicMock(bot=False, id=2000, display_name="Alice")
        prior_2 = _message("<@1000> can ignore old bot ping", [bot])
        prior_2.author = MagicMock(bot=False, id=3000, display_name="Bob")
        bot_msg = _message("old bot output", [])
        bot_msg.author = MagicMock(bot=True, id=1000, display_name="AutoPilot")

        channel = MagicMock(spec=discord.Thread)
        channel.history.return_value = _AsyncHistory([prior_1, bot_msg, prior_2])
        message = _message("<@1000> help", [bot])
        message.channel = channel

        history = await adapter._thread_history(message)

        channel.history.assert_called_once_with(
            limit=THREAD_HISTORY_LIMIT,
            before=message,
            oldest_first=True,
        )
        assert [entry.username for entry in history] == ["Alice", "Bob"]
        assert [entry.user_id for entry in history] == ["2000", "3000"]
        assert [entry.text for entry in history] == [
            "first idea",
            "can ignore old bot ping",
        ]


class TestProperties:
    def test_platform_name_is_discord(self):
        adapter, _ = _bare_adapter()
        assert adapter.platform_name == "discord"

    def test_chunk_flush_at_is_under_message_limit(self):
        adapter, _ = _bare_adapter()
        assert adapter.chunk_flush_at < adapter.max_message_length


class TestResolveMentions:
    def test_empty_allowlist_returns_text_unchanged_and_no_mentions(self):
        text = "Hello @World, ping @everyone"
        rendered, allowed = _resolve_mentions(text, ())
        assert rendered == text
        assert allowed.everyone is False
        assert allowed.users is False

    def test_resolves_allowlisted_displayname_to_id_markup(self):
        rendered, allowed = _resolve_mentions(
            "Hey @Sue, did you see this?",
            (("Sue", "12345"),),
        )
        assert rendered == "Hey <@12345>, did you see this?"
        assert isinstance(allowed.users, list)
        assert [getattr(u, "id", None) for u in allowed.users] == [12345]
        assert allowed.everyone is False

    def test_leaves_unknown_handles_as_plain_text(self):
        rendered, allowed = _resolve_mentions(
            "Hey @Sue and @Random",
            (("Sue", "12345"),),
        )
        assert "<@12345>" in rendered
        assert "@Random" in rendered  # not on allowlist → stays plain
        assert isinstance(allowed.users, list)
        assert [getattr(u, "id", None) for u in allowed.users] == [12345]

    def test_longest_name_wins_when_handles_share_a_prefix(self):
        # "@John Smith" should match before "@John" so the right user pings.
        rendered, allowed = _resolve_mentions(
            "Ask @John Smith about it",
            (("John", "111"), ("John Smith", "222")),
        )
        assert "<@222>" in rendered
        assert "<@111>" not in rendered
        assert isinstance(allowed.users, list)
        assert any(getattr(u, "id", None) == 222 for u in allowed.users)

    def test_does_not_ping_everyone_or_here(self):
        rendered, allowed = _resolve_mentions(
            "Heads up @everyone and @here",
            (("everyone", "9"),),  # even if a user is named "everyone"…
        )
        # "@everyone" gets substituted to <@9> because the user is on the
        # allowlist, but AllowedMentions.everyone is False so Discord
        # won't ping the @everyone role. "@here" stays plain text.
        assert "<@9>" in rendered
        assert "@here" in rendered
        assert allowed.everyone is False
        assert allowed.roles is False

    def test_does_not_match_inside_emails_or_urls(self):
        # The whole point of word boundaries: a stray "@Sue" inside an
        # email address must stay an email, not a ping.
        rendered, allowed = _resolve_mentions(
            "Email me at hello@Sue.example.com or visit https://Sue.dev",
            (("Sue", "12345"),),
        )
        assert rendered == (
            "Email me at hello@Sue.example.com or visit https://Sue.dev"
        )
        assert allowed.everyone is False

    def test_resolves_standalone_mention_alongside_email_in_same_message(self):
        rendered, _ = _resolve_mentions(
            "@Sue, can you check sue@Sue.com?",
            (("Sue", "12345"),),
        )
        # Leading "@Sue" resolves; the one inside the email survives intact.
        assert rendered == "<@12345>, can you check sue@Sue.com?"


class TestCollectMentionableUsers:
    def test_excludes_the_bot_itself(self):
        adapter, _ = _bare_adapter(bot_id=1000)
        msg = _message(
            "<@1000> please tell <@2000> something",
            mentions=[
                _mention(1000, "AutoPilot"),
                _mention(2000, "Sue"),
            ],
        )
        result = adapter._collect_mentionable_users(msg)
        assert result == (("Sue", "2000"),)

    def test_returns_empty_when_only_bot_mentioned(self):
        adapter, _ = _bare_adapter(bot_id=1000)
        msg = _message("<@1000> hi", mentions=[_mention(1000, "AutoPilot")])
        assert adapter._collect_mentionable_users(msg) == ()


def _bare_adapter_with_api() -> tuple[DiscordAdapter, MagicMock, MagicMock]:
    adapter, client = _bare_adapter(bot_id=1000)
    api = MagicMock()
    api.refresh_server_name = AsyncMock()
    adapter._api = api
    return adapter, client, api


def _guild(guild_id: int, name: str | None) -> MagicMock:
    g = MagicMock()
    g.id = guild_id
    g.name = name
    return g


class TestRefreshServerNames:
    @pytest.mark.asyncio
    async def test_pushes_every_guild_to_the_backend(self):
        adapter, client, api = _bare_adapter_with_api()
        client.guilds = [_guild(1, "Server One"), _guild(2, "Server Two")]
        await adapter._refresh_known_server_names()
        assert api.refresh_server_name.await_count == 2
        api.refresh_server_name.assert_any_await(
            platform="discord", platform_server_id="1", server_name="Server One"
        )
        api.refresh_server_name.assert_any_await(
            platform="discord", platform_server_id="2", server_name="Server Two"
        )

    @pytest.mark.asyncio
    async def test_skips_guild_with_blank_name(self):
        adapter, _, api = _bare_adapter_with_api()
        await adapter._refresh_server_name(_guild(99, ""))
        api.refresh_server_name.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_swallows_backend_errors(self):
        adapter, _, api = _bare_adapter_with_api()
        api.refresh_server_name.side_effect = RuntimeError("rpc down")
        # Must not raise — refreshing names is never critical-path.
        await adapter._refresh_server_name(_guild(1, "Server One"))


# ── on_thread_remove ────────────────────────────────────────────────────


def _thread(thread_id: int) -> MagicMock:
    thread = MagicMock(spec=discord.Thread)
    thread.id = thread_id
    return thread


def _register_events_with_mocked_decorator(adapter: DiscordAdapter) -> dict:
    """Capture the @client.event handlers without actually attaching to
    discord.py. Returns a name→coroutine map for direct invocation."""
    handlers: dict = {}

    def _event(coro):
        handlers[coro.__name__] = coro
        return coro

    adapter._client.event = _event  # type: ignore[assignment]
    adapter._register_events()
    return handlers


class TestOnThreadRemove:
    @pytest.mark.asyncio
    async def test_removal_unsubscribes_thread(self):
        # We use on_thread_remove instead of on_thread_member_remove so we
        # don't need the privileged `members` intent. The trade-off is that
        # this only tells us the bot lost access — which is exactly when we
        # want to drop the subscription, so the trade-off is free.
        adapter, _ = _bare_adapter(bot_id=1000)
        handlers = _register_events_with_mocked_decorator(adapter)

        with patch(
            "backend.copilot.bot.threads.unsubscribe",
            new=AsyncMock(),
        ) as mock_unsub:
            await handlers["on_thread_remove"](_thread(555))

        mock_unsub.assert_awaited_once_with("discord", "555")

    @pytest.mark.asyncio
    async def test_swallows_redis_failures(self):
        adapter, _ = _bare_adapter(bot_id=1000)
        handlers = _register_events_with_mocked_decorator(adapter)

        with patch(
            "backend.copilot.bot.threads.unsubscribe",
            new=AsyncMock(side_effect=RuntimeError("redis down")),
        ):
            # Must not raise — cleanup is never critical-path.
            await handlers["on_thread_remove"](_thread(555))
