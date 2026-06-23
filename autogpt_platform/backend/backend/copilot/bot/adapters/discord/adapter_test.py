"""Tests for DiscordAdapter helpers that don't need a live gateway."""

from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest

from backend.copilot.bot.adapters.base import FileAttachment
from backend.copilot.bot.adapters.discord.adapter import (
    THREAD_HISTORY_CHAR_BUDGET,
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


# ── _message_text (forwarded messages) ─────────────────────────────────


def _snapshot(content: str = "", filenames: tuple[str, ...] = ()) -> MagicMock:
    snapshot = MagicMock()
    snapshot.content = content
    snapshot.attachments = [MagicMock(filename=name) for name in filenames]
    return snapshot


class TestMessageText:
    def test_plain_message_without_forward(self):
        adapter, _ = _bare_adapter(bot_id=1000)
        msg = _message("hello", [])
        msg.message_snapshots = []
        assert adapter._message_text(msg) == "hello"

    def test_forward_with_comment_includes_both(self):
        # The dangerous case Toran hit: a forward + comment used to arrive as
        # just the comment, losing the forwarded message entirely.
        adapter, _ = _bare_adapter(bot_id=1000)
        msg = _message("can you make a ticket for this?", [])
        msg.message_snapshots = [_snapshot("The original forwarded request")]
        result = adapter._message_text(msg)
        assert "can you make a ticket for this?" in result
        assert "[Forwarded message]" in result
        assert "The original forwarded request" in result

    def test_forward_without_comment_uses_forwarded_content(self):
        adapter, _ = _bare_adapter(bot_id=1000)
        msg = _message("", [])
        msg.message_snapshots = [_snapshot("Just the forwarded text")]
        result = adapter._message_text(msg)
        assert result.startswith("[Forwarded message]")
        assert "Just the forwarded text" in result

    def test_forward_notes_attachment_filenames(self):
        adapter, _ = _bare_adapter(bot_id=1000)
        msg = _message("look at this", [])
        msg.message_snapshots = [_snapshot("", filenames=("report.pdf",))]
        result = adapter._message_text(msg)
        assert "[Attached file: report.pdf]" in result


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
    async def test_send_file_attaches_bytes_with_display_name(self):
        adapter, client = _bare_adapter()
        channel = MagicMock(spec=discord.TextChannel)
        channel.send = AsyncMock()
        client.get_channel.return_value = channel

        await adapter.send_file(
            "123",
            "here's your result",
            FileAttachment(
                filename="chart.png", mime_type="image/png", content=b"\x89PNG"
            ),
        )

        channel.send.assert_awaited_once()
        args, kwargs = channel.send.await_args
        assert args == ("here's your result",)
        attached = kwargs["file"]
        assert isinstance(attached, discord.File)
        assert attached.filename == "chart.png"
        assert kwargs["tts"] is False

    @pytest.mark.asyncio
    async def test_send_file_drops_empty_caption_to_none(self):
        # discord.py rejects empty-string content alongside a file; we
        # collapse `text=""` to None so the upload still succeeds.
        adapter, client = _bare_adapter()
        channel = MagicMock(spec=discord.TextChannel)
        channel.send = AsyncMock()
        client.get_channel.return_value = channel

        await adapter.send_file(
            "123",
            "",
            FileAttachment(
                filename="x.bin", mime_type="application/octet-stream", content=b"x"
            ),
        )

        args, _ = channel.send.await_args
        assert args == (None,)

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
    async def test_fetches_user_thread_history_chronological(self):
        # Discord returns history newest-first; the adapter reverses it back to
        # chronological order, dropping its own outputs.
        adapter, _ = _bare_adapter(bot_id=1000)
        bot = _mention(1000, "AutoPilot")

        prior_1 = _message("first idea", [])
        prior_1.author = MagicMock(bot=False, id=2000, display_name="Alice")
        prior_2 = _message("<@1000> can ignore old bot ping", [bot])
        prior_2.author = MagicMock(bot=False, id=3000, display_name="Bob")
        bot_msg = _message("old bot output", [])
        bot_msg.author = MagicMock(bot=True, id=1000, display_name="AutoPilot")

        channel = MagicMock(spec=discord.Thread)
        # newest-first as the Discord API delivers it: Bob, (bot), Alice
        channel.history.return_value = _AsyncHistory([prior_2, bot_msg, prior_1])
        message = _message("<@1000> help", [bot])
        message.channel = channel

        history = await adapter._thread_history(message)

        channel.history.assert_called_once_with(
            limit=THREAD_HISTORY_LIMIT,
            before=message,
            oldest_first=False,
        )
        assert [entry.username for entry in history] == ["Alice", "Bob"]
        assert [entry.user_id for entry in history] == ["2000", "3000"]
        assert [entry.text for entry in history] == [
            "first idea",
            "can ignore old bot ping",
        ]

    @pytest.mark.asyncio
    async def test_drops_oldest_messages_past_the_size_budget(self):
        # A very long thread can't all fit in one copilot request. Keep the most
        # recent messages (newest-first scan, budget cut) and drop the oldest.
        adapter, _ = _bare_adapter(bot_id=1000)

        big = "x" * 5000  # 4 fit in the 24000-char budget, the 5th overflows
        newest_first = []
        for i in range(6, 0, -1):  # User6 (newest) .. User1 (oldest)
            msg = _message(big, [])
            msg.author = MagicMock(bot=False, id=i, display_name=f"User{i}")
            newest_first.append(msg)

        channel = MagicMock(spec=discord.Thread)
        channel.history.return_value = _AsyncHistory(newest_first)
        message = _message("help", [])
        message.channel = channel

        history = await adapter._thread_history(message)

        # Most-recent 4 kept, returned chronologically; oldest two dropped.
        assert [entry.username for entry in history] == [
            "User3",
            "User4",
            "User5",
            "User6",
        ]

    @pytest.mark.asyncio
    async def test_truncates_a_single_oversized_message(self):
        # If the latest message alone exceeds the budget, keep a truncated head
        # rather than drop all context or emit an oversized payload.
        adapter, _ = _bare_adapter(bot_id=1000)
        huge = "y" * (THREAD_HISTORY_CHAR_BUDGET + 6000)
        msg = _message(huge, [])
        msg.author = MagicMock(bot=False, id=2000, display_name="Alice")

        channel = MagicMock(spec=discord.Thread)
        channel.history.return_value = _AsyncHistory([msg])
        message = _message("help", [])
        message.channel = channel

        history = await adapter._thread_history(message)

        assert len(history) == 1
        assert len(history[0].text) <= THREAD_HISTORY_CHAR_BUDGET
        assert history[0].text.endswith("[message truncated]")

    @pytest.mark.asyncio
    async def test_truncated_newest_message_stops_older_messages(self):
        # Once the newest message is truncated to the budget, older messages
        # must not be appended into the whitespace the truncation freed up.
        adapter, _ = _bare_adapter(bot_id=1000)
        huge = "y" * (THREAD_HISTORY_CHAR_BUDGET + 6000)
        newest = _message(huge, [])
        newest.author = MagicMock(bot=False, id=2000, display_name="Newest")
        older = _message("older context", [])
        older.author = MagicMock(bot=False, id=3000, display_name="Older")

        channel = MagicMock(spec=discord.Thread)
        channel.history.return_value = _AsyncHistory([newest, older])  # newest-first
        message = _message("help", [])
        message.channel = channel

        history = await adapter._thread_history(message)

        assert [entry.username for entry in history] == ["Newest"]


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


# ── on_message: locked threads ──────────────────────────────────────────


class TestLockedThread:
    @pytest.mark.asyncio
    async def test_locked_thread_message_is_skipped(self):
        # A locked thread rejects bot sends, so processing the message would
        # just burn a turn and error on every reply. Bail before the handler.
        adapter, _ = _bare_adapter(bot_id=1000)
        callback = AsyncMock()
        adapter.on_message(callback)
        handlers = _register_events_with_mocked_decorator(adapter)

        thread = _thread(555)
        thread.locked = True
        msg = MagicMock()
        msg.author = MagicMock(id=2000, bot=False)
        msg.channel = thread

        await handlers["on_message"](msg)

        callback.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unlocked_thread_message_is_processed(self):
        adapter, _ = _bare_adapter(bot_id=1000)
        callback = AsyncMock()
        adapter.on_message(callback)
        handlers = _register_events_with_mocked_decorator(adapter)

        thread = _thread(555)
        thread.locked = False
        msg = MagicMock()
        msg.id = 999
        msg.author = MagicMock(id=2000, bot=False, display_name="Bently")
        msg.guild = MagicMock(id=111)
        msg.channel = thread
        msg.content = "hi"
        msg.mentions = []

        await handlers["on_message"](msg)

        callback.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_on_message_rewrites_referenced_links_in_text(self):
        # End-to-end through on_message: a channel @mention linking another
        # channel attaches that conversation and rewrites the raw link into a
        # readable #name in the forwarded text.
        adapter, client = _bare_adapter(bot_id=1000)
        callback = AsyncMock()
        adapter.on_message(callback)
        handlers = _register_events_with_mocked_decorator(adapter)

        ref = _referenced_channel(
            "general", 111, [_prior(2000, "Krz", "the decision was X")]
        )
        client.get_channel.return_value = ref

        guild = MagicMock(id=111)
        guild.get_member = MagicMock(return_value=MagicMock(spec=discord.Member))
        msg = MagicMock()
        msg.id = 999
        msg.author = MagicMock(id=2000, bot=False, display_name="Bently")
        msg.guild = guild
        msg.channel = MagicMock(id=555)  # a normal channel, not a Thread
        msg.content = "<@1000> read https://discord.com/channels/111/222/333"
        msg.mentions = [_mention(1000, "AutoPilot")]
        msg.message_snapshots = []

        await handlers["on_message"](msg)

        callback.assert_awaited_once()
        ctx = callback.await_args.args[0]
        assert len(ctx.referenced_conversations) == 1
        assert ctx.referenced_conversations[0].title == "general"
        assert "#general" in ctx.text
        assert "discord.com/channels" not in ctx.text


class TestReplyContext:
    @staticmethod
    def _replied(content: str, author_name: str = "Bently") -> MagicMock:
        replied = MagicMock(spec=discord.Message)
        replied.content = content
        replied.mentions = []
        replied.message_snapshots = []
        replied.author = MagicMock(display_name=author_name)
        return replied

    @pytest.mark.asyncio
    async def test_resolve_reply_uses_resolved_message(self):
        adapter, _ = _bare_adapter()
        replied = self._replied("fact about space")
        msg = MagicMock()
        msg.message_snapshots = []
        msg.reference = MagicMock(resolved=replied)
        assert await adapter._resolve_reply(msg) is replied

    @pytest.mark.asyncio
    async def test_no_reference_resolves_to_none(self):
        adapter, _ = _bare_adapter()
        msg = MagicMock()
        msg.reference = None
        assert await adapter._resolve_reply(msg) is None

    @pytest.mark.asyncio
    async def test_forward_is_not_treated_as_reply(self):
        # A forward also populates ``reference`` but is handled via snapshots.
        adapter, _ = _bare_adapter()
        msg = MagicMock()
        msg.message_snapshots = [MagicMock()]
        msg.reference = MagicMock(resolved=self._replied("x"))
        assert await adapter._resolve_reply(msg) is None

    @pytest.mark.asyncio
    async def test_fetches_reply_when_not_resolved(self):
        adapter, _ = _bare_adapter()
        replied = self._replied("fetched body")
        channel = MagicMock(spec=discord.TextChannel)
        channel.fetch_message = AsyncMock(return_value=replied)
        msg = MagicMock()
        msg.message_snapshots = []
        msg.reference = MagicMock(resolved=None, message_id=42)
        msg.channel = channel
        assert await adapter._resolve_reply(msg) is replied
        channel.fetch_message.assert_awaited_once_with(42)

    @pytest.mark.asyncio
    async def test_with_reply_context_prepends_quoted_message(self):
        adapter, _ = _bare_adapter()
        replied = self._replied("fact about space", author_name="AutoBoostBot")
        msg = MagicMock()
        msg.message_snapshots = []
        msg.reference = MagicMock(resolved=replied)
        out = await adapter._with_reply_context(msg, "can you tell me?")
        assert "[Replying to AutoBoostBot]" in out
        assert "fact about space" in out
        assert out.endswith("can you tell me?")

    @pytest.mark.asyncio
    async def test_with_reply_context_noop_without_reply(self):
        adapter, _ = _bare_adapter()
        msg = MagicMock()
        msg.reference = None
        assert await adapter._with_reply_context(msg, "hi") == "hi"

    @pytest.mark.asyncio
    async def test_on_message_includes_replied_message(self):
        adapter, _ = _bare_adapter(bot_id=1000)
        callback = AsyncMock()
        adapter.on_message(callback)
        handlers = _register_events_with_mocked_decorator(adapter)

        replied = self._replied("fact about space", author_name="AutoBoostBot")
        guild = MagicMock(id=111)
        guild.get_member = MagicMock(return_value=MagicMock(spec=discord.Member))
        msg = MagicMock()
        msg.id = 999
        msg.author = MagicMock(id=2000, bot=False, display_name="Bently")
        msg.guild = guild
        msg.channel = MagicMock(id=555)  # a normal channel, not a Thread
        msg.content = "<@1000> can you tell me?"
        msg.mentions = [_mention(1000, "AutoPilot")]
        msg.message_snapshots = []
        msg.reference = MagicMock(resolved=replied)

        await handlers["on_message"](msg)

        callback.assert_awaited_once()
        ctx = callback.await_args.args[0]
        assert "fact about space" in ctx.text
        assert "can you tell me?" in ctx.text

    @pytest.mark.asyncio
    async def test_links_inside_replied_message_are_not_fetched(self):
        # A link that only appears in the quoted reply (not the user's own
        # message) is context, not a request — it must not trigger a fetch or
        # get rewritten. Guards against scanning the combined text.
        adapter, client = _bare_adapter(bot_id=1000)
        callback = AsyncMock()
        adapter.on_message(callback)
        handlers = _register_events_with_mocked_decorator(adapter)

        replied = self._replied(
            "see https://discord.com/channels/111/222/333", author_name="Krz"
        )
        guild = MagicMock(id=111)
        guild.get_member = MagicMock(return_value=MagicMock(spec=discord.Member))
        msg = MagicMock()
        msg.id = 999
        msg.author = MagicMock(id=2000, bot=False, display_name="Bently")
        msg.guild = guild
        msg.channel = MagicMock(id=555)
        msg.content = "<@1000> thanks!"  # no link of its own
        msg.mentions = [_mention(1000, "AutoPilot")]
        msg.message_snapshots = []
        msg.reference = MagicMock(resolved=replied)

        await handlers["on_message"](msg)

        callback.assert_awaited_once()
        ctx = callback.await_args.args[0]
        # No channel history fetched, and the quoted link is left as-is.
        client.get_channel.assert_not_called()
        assert ctx.referenced_conversations == ()
        assert "https://discord.com/channels/111/222/333" in ctx.text

    @pytest.mark.asyncio
    async def test_links_inside_forwarded_message_are_not_fetched(self):
        # A link that only appears in a forwarded message is quoted context,
        # not the user's request — it must not be fetched or rewritten.
        adapter, client = _bare_adapter(bot_id=1000)
        callback = AsyncMock()
        adapter.on_message(callback)
        handlers = _register_events_with_mocked_decorator(adapter)

        guild = MagicMock(id=111)
        guild.get_member = MagicMock(return_value=MagicMock(spec=discord.Member))
        msg = MagicMock()
        msg.id = 999
        msg.author = MagicMock(id=2000, bot=False, display_name="Bently")
        msg.guild = guild
        msg.channel = MagicMock(id=555)
        msg.content = "<@1000> look at this"  # no link of its own
        msg.mentions = [_mention(1000, "AutoPilot")]
        msg.message_snapshots = [
            _snapshot("see https://discord.com/channels/111/222/333")
        ]
        msg.reference = None

        await handlers["on_message"](msg)

        callback.assert_awaited_once()
        ctx = callback.await_args.args[0]
        client.get_channel.assert_not_called()
        assert ctx.referenced_conversations == ()
        assert "[Forwarded message]" in ctx.text
        assert "https://discord.com/channels/111/222/333" in ctx.text


# ── Proactive output (backend → platform) ──────────────────────────────


def _guild_channel(channel_id: int, name: str, can_send: bool) -> MagicMock:
    channel = MagicMock(spec=discord.TextChannel)
    channel.id = channel_id
    channel.name = name
    perms = MagicMock()
    perms.send_messages = can_send
    channel.permissions_for = MagicMock(return_value=perms)
    return channel


def _guild_with_channels(
    guild_id: int, name: str, channels: list[MagicMock]
) -> MagicMock:
    guild = MagicMock()
    guild.id = guild_id
    guild.name = name
    guild.me = MagicMock()
    guild.text_channels = channels
    return guild


class TestProactiveOutput:
    @pytest.mark.asyncio
    async def test_list_text_channels_filters_servers_and_permissions(self):
        adapter, client = _bare_adapter()
        g1 = _guild_with_channels(
            111,
            "Guild One",
            [
                _guild_channel(10, "general", True),
                _guild_channel(11, "locked", False),
            ],
        )
        g2 = _guild_with_channels(222, "Guild Two", [_guild_channel(20, "other", True)])
        client.guilds = [g1, g2]

        result = await adapter.list_text_channels(("111",))

        assert [(c.id, c.name, c.server_id) for c in result] == [
            ("10", "general", "111")
        ]

    @pytest.mark.asyncio
    async def test_get_channel_server_id_returns_guild(self):
        adapter, client = _bare_adapter()
        channel = MagicMock(spec=discord.TextChannel)
        channel.guild = MagicMock(id=111)
        client.get_channel.return_value = channel

        assert await adapter.get_channel_server_id("10") == "111"

    @pytest.mark.asyncio
    async def test_get_channel_server_id_none_when_missing(self):
        adapter, client = _bare_adapter()
        client.get_channel.return_value = None
        client.fetch_channel = AsyncMock(
            side_effect=discord.NotFound(MagicMock(status=404), "gone")
        )
        assert await adapter.get_channel_server_id("10") is None

    @pytest.mark.asyncio
    async def test_post_channel_message_returns_ref_with_url(self):
        adapter, client = _bare_adapter()
        channel = MagicMock(spec=discord.TextChannel)
        sent = MagicMock(id=999, jump_url="https://discord.com/channels/1/2/999")
        channel.send = AsyncMock(return_value=sent)
        client.get_channel.return_value = channel

        ref = await adapter.post_channel_message("10", "hello")

        assert ref is not None
        assert ref.id == "999"
        assert ref.url == "https://discord.com/channels/1/2/999"

    @pytest.mark.asyncio
    async def test_create_channel_thread_creates_and_posts(self):
        adapter, client = _bare_adapter()
        channel = MagicMock(spec=discord.TextChannel)
        thread = MagicMock(spec=discord.Thread)
        thread.id = 555
        thread.jump_url = "https://discord.com/channels/1/555"
        thread.send = AsyncMock()
        channel.create_thread = AsyncMock(return_value=thread)
        client.get_channel.return_value = channel

        ref = await adapter.create_channel_thread("10", "Monday update", "body")

        assert ref is not None
        assert ref.id == "555"
        channel.create_thread.assert_awaited_once()
        thread.send.assert_awaited()

    @pytest.mark.asyncio
    async def test_create_channel_thread_rejects_non_text_channel(self):
        adapter, client = _bare_adapter()
        client.get_channel.return_value = MagicMock(spec=discord.Thread)

        assert await adapter.create_channel_thread("10", "x", "body") is None

    @pytest.mark.asyncio
    async def test_send_chunked_splits_long_text(self):
        adapter, client = _bare_adapter()
        channel = MagicMock(spec=discord.TextChannel)
        channel.send = AsyncMock(
            side_effect=lambda *a, **k: MagicMock(id=1, jump_url="u")
        )
        client.get_channel.return_value = channel

        long_text = ("word " * 1000).strip()
        ref = await adapter.post_channel_message("10", long_text)

        assert ref is not None
        # 5000 chars at ~1900/chunk → more than one send.
        assert channel.send.await_count >= 2

    @pytest.mark.asyncio
    async def test_post_message_keeps_first_chunk_on_later_failure(self):
        # First chunk posts, a later chunk 500s: the already-posted message is
        # surfaced (partial success) rather than discarded into a retry-duplicate.
        adapter, client = _bare_adapter()
        channel = MagicMock(spec=discord.TextChannel)
        first = MagicMock(id=111, jump_url="u1")
        channel.send = AsyncMock(
            side_effect=[first, discord.HTTPException(MagicMock(status=500), "boom")]
        )
        client.get_channel.return_value = channel

        ref = await adapter.post_channel_message("10", ("word " * 1000).strip())

        assert ref is not None
        assert ref.id == "111"
        # Prove the later (failing) chunk was actually reached, so chunk-sizing
        # changes can't quietly turn this into a single-send happy path.
        assert channel.send.await_count >= 2

    @pytest.mark.asyncio
    async def test_post_message_returns_none_when_first_chunk_fails(self):
        adapter, client = _bare_adapter()
        channel = MagicMock(spec=discord.TextChannel)
        channel.send = AsyncMock(
            side_effect=discord.HTTPException(MagicMock(status=500), "boom")
        )
        client.get_channel.return_value = channel

        assert await adapter.post_channel_message("10", "hi") is None

    @pytest.mark.asyncio
    async def test_create_thread_returns_ref_when_content_post_fails(self):
        # Thread created but posting body fails → still return the thread ref so
        # the caller doesn't retry and spawn a duplicate thread.
        adapter, client = _bare_adapter()
        channel = MagicMock(spec=discord.TextChannel)
        thread = MagicMock(spec=discord.Thread)
        thread.id = 777
        thread.jump_url = "https://discord.com/t/777"
        thread.send = AsyncMock(
            side_effect=discord.HTTPException(MagicMock(status=500), "boom")
        )
        channel.create_thread = AsyncMock(return_value=thread)
        client.get_channel.return_value = channel

        ref = await adapter.create_channel_thread("10", "Monday", "body")

        assert ref is not None
        assert ref.id == "777"


# ── Referenced-conversation fetch ──────────────────────────────────────


def _referenced_channel(
    name: str, guild_id: int, priors: list[MagicMock], *, can_access: bool = True
) -> MagicMock:
    channel = MagicMock(spec=discord.TextChannel)
    channel.name = name
    channel.guild = MagicMock(id=guild_id)
    channel.history = MagicMock(return_value=_AsyncHistory(priors))
    perms = MagicMock(view_channel=can_access, read_message_history=can_access)
    channel.permissions_for = MagicMock(return_value=perms)
    return channel


def _referenced_thread(
    guild_id: int,
    priors: list[MagicMock],
    *,
    is_private: bool = True,
    is_member: bool = True,
    manage_threads: bool = False,
) -> MagicMock:
    thread = MagicMock(spec=discord.Thread)
    thread.name = "secret-thread"
    thread.guild = MagicMock(id=guild_id)
    thread.history = MagicMock(return_value=_AsyncHistory(priors))
    thread.is_private = MagicMock(return_value=is_private)
    perms = MagicMock(
        view_channel=True, read_message_history=True, manage_threads=manage_threads
    )
    thread.permissions_for = MagicMock(return_value=perms)
    if is_member:
        thread.fetch_member = AsyncMock(return_value=MagicMock())
    else:
        thread.fetch_member = AsyncMock(
            side_effect=discord.NotFound(MagicMock(status=404), "not a member")
        )
    return thread


def _prior(author_id: int, display_name: str, content: str) -> MagicMock:
    msg = MagicMock()
    msg.author = MagicMock(id=author_id, display_name=display_name)
    msg.content = content
    msg.mentions = []
    return msg


def _incoming(guild_id: int | None, channel_id: int) -> MagicMock:
    message = MagicMock()
    message.channel = MagicMock(id=channel_id)
    message.author = MagicMock(id=42)
    if guild_id is None:
        message.guild = None
    else:
        guild = MagicMock(id=guild_id)
        # By default the requester is a member of the guild; tests override
        # get_member / channel perms to exercise the access checks.
        guild.get_member = MagicMock(return_value=MagicMock(spec=discord.Member))
        message.guild = guild
    return message


class TestFetchReferencedConversations:
    @pytest.mark.asyncio
    async def test_fetches_same_guild_referenced_thread(self):
        adapter, client = _bare_adapter()
        ref = _referenced_channel(
            "Release v0.6.61", 111, [_prior(2000, "Krz", "bump the version")]
        )
        client.get_channel.return_value = ref

        result = await adapter._fetch_referenced_conversations(
            _incoming(111, 555),
            "find my mentions in https://discord.com/channels/111/222/333",
        )

        assert len(result) == 1
        assert result[0].title == "Release v0.6.61"
        assert result[0].channel_id == "222"
        assert result[0].messages[0].text == "bump the version"

    @pytest.mark.asyncio
    async def test_uses_message_author_when_member_cache_is_empty(self):
        # The bot runs without the privileged members intent, so
        # guild.get_member can return None. message.author is the authoritative
        # Member for guild messages and must be used instead.
        adapter, client = _bare_adapter()
        ref = _referenced_channel("general", 111, [_prior(1, "A", "hi there")])
        client.get_channel.return_value = ref
        message = _incoming(111, 555)
        message.guild.get_member = MagicMock(return_value=None)  # cache miss
        message.author = MagicMock(spec=discord.Member, id=42)

        result = await adapter._fetch_referenced_conversations(
            message, "https://discord.com/channels/111/222/333"
        )

        assert len(result) == 1
        assert result[0].messages[0].text == "hi there"

    @pytest.mark.asyncio
    async def test_fetches_specific_message_in_current_channel(self):
        # A permalink to a specific message in the channel the user is posting
        # in is a real "read this exact message" request, not redundant context.
        adapter, client = _bare_adapter()
        ref = _referenced_channel("autopilot", 111, [_prior(1, "A", "the answer")])
        client.get_channel.return_value = ref

        result = await adapter._fetch_referenced_conversations(
            _incoming(111, 222),  # current channel == the linked channel
            "https://discord.com/channels/111/222/333",
        )

        assert len(result) == 1
        assert result[0].messages[0].text == "the answer"

    @pytest.mark.asyncio
    async def test_skips_private_thread_when_requester_not_member(self):
        # The bot is in the private thread, but the requester isn't — its
        # content must not leak to them.
        adapter, client = _bare_adapter()
        thread = _referenced_thread(111, [_prior(1, "A", "secret")], is_member=False)
        client.get_channel.return_value = thread

        result = await adapter._fetch_referenced_conversations(
            _incoming(111, 555), "https://discord.com/channels/111/222/333"
        )

        assert result == ()

    @pytest.mark.asyncio
    async def test_reads_private_thread_when_requester_is_member(self):
        adapter, client = _bare_adapter()
        thread = _referenced_thread(111, [_prior(1, "A", "secret")], is_member=True)
        client.get_channel.return_value = thread

        result = await adapter._fetch_referenced_conversations(
            _incoming(111, 555), "https://discord.com/channels/111/222/333"
        )

        assert len(result) == 1
        assert result[0].messages[0].text == "secret"

    @pytest.mark.asyncio
    async def test_reads_private_thread_with_manage_threads(self):
        # manage_threads grants access like it does in the Discord client.
        adapter, client = _bare_adapter()
        thread = _referenced_thread(
            111, [_prior(1, "A", "secret")], is_member=False, manage_threads=True
        )
        client.get_channel.return_value = thread

        result = await adapter._fetch_referenced_conversations(
            _incoming(111, 555), "https://discord.com/channels/111/222/333"
        )

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_skips_cross_guild_reference(self):
        adapter, client = _bare_adapter()
        ref = _referenced_channel("Other", 999, [_prior(2000, "X", "secret")])
        client.get_channel.return_value = ref

        result = await adapter._fetch_referenced_conversations(
            _incoming(111, 555), "https://discord.com/channels/999/222/333"
        )

        assert result == ()

    @pytest.mark.asyncio
    async def test_skips_channel_requester_cannot_view(self):
        # The bot's gateway can read this channel, but the requesting member
        # can't — its history must not leak to them.
        adapter, client = _bare_adapter()
        ref = _referenced_channel(
            "Private", 111, [_prior(2000, "X", "secret")], can_access=False
        )
        client.get_channel.return_value = ref

        result = await adapter._fetch_referenced_conversations(
            _incoming(111, 555), "https://discord.com/channels/111/222/333"
        )

        assert result == ()

    @pytest.mark.asyncio
    async def test_skips_when_requester_not_a_member(self):
        adapter, client = _bare_adapter()
        message = _incoming(111, 555)
        message.guild.get_member = MagicMock(return_value=None)

        result = await adapter._fetch_referenced_conversations(
            message, "https://discord.com/channels/111/222/333"
        )

        assert result == ()
        client.get_channel.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_reference_makes_no_fetch(self):
        adapter, client = _bare_adapter()
        result = await adapter._fetch_referenced_conversations(
            _incoming(111, 555), "just a normal message"
        )
        assert result == ()
        client.get_channel.assert_not_called()

    @pytest.mark.asyncio
    async def test_dm_with_no_guild_skips(self):
        adapter, client = _bare_adapter()
        result = await adapter._fetch_referenced_conversations(
            _incoming(None, 555), "<#222>"
        )
        assert result == ()
        client.get_channel.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_non_text_channel(self):
        # A reference that resolves to e.g. a voice/category channel — not
        # something we can read history from.
        adapter, client = _bare_adapter()
        client.get_channel.return_value = MagicMock()  # not a TextChannel/Thread
        result = await adapter._fetch_referenced_conversations(
            _incoming(111, 555), "https://discord.com/channels/111/222/333"
        )
        assert result == ()

    @pytest.mark.asyncio
    async def test_returns_nothing_when_history_errors(self):
        adapter, client = _bare_adapter()
        ref = _referenced_channel("Boom", 111, [])
        ref.history = MagicMock(side_effect=discord.HTTPException(MagicMock(), "boom"))
        client.get_channel.return_value = ref
        result = await adapter._fetch_referenced_conversations(
            _incoming(111, 555), "https://discord.com/channels/111/222/333"
        )
        assert result == ()

    @pytest.mark.asyncio
    async def test_skips_channel_with_no_readable_messages(self):
        adapter, client = _bare_adapter()
        ref = _referenced_channel("Empty", 111, [])
        client.get_channel.return_value = ref
        result = await adapter._fetch_referenced_conversations(
            _incoming(111, 555), "https://discord.com/channels/111/222/333"
        )
        assert result == ()
