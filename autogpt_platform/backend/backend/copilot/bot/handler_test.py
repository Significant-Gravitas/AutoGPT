"""Tests for the platform-agnostic message handler."""

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.util.exceptions import DuplicateChatMessageError, NotFoundError

from .adapters.base import ChannelType, MessageContext, MessageHistoryEntry
from .bot_backend import LinkTokenResult, ResolveResult
from .handler import MessageHandler, TargetState, build_thread_name, clamp_thread_name


def _ctx(
    *,
    channel_type: ChannelType = "channel",
    server_id: str | None = "guild-1",
    channel_id: str = "chan-1",
    message_id: str = "msg-1",
    user_id: str = "user-1",
    username: str = "Bently",
    text: str = "hello bot",
    bot_mentioned: bool = False,
    thread_history: tuple[MessageHistoryEntry, ...] = (),
) -> MessageContext:
    return MessageContext(
        platform="discord",
        channel_type=channel_type,
        server_id=server_id,
        channel_id=channel_id,
        message_id=message_id,
        user_id=user_id,
        username=username,
        text=text,
        bot_mentioned=bot_mentioned,
        thread_history=thread_history,
    )


def _adapter() -> MagicMock:
    adapter = MagicMock()
    adapter.chunk_flush_at = 1900
    adapter.send_message = AsyncMock()
    adapter.send_reply = AsyncMock()
    adapter.send_link = AsyncMock()
    adapter.start_typing = AsyncMock()
    adapter.stop_typing = AsyncMock()
    adapter.create_thread = AsyncMock(return_value="thread-new")
    adapter.rename_thread = AsyncMock(return_value=True)
    return adapter


def _api(*, server_linked: bool = True, user_linked: bool = True) -> MagicMock:
    api = MagicMock()
    api.resolve_server = AsyncMock(return_value=ResolveResult(linked=server_linked))
    api.resolve_user = AsyncMock(return_value=ResolveResult(linked=user_linked))
    api.create_user_link_token = AsyncMock(
        return_value=LinkTokenResult(
            token="t",
            link_url="https://example.com/link/t",
            expires_at="2099-01-01T00:00:00Z",
        )
    )

    async def _empty_stream(*args, **kwargs):
        if False:
            yield ""

    api.stream_chat = _empty_stream
    return api


@contextlib.contextmanager
def _capture_handler_tasks():
    """Capture handler-spawned tasks at creation time.

    Rename tasks self-discard from `handler._rename_tasks` via a done-
    callback, so reading that set post-hoc is racy.
    """
    tasks: list[asyncio.Task[None]] = []
    real_create_task = asyncio.create_task

    def _capturing(coro, **kwargs):
        task = real_create_task(coro, **kwargs)
        tasks.append(task)
        return task

    with patch(
        "backend.copilot.bot.handler.asyncio.create_task",
        side_effect=_capturing,
    ):
        yield tasks


class TestEmptyMessage:
    @pytest.mark.asyncio
    async def test_channel_mention_without_text_gets_nudge(self):
        handler = MessageHandler(_api())
        adapter = _adapter()
        await handler.handle(_ctx(text="   "), adapter)
        adapter.send_reply.assert_awaited_once()
        adapter.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_dm_is_silently_dropped(self):
        handler = MessageHandler(_api())
        adapter = _adapter()
        await handler.handle(_ctx(channel_type="dm", text=""), adapter)
        adapter.send_reply.assert_not_awaited()
        adapter.send_message.assert_not_awaited()


class TestEnsureLinked:
    @pytest.mark.asyncio
    async def test_unlinked_server_tells_user_to_setup(self):
        handler = MessageHandler(_api(server_linked=False))
        adapter = _adapter()
        await handler.handle(_ctx(), adapter)
        call_args = adapter.send_message.await_args.args
        assert "isn't linked" in call_args[1]
        assert "/setup" in call_args[1]

    @pytest.mark.asyncio
    async def test_unlinked_unsubscribed_thread_is_ignored(self):
        api = _api(server_linked=False)
        handler = MessageHandler(api)
        adapter = _adapter()
        with patch(
            "backend.copilot.bot.handler.threads.is_subscribed",
            new=AsyncMock(return_value=False),
        ):
            await handler.handle(_ctx(channel_type="thread"), adapter)

        api.resolve_server.assert_not_awaited()
        adapter.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unlinked_mentioned_thread_tells_user_to_setup(self):
        api = _api(server_linked=False)
        handler = MessageHandler(api)
        adapter = _adapter()
        with (
            patch(
                "backend.copilot.bot.handler.threads.is_subscribed",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "backend.copilot.bot.handler.threads.subscribe", new=AsyncMock()
            ) as subscribe,
        ):
            await handler.handle(
                _ctx(channel_type="thread", bot_mentioned=True), adapter
            )

        subscribe.assert_not_awaited()
        adapter.send_message.assert_awaited_once()
        assert "/setup" in adapter.send_message.await_args.args[1]

    @pytest.mark.asyncio
    async def test_unlinked_dm_prompts_link_flow(self):
        handler = MessageHandler(_api(user_linked=False))
        adapter = _adapter()
        await handler.handle(_ctx(channel_type="dm", server_id=None), adapter)
        adapter.send_link.assert_awaited_once()
        assert adapter.send_link.await_args.kwargs["link_url"].startswith(
            "https://example.com/link/"
        )

    @pytest.mark.asyncio
    async def test_non_dm_without_server_id_is_rejected(self):
        handler = MessageHandler(_api())
        adapter = _adapter()
        await handler.handle(_ctx(server_id=None), adapter)
        # Guard short-circuits before calling resolve_server.
        handler._api.resolve_server.assert_not_awaited()
        adapter.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_backend_error_in_resolve_produces_message(self):
        api = _api()
        api.resolve_server = AsyncMock(side_effect=NotFoundError("boom"))
        handler = MessageHandler(api)
        adapter = _adapter()
        await handler.handle(_ctx(), adapter)
        adapter.send_message.assert_awaited_once()
        assert "went wrong" in adapter.send_message.await_args.args[1].lower()


class TestResolveTarget:
    @pytest.mark.asyncio
    async def test_dm_reuses_channel_id(self):
        handler = MessageHandler(_api())
        adapter = _adapter()
        ctx = _ctx(channel_type="dm", server_id=None, channel_id="dm-42")
        result = await handler._resolve_target(ctx, adapter)
        assert result == "dm-42"

    @pytest.mark.asyncio
    async def test_resolve_target_reuses_thread_after_subscription_gate(self):
        handler = MessageHandler(_api())
        adapter = _adapter()
        ctx = _ctx(channel_type="thread", channel_id="thread-old")
        assert await handler._resolve_target(ctx, adapter) == "thread-old"

    @pytest.mark.asyncio
    async def test_subscribed_thread_keeps_channel(self):
        handler = MessageHandler(_api())
        adapter = _adapter()
        ctx = _ctx(channel_type="thread", channel_id="thread-ok")
        assert await handler._resolve_target(ctx, adapter) == "thread-ok"

    @pytest.mark.asyncio
    async def test_channel_creates_and_subscribes_thread(self):
        handler = MessageHandler(_api())
        adapter = _adapter()
        adapter.create_thread = AsyncMock(return_value="thread-created")
        with patch(
            "backend.copilot.bot.handler.threads.subscribe", new=AsyncMock()
        ) as subscribe:
            result = await handler._resolve_target(_ctx(), adapter)
        assert result == "thread-created"
        subscribe.assert_awaited_once_with("discord", "thread-created")
        assert adapter.create_thread.await_args.args[2] == "AutoPilot: hello bot"

    @pytest.mark.asyncio
    async def test_channel_falls_back_to_parent_when_thread_creation_fails(self):
        handler = MessageHandler(_api())
        adapter = _adapter()
        adapter.create_thread = AsyncMock(return_value=None)
        result = await handler._resolve_target(_ctx(channel_id="parent-chan"), adapter)
        assert result == "parent-chan"


class TestThreadAdoption:
    @pytest.mark.asyncio
    async def test_mentioned_unsubscribed_thread_is_subscribed(self):
        handler = MessageHandler(_api())
        adapter = _adapter()
        enqueue = AsyncMock()

        with (
            patch.object(handler, "_enqueue_and_process", new=enqueue),
            patch(
                "backend.copilot.bot.handler.threads.is_subscribed",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "backend.copilot.bot.handler.threads.subscribe", new=AsyncMock()
            ) as subscribe,
        ):
            await handler.handle(
                _ctx(
                    channel_type="thread",
                    channel_id="thread-existing",
                    bot_mentioned=True,
                ),
                adapter,
            )

        subscribe.assert_awaited_once_with("discord", "thread-existing")
        enqueue.assert_awaited_once()

    def test_thread_history_is_included_in_message_text(self):
        handler = MessageHandler(_api())
        text = handler._message_text(
            _ctx(
                channel_type="thread",
                text="what should we do?",
                thread_history=(
                    MessageHistoryEntry("Alice", "u-1", "I think option A"),
                    MessageHistoryEntry("Bob", "u-2", "Option B is safer"),
                ),
            )
        )

        assert "Recent thread context" in text
        assert "Alice (Discord user ID: u-1)" in text
        assert "I think option A" in text
        assert "Current message" in text
        assert "what should we do?" in text


class TestBatching:
    @pytest.mark.asyncio
    async def test_concurrent_message_queues_when_processing(self):
        """Second caller with processing=True returns without starting a new stream."""
        handler = MessageHandler(_api())
        adapter = _adapter()
        state = TargetState(processing=True)
        handler._targets["target-1"] = state

        await handler._enqueue_and_process(_ctx(text="second"), adapter, "target-1")

        assert state.processing is True
        assert state.pending == [("Bently", "user-1", "second")]

    @pytest.mark.asyncio
    async def test_target_state_cleared_after_drain(self):
        handler = MessageHandler(_api())
        adapter = _adapter()

        stream_calls: list[list] = []

        async def fake_stream_batch(batch, ctx, ad, tid):
            stream_calls.append(list(batch))

        handler._stream_batch = fake_stream_batch  # type: ignore[method-assign]

        await handler._enqueue_and_process(_ctx(text="hello"), adapter, "target-1")
        assert stream_calls == [[("Bently", "user-1", "hello")]]
        # Dict entry should be gone once processing finishes with empty pending.
        assert "target-1" not in handler._targets

    @pytest.mark.asyncio
    async def test_drain_loop_picks_up_appended_messages(self):
        """Messages appended to pending mid-drain are processed in the next iter."""
        handler = MessageHandler(_api())
        adapter = _adapter()

        state = TargetState()
        handler._targets["target-1"] = state

        seen: list[list] = []

        async def fake_stream_batch(batch, ctx, ad, tid):
            seen.append(list(batch))
            if len(seen) == 1:
                # Simulate another caller appending during the first stream.
                state.pending.append(("Later", "u2", "follow-up"))

        handler._stream_batch = fake_stream_batch  # type: ignore[method-assign]
        await handler._enqueue_and_process(_ctx(text="first"), adapter, "target-1")

        assert seen == [
            [("Bently", "user-1", "first")],
            [("Later", "u2", "follow-up")],
        ]
        assert "target-1" not in handler._targets

    @pytest.mark.asyncio
    async def test_duplicate_message_is_silently_dropped(self):
        api = _api()

        async def duplicate_stream(*args, **kwargs):
            raise DuplicateChatMessageError("in flight")
            yield ""  # pragma: no cover

        api.stream_chat = duplicate_stream
        handler = MessageHandler(api)
        adapter = _adapter()

        with patch(
            "backend.copilot.bot.handler.get_redis_async",
            new=AsyncMock(return_value=AsyncMock(get=AsyncMock(return_value=None))),
        ):
            await handler._stream_batch(
                [("Bently", "u1", "hi")], _ctx(), adapter, "target-1"
            )

        adapter.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_setup_requirements_sends_web_handoff_link(self):
        api = _api()

        async def setup_stream(*args, **kwargs):
            await kwargs["on_setup_required"](
                "session-1",
                {
                    "type": "setup_requirements",
                    "message": "Connect GitHub to continue.",
                },
                "connect_integration",
            )
            yield "Once connected, I can retry."

        api.stream_chat = setup_stream
        handler = MessageHandler(api)
        adapter = _adapter()

        fake_settings = MagicMock()
        fake_settings.config.frontend_base_url = "https://app.example.com"
        fake_settings.config.platform_base_url = ""

        with (
            patch(
                "backend.copilot.bot.handler.get_redis_async",
                new=AsyncMock(return_value=AsyncMock(get=AsyncMock(return_value=None))),
            ),
            patch("backend.copilot.bot.handler.Settings", return_value=fake_settings),
        ):
            await handler._stream_batch(
                [("Bently", "u1", "hi")], _ctx(), adapter, "target-1"
            )

        adapter.send_link.assert_awaited_once()
        assert adapter.send_link.await_args.kwargs["link_label"] == "Open AutoGPT"
        assert (
            adapter.send_link.await_args.kwargs["link_url"]
            == "https://app.example.com/copilot?sessionId=session-1"
        )
        assert "Connect GitHub" in adapter.send_link.await_args.args[1]
        # Trailing yield after on_setup_required still flushes at end-of-stream.
        adapter.send_message.assert_awaited_once()
        assert "Once connected, I can retry." in adapter.send_message.await_args.args[1]

    @pytest.mark.asyncio
    async def test_setup_requirements_without_text_does_not_send_empty_fallback(self):
        api = _api()

        async def setup_stream(*args, **kwargs):
            await kwargs["on_setup_required"](
                "session-1",
                {
                    "type": "setup_requirements",
                    "message": "Connect GitHub to continue.",
                },
                "connect_integration",
            )
            if False:
                yield ""

        api.stream_chat = setup_stream
        handler = MessageHandler(api)
        adapter = _adapter()

        fake_settings = MagicMock()
        fake_settings.config.frontend_base_url = "https://app.example.com"
        fake_settings.config.platform_base_url = ""

        with (
            patch(
                "backend.copilot.bot.handler.get_redis_async",
                new=AsyncMock(return_value=AsyncMock(get=AsyncMock(return_value=None))),
            ),
            patch("backend.copilot.bot.handler.Settings", return_value=fake_settings),
        ):
            await handler._stream_batch(
                [("Bently", "u1", "hi")], _ctx(), adapter, "target-1"
            )

        adapter.send_link.assert_awaited_once()
        adapter.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_channel_thread_renames_from_generated_chat_title(self):
        api = _api()

        async def title_stream(*args, **kwargs):
            await kwargs["on_session_id"]("session-1")
            yield "Done"

        api.stream_chat = title_stream
        api.get_session_title = AsyncMock(return_value="Generated Web Title")
        handler = MessageHandler(api)
        adapter = _adapter()
        redis = AsyncMock(get=AsyncMock(return_value=None), set=AsyncMock())

        with (
            patch(
                "backend.copilot.bot.handler.get_redis_async",
                new=AsyncMock(return_value=redis),
            ),
            patch(
                "backend.copilot.bot.handler.asyncio.sleep",
                new=AsyncMock(),
            ),
            _capture_handler_tasks() as captured,
        ):
            await handler._stream_batch(
                [("Bently", "u1", "hi")],
                _ctx(channel_type="channel", channel_id="parent-1"),
                adapter,
                "thread-1",
            )
            await asyncio.gather(*captured, return_exceptions=True)

        api.get_session_title.assert_awaited_once_with("session-1")
        adapter.rename_thread.assert_awaited_once_with(
            "thread-1", "Generated Web Title"
        )

    @pytest.mark.asyncio
    async def test_thread_rename_retries_when_title_not_ready_yet(self):
        api = _api()

        async def title_stream(*args, **kwargs):
            await kwargs["on_session_id"]("session-2")
            yield "Done"

        api.stream_chat = title_stream
        # First two polls return None (title not generated yet), third
        # returns the real title.
        api.get_session_title = AsyncMock(side_effect=[None, None, "Late Title"])
        handler = MessageHandler(api)
        adapter = _adapter()
        redis = AsyncMock(get=AsyncMock(return_value=None), set=AsyncMock())

        with (
            patch(
                "backend.copilot.bot.handler.get_redis_async",
                new=AsyncMock(return_value=redis),
            ),
            patch(
                "backend.copilot.bot.handler.asyncio.sleep",
                new=AsyncMock(),
            ),
            _capture_handler_tasks() as captured,
        ):
            await handler._stream_batch(
                [("Bently", "u1", "hi")],
                _ctx(channel_type="channel", channel_id="parent-1"),
                adapter,
                "thread-2",
            )
            await asyncio.gather(*captured, return_exceptions=True)

        assert api.get_session_title.await_count == 3
        adapter.rename_thread.assert_awaited_once_with("thread-2", "Late Title")

    @pytest.mark.asyncio
    async def test_thread_rename_keeps_retrying_after_transient_exception(self):
        api = _api()

        async def title_stream(*args, **kwargs):
            await kwargs["on_session_id"]("session-3")
            yield "Done"

        api.stream_chat = title_stream
        # Simulate a transient backend error on the first attempt; the
        # retry should still go through and pick up the title.
        api.get_session_title = AsyncMock(
            side_effect=[RuntimeError("transient"), "Recovered Title"]
        )
        handler = MessageHandler(api)
        adapter = _adapter()
        redis = AsyncMock(get=AsyncMock(return_value=None), set=AsyncMock())

        with (
            patch(
                "backend.copilot.bot.handler.get_redis_async",
                new=AsyncMock(return_value=redis),
            ),
            patch(
                "backend.copilot.bot.handler.asyncio.sleep",
                new=AsyncMock(),
            ),
            _capture_handler_tasks() as captured,
        ):
            await handler._stream_batch(
                [("Bently", "u1", "hi")],
                _ctx(channel_type="channel", channel_id="parent-1"),
                adapter,
                "thread-3",
            )
            await asyncio.gather(*captured, return_exceptions=True)

        assert api.get_session_title.await_count == 2
        adapter.rename_thread.assert_awaited_once_with("thread-3", "Recovered Title")


class TestStreamFallback:
    """Covers the empty-response fallback, including the boundary-flush bug
    where prior code posted 'AutoPilot didn't produce a response' even though
    content had already been flushed mid-stream.
    """

    @staticmethod
    def _redis_patch():
        return patch(
            "backend.copilot.bot.handler.get_redis_async",
            new=AsyncMock(return_value=AsyncMock(get=AsyncMock(return_value=None))),
        )

    @pytest.mark.asyncio
    async def test_empty_stream_sends_fallback(self):
        api = _api()

        async def empty(*args, **kwargs):
            if False:
                yield ""

        api.stream_chat = empty
        handler = MessageHandler(api)
        adapter = _adapter()

        with TestStreamFallback._redis_patch():
            await handler._stream_batch(
                [("Bently", "u1", "hi")], _ctx(), adapter, "target-1"
            )

        msgs = [c.args[1] for c in adapter.send_message.await_args_list]
        assert any("didn't produce a response" in m for m in msgs)

    @pytest.mark.asyncio
    async def test_whitespace_only_stream_sends_fallback(self):
        api = _api()

        async def whitespace(*args, **kwargs):
            yield "   "
            yield "\n\n"

        api.stream_chat = whitespace
        handler = MessageHandler(api)
        adapter = _adapter()

        with TestStreamFallback._redis_patch():
            await handler._stream_batch(
                [("Bently", "u1", "hi")], _ctx(), adapter, "target-1"
            )

        msgs = [c.args[1] for c in adapter.send_message.await_args_list]
        assert any("didn't produce a response" in m for m in msgs)

    @pytest.mark.asyncio
    async def test_content_flushed_mid_stream_does_not_trigger_fallback(self):
        """Regression: before the fix, a response that flushed exactly at a
        boundary left buffer == "" and the fallback fired after real content
        had already been posted.
        """
        api = _api()
        adapter = _adapter()
        adapter.chunk_flush_at = 50

        async def streaming_content(*args, **kwargs):
            # Exactly flush_at chars → split_at_boundary returns the whole
            # payload as the post and an empty remainder, so the stream ends
            # with buffer == "". That USED to fall into the `elif not buffer`
            # branch and send the "didn't produce a response" fallback.
            yield "x" * 50

        api.stream_chat = streaming_content
        handler = MessageHandler(api)

        with TestStreamFallback._redis_patch():
            await handler._stream_batch(
                [("Bently", "u1", "hi")], _ctx(), adapter, "target-1"
            )

        msgs = [c.args[1] for c in adapter.send_message.await_args_list]
        assert not any("didn't produce a response" in m for m in msgs)
        assert msgs == ["x" * 50]


class TestThreadNames:
    def test_build_thread_name_from_prompt(self):
        assert (
            build_thread_name("  tell me\nabout space  ", "Bently")
            == "AutoPilot: tell me about space"
        )

    def test_build_thread_name_truncates_to_discord_limit(self):
        name = build_thread_name("x" * 200, "Bently")
        assert len(name) <= 100
        assert name.startswith("AutoPilot: ")
        assert name.endswith("...")

    def test_clamp_thread_name_handles_generated_titles(self):
        assert clamp_thread_name("  Generated\nWeb   Title  ") == "Generated Web Title"

    def test_clamp_thread_name_falls_back_when_blank(self):
        assert clamp_thread_name("   ") == "AutoPilot Chat"
