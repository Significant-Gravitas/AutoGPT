"""Tests for the per-turn streaming helpers, focused on live draft previews."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from .adapters.base import ChannelType, MessageContext, StreamDraftOutcome
from .turn_stream import DraftStreamer, TurnStreamer

_MODULE = "backend.copilot.bot.turn_stream"


def _adapter(*, drafts: bool = False) -> MagicMock:
    adapter = MagicMock()
    adapter.chunk_flush_at = 1900
    adapter.typing_refresh_interval = 8.0
    adapter.send_message = AsyncMock()
    adapter.send_link = AsyncMock()
    adapter.send_file = AsyncMock()
    adapter.start_typing = AsyncMock()
    adapter.stop_typing = AsyncMock()
    adapter.rename_thread = AsyncMock(return_value=True)
    adapter.supports_stream_drafts = drafts
    outcome = StreamDraftOutcome.SHOWN if drafts else StreamDraftOutcome.STOPPED
    adapter.send_stream_draft = AsyncMock(return_value=outcome)
    return adapter


def _ctx(channel_type: ChannelType = "dm") -> MessageContext:
    return MessageContext(
        platform="telegram",
        channel_type=channel_type,
        server_id=None,
        channel_id="42",
        message_id="msg-1",
        user_id="user-1",
        username="Bently",
        text="hello",
    )


def _api(chunks: list[str]) -> MagicMock:
    api = MagicMock()

    async def _stream(*args, **kwargs):
        for chunk in chunks:
            yield chunk

    api.stream_chat = _stream
    return api


def _patch_redis():
    return patch(
        f"{_MODULE}.get_redis_async",
        new=AsyncMock(
            return_value=AsyncMock(get=AsyncMock(return_value=None), set=AsyncMock())
        ),
    )


class TestDraftStreamer:
    @pytest.mark.asyncio
    async def test_noop_when_adapter_does_not_support_drafts(self):
        adapter = _adapter(drafts=False)
        draft = DraftStreamer(adapter, "42")
        await draft.update("hello")
        adapter.send_stream_draft.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_sends_preview_with_stable_nonzero_draft_id(self):
        adapter = _adapter(drafts=True)
        draft = DraftStreamer(adapter, "42")
        with patch(f"{_MODULE}.time.monotonic", side_effect=[100.0, 200.0]):
            await draft.update("hello")
            await draft.update("hello world")
        assert adapter.send_stream_draft.await_count == 2
        calls = adapter.send_stream_draft.await_args_list
        first_id = calls[0].args[1]
        assert first_id != 0
        assert all(
            c.args == ("42", first_id, t)
            for c, t in zip(calls, ["hello", "hello world"])
        )

    @pytest.mark.asyncio
    async def test_updates_are_throttled(self):
        adapter = _adapter(drafts=True)
        draft = DraftStreamer(adapter, "42")
        with patch(f"{_MODULE}.time.monotonic", side_effect=[100.0, 100.5]):
            await draft.update("hello")
            await draft.update("hello world")
        assert adapter.send_stream_draft.await_count == 1

    @pytest.mark.asyncio
    async def test_unchanged_or_empty_text_is_skipped(self):
        adapter = _adapter(drafts=True)
        draft = DraftStreamer(adapter, "42")
        with patch(f"{_MODULE}.time.monotonic", side_effect=[100.0, 200.0, 300.0]):
            await draft.update("   ")
            await draft.update("hello")
            await draft.update("hello ")  # same after strip
        assert adapter.send_stream_draft.await_count == 1

    @pytest.mark.asyncio
    async def test_stopped_disables_drafting_for_the_turn(self):
        adapter = _adapter(drafts=True)
        adapter.send_stream_draft = AsyncMock(return_value=StreamDraftOutcome.STOPPED)
        draft = DraftStreamer(adapter, "42")
        with patch(f"{_MODULE}.time.monotonic", side_effect=[100.0, 200.0]):
            await draft.update("hello")
            await draft.update("hello world")
        assert adapter.send_stream_draft.await_count == 1

    @pytest.mark.asyncio
    async def test_skipped_keeps_drafting_without_burning_the_throttle(self):
        # A SKIPPED update (preview momentarily too long) must not advance the
        # throttle or last_text — the very next chunk should retry immediately.
        adapter = _adapter(drafts=True)
        adapter.send_stream_draft = AsyncMock(
            side_effect=[StreamDraftOutcome.SKIPPED, StreamDraftOutcome.SHOWN]
        )
        draft = DraftStreamer(adapter, "42")
        # Second call is only 0.1s later — it would be throttled if the SKIP
        # had advanced _last_sent_at, but it must go through.
        with patch(f"{_MODULE}.time.monotonic", side_effect=[100.0, 100.1]):
            await draft.update("too-long-preview")
            await draft.update("shorter now")
        assert adapter.send_stream_draft.await_count == 2

    @pytest.mark.asyncio
    async def test_exception_disables_drafting_for_the_turn(self):
        adapter = _adapter(drafts=True)
        adapter.send_stream_draft = AsyncMock(side_effect=RuntimeError("boom"))
        draft = DraftStreamer(adapter, "42")
        with patch(f"{_MODULE}.time.monotonic", side_effect=[100.0, 200.0]):
            await draft.update("hello")
            await draft.update("hello world")
        assert adapter.send_stream_draft.await_count == 1


class TestStreamBatchDrafts:
    @pytest.mark.asyncio
    async def test_drafts_flow_during_stream_and_final_send_still_happens(self):
        adapter = _adapter(drafts=True)
        api = _api(["Hello", " world"])
        with _patch_redis():
            await TurnStreamer(api).stream_batch(
                [("Bently", "user-1", "hi")], _ctx(), adapter, "42"
            )
        assert adapter.send_stream_draft.await_count >= 1
        first = adapter.send_stream_draft.await_args_list[0]
        assert first.args[0] == "42"
        assert first.args[2] == "Hello"
        # The buffered final send is untouched by drafting.
        adapter.send_message.assert_awaited_once()
        assert adapter.send_message.await_args.args[1] == "Hello world"

    @pytest.mark.asyncio
    async def test_non_drafting_adapters_are_byte_identical(self):
        adapter = _adapter(drafts=False)
        api = _api(["Hello", " world"])
        with _patch_redis():
            await TurnStreamer(api).stream_batch(
                [("Bently", "user-1", "hi")], _ctx(), adapter, "42"
            )
        adapter.send_stream_draft.assert_not_awaited()
        adapter.send_message.assert_awaited_once()
        assert adapter.send_message.await_args.args[1] == "Hello world"
