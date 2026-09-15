"""Tests for the per-turn streaming helpers, focused on live draft previews."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from .adapters.base import ChannelType, MessageContext, StreamDraftOutcome
from .turn_stream import DraftStreamer, TurnStreamer, _send_clarification

_MODULE = "backend.copilot.bot.turn_stream"


def _adapter(*, drafts: bool = False) -> MagicMock:
    adapter = MagicMock()
    adapter.chunk_flush_at = 1900
    adapter.typing_refresh_interval = 8.0
    adapter.send_message = AsyncMock()
    adapter.send_link = AsyncMock()
    adapter.send_file = AsyncMock()
    adapter.supports_choice_buttons = False
    adapter.send_choice_buttons = AsyncMock(return_value=False)
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


# -- Native choice buttons: the branch the feature is named after --


def _choice_adapter(
    *, max_options: int = 10, max_len: int = 4096, max_label: int = 64
) -> MagicMock:
    adapter = _adapter()
    adapter.platform_name = "telegram"
    adapter.supports_choice_buttons = True
    adapter.max_choice_options = max_options
    adapter.max_choice_label_length = max_label
    adapter.max_message_length = max_len
    adapter.localize_markup = lambda text: text
    adapter.send_choice_buttons = AsyncMock(return_value=True)
    return adapter


def _patch_choices():
    return patch(
        f"{_MODULE}.choices",
        new=MagicMock(
            store_choice=AsyncMock(return_value="tok"),
            clear_choice=AsyncMock(),
        ),
    )


async def _clarify(adapter, questions):
    with _patch_choices() as choices_mock:
        await _send_clarification(adapter, "42", _ctx(), {"questions": questions})
    return choices_mock


class TestNativeChoices:
    @pytest.mark.asyncio
    async def test_sends_native_buttons_and_no_text(self):
        adapter = _choice_adapter()
        await _clarify(adapter, [{"question": "Region?", "options": ["EU", "US"]}])

        adapter.send_choice_buttons.assert_awaited_once()
        assert adapter.send_choice_buttons.await_args.args[1] == "❓ Region?"
        adapter.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_adapter_returning_false_falls_back_to_text(self):
        adapter = _choice_adapter()
        adapter.send_choice_buttons = AsyncMock(return_value=False)

        choices_mock = await _clarify(
            adapter, [{"question": "Region?", "options": ["EU", "US"]}]
        )

        adapter.send_message.assert_awaited()
        choices_mock.clear_choice.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_adapter_raising_falls_back_to_text(self):
        # Telegram and Teams have no non-raising failure path, so a 429 used
        # to escape into the caller's generic handler: the user got "AutoGPT
        # ran into an error" and the question was delivered in no form.
        adapter = _choice_adapter()
        adapter.send_choice_buttons = AsyncMock(side_effect=RuntimeError("429"))

        choices_mock = await _clarify(
            adapter, [{"question": "Region?", "options": ["EU", "US"]}]
        )

        adapter.send_message.assert_awaited()
        assert "Region?" in adapter.send_message.await_args.args[1]
        choices_mock.clear_choice.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_too_many_options_for_this_platform_uses_text(self):
        adapter = _choice_adapter(max_options=6)

        await _clarify(
            adapter,
            [{"question": "Pick", "options": [f"o{i}" for i in range(7)]}],
        )

        adapter.send_choice_buttons.assert_not_awaited()
        adapter.send_message.assert_awaited()

    @pytest.mark.asyncio
    async def test_question_over_the_message_cap_uses_text(self):
        # Nothing chunks the native question text, and the send raises past
        # the cap — the numbered text fits because it chunks.
        adapter = _choice_adapter(max_len=50)

        await _clarify(adapter, [{"question": "x" * 200, "options": ["a", "b"]}])

        adapter.send_choice_buttons.assert_not_awaited()
        adapter.send_message.assert_awaited()

    @pytest.mark.asyncio
    async def test_mixed_payload_goes_all_text_to_keep_order(self):
        # Natives post per-question while text is batched afterwards, so a
        # mixed payload would arrive Q1, Q3, Q2 and the user would answer
        # against the wrong numbering.
        adapter = _choice_adapter()

        await _clarify(
            adapter,
            [
                {"question": "Q1", "options": ["a", "b"]},
                {"question": "Q2 free text"},
                {"question": "Q3", "options": ["c", "d"]},
            ],
        )

        adapter.send_choice_buttons.assert_not_awaited()
        sent = "".join(c.args[1] for c in adapter.send_message.await_args_list)
        assert sent.index("Q1") < sent.index("Q2") < sent.index("Q3")

    @pytest.mark.asyncio
    async def test_unsupported_adapter_still_uses_text(self):
        adapter = _choice_adapter()
        adapter.supports_choice_buttons = False

        await _clarify(adapter, [{"question": "Region?", "options": ["EU", "US"]}])

        adapter.send_choice_buttons.assert_not_awaited()
        adapter.send_message.assert_awaited()

    @pytest.mark.asyncio
    async def test_option_longer_than_the_label_cap_uses_text(self):
        # Native widgets clip labels, so two options sharing a prefix render
        # identically while each still dispatches its own full text — the
        # user can't tell which button they're pressing.
        adapter = _choice_adapter(max_label=10)

        await _clarify(
            adapter,
            [{"question": "Pick", "options": ["short", "x" * 40]}],
        )

        adapter.send_choice_buttons.assert_not_awaited()
        adapter.send_message.assert_awaited()

    @pytest.mark.asyncio
    async def test_question_measured_after_localization(self):
        # Adapters localize before sending, and HTML/mrkdwn escaping can push
        # a question that fitted over the cap — the send then raises.
        adapter = _choice_adapter(max_len=50)
        adapter.localize_markup = lambda text: text * 10

        await _clarify(adapter, [{"question": "short one", "options": ["a", "b"]}])

        adapter.send_choice_buttons.assert_not_awaited()
        adapter.send_message.assert_awaited()

    @pytest.mark.asyncio
    async def test_non_list_options_render_as_a_free_text_question(self):
        # Model-shaped payload: a string here used to be iterated character
        # by character into one button per letter.
        adapter = _choice_adapter()

        await _clarify(adapter, [{"question": "Region?", "options": "EU"}])

        adapter.send_choice_buttons.assert_not_awaited()
        adapter.send_message.assert_awaited()

    @pytest.mark.asyncio
    async def test_store_choice_failing_still_falls_back_to_text(self):
        # A Redis failure minting the token used to escape before the
        # adapter was ever called, reaching the generic stream error.
        adapter = _choice_adapter()
        with patch(
            f"{_MODULE}.choices",
            new=MagicMock(
                store_choice=AsyncMock(side_effect=RuntimeError("redis down")),
                clear_choice=AsyncMock(),
            ),
        ):
            await _send_clarification(
                adapter,
                "42",
                _ctx(),
                {"questions": [{"question": "Q", "options": ["a"]}]},
            )

        adapter.send_message.assert_awaited()

    @pytest.mark.asyncio
    async def test_clear_choice_failing_still_falls_back_to_text(self):
        adapter = _choice_adapter()
        adapter.send_choice_buttons = AsyncMock(return_value=False)
        with patch(
            f"{_MODULE}.choices",
            new=MagicMock(
                store_choice=AsyncMock(return_value="tok"),
                clear_choice=AsyncMock(side_effect=RuntimeError("redis down")),
            ),
        ):
            await _send_clarification(
                adapter,
                "42",
                _ctx(),
                {"questions": [{"question": "Q", "options": ["a"]}]},
            )

        adapter.send_message.assert_awaited()

    @pytest.mark.asyncio
    async def test_partial_native_delivery_clears_every_token(self):
        # Q1 sends, Q2 fails, so both are re-rendered as text. Q1's button is
        # still on screen above that text and must not answer the turn again.
        adapter = _choice_adapter()
        adapter.send_choice_buttons = AsyncMock(side_effect=[True, False])
        tokens = iter(["tok-1", "tok-2"])
        choices_mock = MagicMock(
            store_choice=AsyncMock(side_effect=lambda *a, **k: next(tokens)),
            clear_choice=AsyncMock(),
        )
        with patch(f"{_MODULE}.choices", new=choices_mock):
            await _send_clarification(
                adapter,
                "42",
                _ctx(),
                {
                    "questions": [
                        {"question": "Q1", "options": ["a", "b"]},
                        {"question": "Q2", "options": ["c", "d"]},
                    ]
                },
            )

        cleared = {c.args[1] for c in choices_mock.clear_choice.await_args_list}
        assert cleared == {"tok-1", "tok-2"}
        adapter.send_message.assert_awaited()
