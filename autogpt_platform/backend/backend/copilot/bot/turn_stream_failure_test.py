"""A failed bot turn tells the user what kind of failure it was."""

import re
from unittest.mock import MagicMock

import pytest

from .bot_backend import BotStreamError
from .prompt import PendingMessage
from .turn_stream import TurnStreamer
from .turn_stream_test import _adapter, _card_api, _choice_adapter, _ctx, _patch_redis


def _failing_api(exc: BaseException) -> MagicMock:
    api = MagicMock()

    async def _stream(*args, **kwargs):
        raise exc
        yield

    api.stream_chat = _stream
    return api


class TestFailureReply:
    """It used to be one fixed sentence for every cause, with nothing to act on
    or search for (Discord, 2026-09-15 and 2026-09-16)."""

    @pytest.mark.asyncio
    async def test_backend_error_names_the_cause_with_a_reference(self):
        adapter = _adapter()
        exc = BotStreamError(
            "backend_stream_error",
            "Unable to complete the tool call",
            code="circuit_breaker_empty_tool_calls",
        )
        with _patch_redis():
            await TurnStreamer(_failing_api(exc)).stream_batch(
                [PendingMessage(username="Bently", user_id="user-1", text="hi")],
                _ctx(),
                adapter,
                "42",
            )

        reply = adapter.send_message.await_args.args[1]
        assert "a step was too large to complete" in reply
        assert "Break your request into smaller parts" in reply
        assert re.search(r"\(ref [0-9a-f]{8}\)$", reply)

    @pytest.mark.asyncio
    async def test_raw_exception_text_never_reaches_chat(self):
        adapter = _adapter()
        exc = RuntimeError("KeyError at /app/backend/copilot/secret_path.py")
        with _patch_redis():
            await TurnStreamer(_failing_api(exc)).stream_batch(
                [PendingMessage(username="Bently", user_id="user-1", text="hi")],
                _ctx(),
                adapter,
                "42",
            )

        reply = adapter.send_message.await_args.args[1]
        assert "an internal error occurred" in reply
        assert "secret_path" not in reply
        assert "KeyError" not in reply

    @pytest.mark.asyncio
    async def test_a_turn_that_fails_after_a_held_call_keeps_its_card(self):
        adapter = _choice_adapter()
        api = _card_api(["I'll post it."])
        exc = BotStreamError("stream_timeout", "response timed out")

        async def _stream(*args, on_approval_needed=None, **kwargs):
            yield "I'll post it."
            await on_approval_needed("sess", "review-1")
            raise exc

        api.stream_chat = _stream
        with _patch_redis():
            await TurnStreamer(api).stream_batch(
                [PendingMessage(username="Bently", user_id="user-1", text="post")],
                _ctx(),
                adapter,
                "42",
            )

        adapter.send_choice_buttons.assert_awaited_once()
        sent = [c.args[1] for c in adapter.send_message.await_args_list]
        assert sent[0] == "I'll post it."
        assert "the request took too long" in sent[-1]
        assert re.search(r"\(ref [0-9a-f]{8}\)$", sent[-1])

    @pytest.mark.asyncio
    async def test_the_chat_reference_is_the_logged_reference(self, caplog):
        adapter = _adapter()
        exc = BotStreamError("stream_timeout", "response timed out")
        with _patch_redis(), caplog.at_level("ERROR"):
            await TurnStreamer(_failing_api(exc)).stream_batch(
                [PendingMessage(username="Bently", user_id="user-1", text="hi")],
                _ctx(),
                adapter,
                "42",
            )

        reply = adapter.send_message.await_args.args[1]
        assert "the request took too long" in reply
        match = re.search(r"\(ref ([0-9a-f]{8})\)", reply)
        assert match
        assert any(f"ref={match.group(1)}" in r.getMessage() for r in caplog.records)
