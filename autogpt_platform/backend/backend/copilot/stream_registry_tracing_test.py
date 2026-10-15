"""Per-chunk and per-poll Redis calls stay off Sentry transactions, so a long
turn's transaction and the ``/stream`` request relaying it stay bounded."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
import sentry_sdk

from backend.copilot import stream_registry
from backend.copilot.response_model import StreamFinish, StreamTextDelta
from backend.util import tracing


@pytest.fixture(autouse=True)
def _clear_listener_sessions():
    stream_registry._listener_sessions.clear()
    yield
    stream_registry._listener_sessions.clear()


class _SpanPerCommandRedis:
    """Redis with Sentry's integration on: every command is a ``db.redis`` span
    named after it. Serves a running turn whose live tail is ``live``."""

    def __init__(self, live: list[tuple[str, dict[str, str]]] | None = None):
        self._live = live or []

    async def _command(self, name: str, result):
        with sentry_sdk.start_span(op="db.redis", name=name):
            return result

    async def hgetall(self, _key):
        meta = {
            "user_id": "u1",
            "session_id": "sess-1",
            "turn_id": "turn-1",
            "status": "running",
        }
        return await self._command("HGETALL", meta)

    async def xread(self, _streams, count=None, block=None):
        if block is None:
            return await self._command("XREAD", [])
        live, self._live = self._live, []
        return await self._command("XREAD BLOCK", [("stream", live)] if live else [])

    async def hmget(self, _key, _fields):
        return await self._command("HMGET", ["completed", "turn-1"])

    async def xadd(self, _key, _fields):
        return await self._command("XADD", "1-0")

    async def expire(self, _key, _ttl):
        return await self._command("EXPIRE", True)


def _stored(chunk) -> dict[str, str]:
    return {"data": chunk.model_dump_json()}


@pytest.mark.asyncio
async def test_the_relay_keeps_its_polls_off_the_request_transaction(
    sentry_transactions,
):
    redis = _SpanPerCommandRedis(
        live=[
            ("1-0", _stored(StreamTextDelta(id="t", delta="hi"))),
            ("1-1", _stored(StreamFinish())),
        ]
    )
    with patch.object(
        stream_registry, "get_redis_async", new=AsyncMock(return_value=redis)
    ):
        with tracing.start_transaction("http.server", "/stream"):
            queue = await stream_registry.subscribe_to_session("sess-1", "u1")
            assert queue is not None
            delivered = [await asyncio.wait_for(queue.get(), 5) for _ in range(2)]

    assert [type(chunk).__name__ for _, chunk in delivered] == [
        "StreamTextDelta",
        "StreamFinish",
    ]
    [request] = sentry_transactions
    assert [span["description"] for span in request["spans"]] == ["HGETALL", "XREAD"]


@pytest.mark.asyncio
async def test_publishing_a_chunk_adds_no_spans_to_the_turn(sentry_transactions):
    with patch.object(
        stream_registry,
        "get_redis_async",
        new=AsyncMock(return_value=_SpanPerCommandRedis()),
    ):
        with tracing.start_transaction("copilot.turn", "copilot.turn"):
            for _ in range(3):
                message_id = await stream_registry.publish_chunk(
                    "turn-1", StreamTextDelta(id="t", delta="x"), session_id="s"
                )

    assert message_id == "1-0"
    [turn] = sentry_transactions
    assert turn["spans"] == []
