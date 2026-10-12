"""Entry ids, checkpoints and the cursor read, against a real Redis stream."""

import asyncio
import uuid
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot import stream_registry
from backend.copilot.response_model import (
    StreamBaseResponse,
    StreamCheckpoint,
    StreamFinish,
    StreamHeartbeat,
    StreamStart,
    StreamTextDelta,
    StreamTextEnd,
    StreamTextStart,
    StreamUsage,
)
from backend.data import redis_client
from backend.util.testing import is_tcp_port_reachable

pytestmark = pytest.mark.skipif(
    not is_tcp_port_reachable(redis_client.HOST, redis_client.PORT),
    reason="no local Redis reachable; the stream registry needs one to run",
)


@pytest.fixture
async def turn():
    """A running turn of its own session, with every key removed after."""
    session_id, turn_id = f"resume-{uuid.uuid4().hex}", str(uuid.uuid4())
    await stream_registry.create_session(session_id, None, "", "", turn_id=turn_id)
    turn_ids = [turn_id]
    yield session_id, turn_id, turn_ids
    redis = await redis_client.get_redis_async()
    await redis.delete(stream_registry.get_session_meta_key(session_id))
    for key_turn in turn_ids:
        await redis.delete(stream_registry._get_turn_stream_key(key_turn))
        await redis.delete(stream_registry._get_turn_meta_key(key_turn))


async def test_every_entry_frame_carries_its_turn_and_entry_id(turn):
    session_id, turn_id, _ = turn
    ids = await _publish(
        session_id,
        turn_id,
        [
            StreamStart(messageId="m1"),
            StreamUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        ],
    )

    queue = await stream_registry.subscribe_to_session(session_id, None, "0-0")
    assert queue is not None
    try:
        start, usage = queue.get_nowait(), queue.get_nowait()
    finally:
        await stream_registry.unsubscribe_from_session(session_id, queue)

    assert stream_registry.sse_frame(start).startswith(
        f"id: {turn_id}:{ids[0]}\ndata: "
    )
    # A comment frame's id would be dropped by the client's parser.
    assert stream_registry.sse_frame(usage).startswith(": usage ")
    assert stream_registry.sse_frame((None, StreamHeartbeat())) == ": heartbeat\n\n"


async def test_a_checkpoint_is_exposed_on_the_active_stream(turn):
    session_id, turn_id, _ = turn
    [entry_id] = await _publish(
        session_id, turn_id, [StreamCheckpoint(rows=3, sequence=7, digest="d")]
    )

    active, _ = await stream_registry.get_active_session(session_id)

    assert active is not None
    assert active.checkpoint == stream_registry.TurnCheckpoint(
        entry_id=entry_id, rows=3, sequence=7
    )


async def test_a_cursor_read_serves_exactly_the_entries_after_the_cursor(turn):
    session_id, turn_id, _ = turn
    ids = await _publish(session_id, turn_id, _text_turn())
    await _complete(session_id, turn_id)

    for position, cursor in enumerate(["0-0", *ids]):
        queue = await stream_registry.subscribe_to_turn(
            session_id, None, turn_id, cursor
        )
        *tail, (finish_id, finish) = await _drain(session_id, queue)

        assert [frame_id for frame_id, _ in tail] == [
            f"{turn_id}:{entry_id}" for entry_id in ids[position:]
        ]
        assert isinstance(finish, StreamFinish)
        assert finish_id is not None and finish_id.startswith(f"{turn_id}:")


async def test_a_finished_turns_tail_ends_with_its_stored_finish(turn):
    session_id, turn_id, _ = turn
    await _publish(session_id, turn_id, _text_turn())
    await _complete(session_id, turn_id)

    queue = await stream_registry.subscribe_to_turn(session_id, None, turn_id, "0-0")
    served = await _drain(session_id, queue)

    frame_id, last = served[-1]
    assert isinstance(last, StreamFinish) and frame_id is not None
    assert all(frame_id is not None for frame_id, _ in served)


async def test_completion_trims_to_the_last_checkpoint(turn):
    session_id, turn_id, _ = turn
    ids = await _publish(
        session_id,
        turn_id,
        [
            *_text_turn(),
            StreamCheckpoint(rows=1, sequence=4, digest="d"),
            StreamTextStart(id="t2"),
            StreamTextEnd(id="t2"),
        ],
    )
    checkpoint_id = ids[len(_text_turn())]
    await _complete(session_id, turn_id)

    with pytest.raises(stream_registry.TurnStreamTrimmed) as trimmed:
        await stream_registry.subscribe_to_turn(session_id, None, turn_id, ids[1])
    queue = await stream_registry.subscribe_to_turn(
        session_id, None, turn_id, checkpoint_id
    )
    served = await _drain(session_id, queue)

    assert trimmed.value.checkpoint == stream_registry.TurnCheckpoint(
        entry_id=checkpoint_id, rows=1, sequence=4
    )
    assert [type(chunk) for _, chunk in served] == [
        StreamTextStart,
        StreamTextEnd,
        StreamFinish,
    ]


async def test_an_expired_stream_is_gone(turn):
    session_id, turn_id, _ = turn
    await _publish(session_id, turn_id, _text_turn())
    await _complete(session_id, turn_id)
    redis = await redis_client.get_redis_async()
    await redis.delete(stream_registry._get_turn_stream_key(turn_id))

    with pytest.raises(stream_registry.TurnStreamGone):
        await stream_registry.subscribe_to_turn(session_id, None, turn_id, "0-0")


async def test_a_finished_turn_is_served_after_the_next_one_starts(turn):
    session_id, turn_id, turn_ids = turn
    ids = await _publish(session_id, turn_id, _text_turn())
    await _complete(session_id, turn_id)
    turn_ids.append(next_turn := str(uuid.uuid4()))
    await stream_registry.create_session(session_id, None, "", "", turn_id=next_turn)

    queue = await stream_registry.subscribe_to_turn(session_id, None, turn_id, ids[0])
    served = await _drain(session_id, queue)

    assert isinstance(served[-1][1], StreamFinish)
    assert len(served) == len(ids)  # everything after the first, plus the finish


async def test_another_sessions_turn_is_refused(turn):
    session_id, turn_id, _ = turn
    await _publish(session_id, turn_id, _text_turn())
    await _complete(session_id, turn_id)
    other = f"resume-{uuid.uuid4().hex}"
    await stream_registry.create_session(other, None, "", "", turn_id="elsewhere")
    try:
        with pytest.raises(stream_registry.TurnStreamGone):
            await stream_registry.subscribe_to_turn(other, None, turn_id, "0-0")
    finally:
        redis = await redis_client.get_redis_async()
        await redis.delete(stream_registry.get_session_meta_key(other))
        await redis.delete(stream_registry._get_turn_meta_key("elsewhere"))


async def test_a_finished_turns_replay_is_not_capped(turn):
    """W9: a finished turn's replay stopped at 200 entries and made up a finish."""
    session_id, turn_id, _ = turn
    deltas = [StreamTextDelta(id="t1", delta=f"{i} ") for i in range(450)]
    await _publish(
        session_id,
        turn_id,
        [
            StreamStart(messageId="m1"),
            StreamTextStart(id="t1"),
            *deltas,
            StreamTextEnd(id="t1"),
        ],
    )
    await _complete(session_id, turn_id)

    queue = await stream_registry.subscribe_to_session(session_id, None, "0-0")
    assert queue is not None
    served = await _drain(session_id, queue)

    assert len(served) == len(deltas) + 4
    assert all(frame_id is not None for frame_id, _ in served)


def _text_turn() -> list[StreamBaseResponse]:
    return [
        StreamStart(messageId="m1"),
        StreamTextStart(id="t1"),
        StreamTextDelta(id="t1", delta="Hello "),
        StreamTextDelta(id="t1", delta="there."),
        StreamTextEnd(id="t1"),
    ]


async def _publish(
    session_id: str, turn_id: str, chunks: list[StreamBaseResponse]
) -> list[str]:
    return [
        await stream_registry.publish_chunk(turn_id, chunk, session_id=session_id)
        for chunk in chunks
    ]


async def _complete(session_id: str, turn_id: str) -> None:
    with patch.object(
        stream_registry.chat_db(), "set_turn_duration", new=AsyncMock(), create=True
    ):
        await stream_registry.mark_session_completed(session_id, turn_id=turn_id)


async def _drain(session_id, queue) -> list[stream_registry.StreamEntry]:
    """Everything the queue serves up to the turn's finish or the end marker."""
    served = []
    try:
        while True:
            frame_id, chunk = await asyncio.wait_for(queue.get(), timeout=10)
            if isinstance(chunk, StreamHeartbeat):
                continue
            served.append((frame_id, chunk))
            if isinstance(chunk, StreamFinish):
                return served
    finally:
        await stream_registry.unsubscribe_from_session(session_id, queue)
