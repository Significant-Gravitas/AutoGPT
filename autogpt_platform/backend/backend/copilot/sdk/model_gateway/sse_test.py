import json
from collections.abc import AsyncIterator

from backend.copilot.sdk.model_gateway.sse import SseFrame, encode_event, read_frames


async def _lines(*lines: str) -> AsyncIterator[str]:
    for line in lines:
        yield line


async def _frames(*lines: str) -> list[SseFrame]:
    return [frame async for frame in read_frames(_lines(*lines))]


async def test_frames_are_grouped_by_blank_lines():
    frames = await _frames(
        "event: message_start",
        'data: {"type":"message_start"}',
        "",
        'data: {"type":"ping"}',
        "",
    )
    assert frames == [
        SseFrame(event="message_start", data='{"type":"message_start"}'),
        SseFrame(event=None, data='{"type":"ping"}'),
    ]


async def test_comments_carriage_returns_and_unknown_fields_are_ignored():
    frames = await _frames(": keep-alive\r", "id: 7\r", "data: x\r", "\r")
    assert frames == [SseFrame(data="x")]


async def test_multi_line_data_is_joined_and_a_final_frame_without_a_blank_line_counts():
    frames = await _frames("data: one", "data:two")
    assert frames == [SseFrame(data="one\ntwo")]


def test_a_relayed_frame_keeps_its_fields():
    frame = SseFrame(event="content_block_delta", data='{"a":1}\n{"b":2}')
    assert frame.encoded() == (
        b'event: content_block_delta\ndata: {"a":1}\ndata: {"b":2}\n\n'
    )
    assert SseFrame(data="[DONE]").payload() is None


def test_events_are_written_with_their_type_as_the_event_name():
    encoded = encode_event({"type": "message_stop"})
    assert encoded == b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    assert json.loads(encoded.decode().split("data: ")[1]) == {"type": "message_stop"}
