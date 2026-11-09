"""A call on the LiveKit leg: the relay joins the call's room as the vendor
connects, carries the person's audio and Otto's through it while the socket
carries the rest, takes the socket instead when it cannot join, and leaves
the room once Otto's last words have played."""

import asyncio

from prometheus_client import REGISTRY

from backend.voice_relay import locks
from backend.voice_relay.conftest import FakeMedia, livekit_edge, silence, speech, until
from backend.voice_relay.fake_vendor import FakeVendor, Reply
from backend.voice_relay.session_test import FAST, USER, Call, open_call
from backend.voice_relay.vendor import AgentAudio, Interrupted

PERSON = f"user:{USER}"


def _fallbacks(why: str) -> float:
    return (
        REGISTRY.get_sample_value(
            "voice_interview_fallbacks_total", {"from": "livekit", "why": why}
        )
        or 0.0
    )


async def _room_call(redis, media: FakeMedia, *, audio_s: float = 0.1, **options):
    vendor = FakeVendor([Reply(text="Hi, I'm Otto, an AI.", audio_s=audio_s)])
    call = await open_call(
        redis, vendor, leg="livekit", edge=livekit_edge(media), **options
    )
    await call.ws.wait_for("turn_complete")
    return call, vendor


async def _hang_up(call: Call) -> dict:
    call.ws.send_json({"type": "hangup"})
    return await call.ended()


async def test_a_livekit_call_joins_its_room_and_carries_otto_through_it(redis):
    media = FakeMedia()
    call, _ = await _room_call(redis, media, audio_s=0.4)

    assert call.ws.sent[0]["type"] == "ready"
    assert call.ws.sent[0]["leg"] == "livekit"
    [room] = media.rooms
    assert room.claims["video"]["room"] == f"voice-{call.session.call_id}"
    await until(lambda: len(media.sources[0].audio) == round(0.4 * 24_000) * 2)
    # Nothing of Otto's travels on the socket.
    assert call.ws.audio == []
    record = await locks.read_call(redis, call.session.call_id)
    assert record is not None and record.leg == "livekit"
    assert (await _hang_up(call))["reason"] == "user_hangup"
    assert room.left


async def test_the_persons_audio_takes_the_room_and_audio_on_the_socket_is_dropped(
    redis,
):
    media = FakeMedia()
    call, vendor = await _room_call(redis, media)
    said = speech(0.4)

    media.say(said)
    await until(lambda: vendor.sessions[0].audio_in_bytes == len(b"".join(said)))
    call.ws.send_audio(silence(0.2))
    await until(lambda: call.session.bad_frames == len(silence(0.2)))

    assert vendor.sessions[0].audio_in_bytes == len(b"".join(said))
    assert call.session.received == len(b"".join(said)) // 2
    await _hang_up(call)


async def test_talking_over_otto_on_the_livekit_leg_drops_what_the_room_holds(redis):
    media = FakeMedia()
    call, vendor = await _room_call(redis, media)

    vendor.sessions[0].emit(Interrupted())

    await call.ws.wait_for("interrupted")
    assert media.sources[0].clears == 1
    await _hang_up(call)


async def test_a_relay_that_cannot_join_the_room_moves_the_calls_audio_to_the_socket(
    redis,
):
    media = FakeMedia(refuse=ConnectionError("the plan's allowance is used up"))
    room_edge = livekit_edge(media)
    before = _fallbacks("relay_join")
    vendor = FakeVendor([Reply(text="Hi, I'm Otto, an AI.", audio_s=0.2)])

    call = await open_call(redis, vendor, leg="livekit", edge=room_edge)

    assert call.ws.sent[0] == {"type": "leg", "leg": "websocket"}
    assert call.ws.sent[1]["type"] == "ready"
    assert call.ws.sent[1]["leg"] == "websocket"
    await call.ws.wait_for("turn_complete")
    assert len(b"".join(call.ws.audio)) == round(0.2 * 24_000) * 2
    call.ws.send_audio(speech(0.2))
    await until(lambda: vendor.sessions[0].audio_in_bytes > 0)
    assert _fallbacks("relay_join") - before == 1
    assert room_edge.rooms == 0
    record = await locks.read_call(redis, call.session.call_id)
    assert record is not None and record.leg == "websocket"
    await _hang_up(call)


async def test_a_relay_with_no_livekit_project_carries_a_livekit_call_on_the_socket(
    redis,
):
    before = _fallbacks("relay_unconfigured")

    call = await open_call(
        redis, FakeVendor([Reply(text="Hi")]), leg="livekit", edge=None
    )

    assert call.ws.sent[0] == {"type": "leg", "leg": "websocket"}
    assert call.ws.sent[1]["leg"] == "websocket"
    assert _fallbacks("relay_unconfigured") - before == 1
    await _hang_up(call)


async def test_the_persons_microphone_out_of_the_room_past_the_window_ends_the_call(
    redis,
):
    media = FakeMedia()
    call, vendor = await _room_call(redis, media)

    media.rooms[0].participant_leaves(PERSON)

    assert (await call.ended())["reason"] == "client_gone"
    # Their open turn was closed as their audio stopped.
    assert vendor.sessions[0].stream_ends >= 1


async def test_the_persons_microphone_back_within_the_window_keeps_the_call(redis):
    media = FakeMedia()
    call, _ = await _room_call(redis, media)
    room = media.rooms[0]

    room.participant_leaves(PERSON)
    await asyncio.sleep(FAST.reattach_s / 2)
    room.add_microphone(PERSON)
    await asyncio.sleep(FAST.reattach_s)

    assert not call.session.ended.is_set()
    assert (await _hang_up(call))["reason"] == "user_hangup"


async def test_the_idle_clock_waits_while_the_persons_microphone_is_out_of_the_room(
    redis,
):
    media = FakeMedia()
    call, _ = await _room_call(redis, media, audio_s=0.04)
    room = media.rooms[0]
    # Otto's few words have played out, so quiet time counts.
    await until(lambda: call.session.quiet_s > 0)

    room.participant_leaves(PERSON)
    quiet = call.session.quiet_s
    await asyncio.sleep(FAST.reattach_s / 2)
    assert call.session.quiet_s == quiet

    room.add_microphone(PERSON)
    await until(lambda: call.session.quiet_s > quiet + 0.1)
    await _hang_up(call)


async def test_a_dropped_socket_on_the_livekit_leg_leaves_the_audio_in_the_room(
    redis,
):
    media = FakeMedia()
    call, vendor = await _room_call(redis, media)
    source = media.sources[0]
    played = len(source.audio)

    call.ws.drop()
    await until(lambda: call.session.detached_at is not None)
    vendor.sessions[0].emit(AgentAudio(pcm=bytes(1920)))

    # Otto goes on through the room, and the person's turn stays open: their
    # microphone is still in it.
    await until(lambda: len(source.audio) == played + 1920)
    assert all(data is None for _, data in call.session.outbox)
    assert vendor.sessions[0].stream_ends == 0
    # The socket never came back.
    await asyncio.wait_for(call.task, timeout=5)
    assert call.session.end_reason == "client_gone"
    assert media.rooms[0].left


async def test_a_livekit_call_lets_ottos_last_words_play_out_before_leaving_its_room(
    redis,
):
    media = FakeMedia()
    room_edge = livekit_edge(media)
    vendor = FakeVendor([Reply(text="Hi, I'm Otto, an AI.", audio_s=0.1)])
    call = await open_call(redis, vendor, leg="livekit", edge=room_edge)
    await call.ws.wait_for("turn_complete")
    source = media.sources[0]
    source.playing = asyncio.Event()

    call.ws.send_json({"type": "hangup"})
    await call.ws.wait_for("ended")
    await asyncio.sleep(0.05)
    assert not media.rooms[0].left and room_edge.rooms == 1

    source.playing.set()
    await asyncio.wait_for(call.task, timeout=5)
    assert media.rooms[0].left and room_edge.rooms == 0


async def test_a_call_ended_from_another_device_leaves_its_room_at_once(redis):
    media = FakeMedia()
    call, _ = await _room_call(redis, media)
    # Otto's last words would never finish playing.
    media.sources[0].playing = asyncio.Event()

    await locks.request_end(redis, call.session.call_id)

    assert (await call.ended())["reason"] == "superseded"
    assert media.rooms[0].left
