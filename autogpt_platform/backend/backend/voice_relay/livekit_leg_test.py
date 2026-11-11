"""A call's audio through its LiveKit room: the relay joins the room as the
vendor connects, carries the person's audio and Otto's through it while the
socket carries the rest, ends the call ``room_error`` when it cannot join the
room or loses it, and leaves the room once Otto's last words have played."""

import asyncio
import gc
from typing import Any

from prometheus_client import REGISTRY

from backend.voice_relay import locks
from backend.voice_relay.conftest import (
    FakeMedia,
    FakeSocket,
    counted_calls,
    livekit_edge,
    speech,
    until,
)
from backend.voice_relay.edge import LiveKitEdge
from backend.voice_relay.fake_vendor import FakeVendor, Reply
from backend.voice_relay.session import AudioLeg, Session
from backend.voice_relay.session_test import FAST, USER, Call, _session, open_call
from backend.voice_relay.vendor import AgentAudio, Interrupted

# A re-attach window far longer than any wait inside it, so a slow runner
# cannot close it early.
LONG_WINDOW = FAST.model_copy(update={"reattach_s": 10.0})


def _calls(reason: str) -> float:
    return (
        REGISTRY.get_sample_value("voice_interview_calls_total", {"reason": reason})
        or 0.0
    )


async def _room_call(redis, media: FakeMedia, *, audio_s: float = 0.1, **options):
    vendor = FakeVendor([Reply(text="Hi, I'm Otto, an AI.", audio_s=audio_s)])
    call = await open_call(redis, vendor, media=media, **options)
    await call.ws.wait_for("turn_complete")
    return call, vendor


async def _hang_up(call: Call) -> dict:
    call.ws.send_json({"type": "hangup"})
    return await call.ended()


def _person(call: Call) -> str:
    person = call.media.rooms[0].person
    assert person is not None
    return person


async def _never_opened(
    redis, media: FakeMedia
) -> tuple[FakeSocket, Session, FakeVendor]:
    """A call whose room the relay could not join, run to its end."""
    ws = FakeSocket()
    vendor = FakeVendor([Reply(text="Hi, I'm Otto, an AI.", audio_s=0.2)])
    session = _session(redis, vendor, ws, [], edge=livekit_edge(media))
    assert await locks.take_user_lock(redis, USER, session.call_id) is None
    await asyncio.wait_for(session.run(), timeout=5)
    return ws, session, vendor


async def test_a_call_joins_its_room_and_carries_otto_through_it(redis):
    media = FakeMedia()
    call, _ = await _room_call(redis, media, audio_s=0.4)

    assert call.ws.sent[0]["type"] == "ready"
    [room] = media.rooms
    assert room.claims["video"]["room"] == f"voice-{call.session.call_id}"
    await until(lambda: len(media.sources[0].audio) == round(0.4 * 24_000) * 2)
    assert (await _hang_up(call))["reason"] == "user_hangup"
    assert room.left


async def test_the_person_is_heard_under_the_identity_of_this_call(redis):
    media = FakeMedia()
    call, _ = await _room_call(redis, media)

    assert _person(call) == f"user:{USER}:{call.session.call_id}"
    # The same person in the room under another call's identity is no one
    # this call hears.
    room = media.rooms[0]
    other = room.publish(f"user:{USER}:{'f' * 32}")
    await asyncio.sleep(0.05)
    assert other.subscribe_calls == []
    await _hang_up(call)


async def test_the_persons_audio_takes_the_room_and_audio_on_the_socket_is_dropped(
    redis,
):
    media = FakeMedia()
    call, vendor = await _room_call(redis, media)
    said = speech(0.4)

    media.say(said)
    await until(lambda: vendor.sessions[0].audio_in_bytes == len(b"".join(said)))
    call.ws.send_binary(speech(0.2))
    await until(lambda: call.session.bad_frames == len(speech(0.2)))

    assert vendor.sessions[0].audio_in_bytes == len(b"".join(said))
    assert call.session.received == len(b"".join(said)) // 2
    await _hang_up(call)


async def test_talking_over_otto_drops_what_the_room_holds(redis):
    media = FakeMedia()
    call, vendor = await _room_call(redis, media)

    vendor.sessions[0].emit(Interrupted())

    await call.ws.wait_for("interrupted")
    assert media.sources[0].clears == 1
    await _hang_up(call)


async def test_a_relay_that_cannot_join_the_room_ends_the_call_before_it_goes_live(
    redis,
):
    media = FakeMedia(refuse=ConnectionError("the plan's allowance is used up"))
    before = _calls("room_error")

    ws, session, vendor = await _never_opened(redis, media)

    # The browser hears the call ended, and never that it was ready.
    assert ws.types() == ["ended"]
    assert ws.sent[0]["reason"] == "room_error"
    assert _calls("room_error") - before == 1
    # The model session it opened meanwhile is closed.
    assert [opened.closed for opened in vendor.sessions] == [True]
    record = await locks.read_call(redis, session.call_id)
    assert record is not None and record.end_reason == "room_error"
    assert (await locks.read_call_record(redis, session.call_id))["error"] == (
        "room: ConnectionError"
    )


async def test_a_room_lost_while_the_model_connects_ends_the_call_before_it_goes_live(
    redis, counted
):
    """The relay joins the room, and the room drops while the model session
    is still connecting: the call never goes live, and costs nothing."""
    media = FakeMedia()
    vendor = FakeVendor([Reply(text="Hi, I'm Otto, an AI.", audio_s=0.2)])
    # A slow model connect, which lands only once the room is gone.
    vendor.connect_gate = asyncio.Event()
    ws = FakeSocket()
    session = _session(redis, vendor, ws, [], edge=livekit_edge(media))
    assert await locks.take_user_lock(redis, USER, session.call_id) is None
    before = _calls("room_error")
    running = asyncio.create_task(session.run())

    await until(lambda: session.audio is not None)
    media.rooms[0].lost()
    vendor.connect_gate.set()
    await asyncio.wait_for(running, timeout=5)

    # The browser hears the call ended, and never that it was ready.
    assert ws.types() == ["ended"]
    assert ws.sent[0]["reason"] == "room_error"
    assert _calls("room_error") - before == 1
    assert await counted_calls(redis) == []
    # The model session that landed after the room was lost is closed.
    assert [opened.closed for opened in vendor.sessions] == [True]
    record = await locks.read_call(redis, session.call_id)
    assert record is not None and record.never_went_live


async def test_a_join_that_fails_before_the_model_refuses_leaves_no_error_unread(
    redis,
):
    """Whichever side fails first, the call collects the join's error: one
    left unread is logged at ERROR, as a fault in the relay, once the join is
    collected."""
    unread: list[str] = []
    loop = asyncio.get_running_loop()
    previous = loop.get_exception_handler()
    loop.set_exception_handler(lambda _, context: unread.append(context["message"]))
    try:
        assert await _refused_twice(redis) == "vendor_error"
        gc.collect()
    finally:
        loop.set_exception_handler(previous)

    assert unread == []


class _TellingEdge:
    """The relay's LiveKit edge, telling when a join has ended, however it
    ended."""

    def __init__(self, edge: LiveKitEdge) -> None:
        self.edge = edge
        self.joined = asyncio.Event()

    @property
    def rooms(self) -> int:
        return self.edge.rooms

    async def join(self, **options: Any) -> AudioLeg:
        try:
            return await self.edge.join(**options)
        finally:
            self.joined.set()


async def _refused_twice(redis) -> str | None:
    """A call whose room join fails, and whose model connect is then
    refused, run to its end. Returns how it ended, and holds nothing of the
    call after."""
    vendor = FakeVendor(refuse=ConnectionError("the vendor is down"))
    vendor.connect_gate = asyncio.Event()
    edge = _TellingEdge(
        livekit_edge(FakeMedia(refuse=ConnectionError("the room is full")))
    )
    session = _session(redis, vendor, FakeSocket(), [], edge=edge)
    assert await locks.take_user_lock(redis, USER, session.call_id) is None
    running = asyncio.create_task(session.run())

    # The model answers only once the join has failed.
    await asyncio.wait_for(edge.joined.wait(), timeout=5)
    vendor.connect_gate.set()
    await asyncio.wait_for(running, timeout=5)

    return session.end_reason


async def test_a_room_join_past_its_deadline_ends_the_call_and_the_room_is_left_once_it_lands(
    redis,
):
    media = FakeMedia()
    media.connect_gate = asyncio.Event()
    room_edge = livekit_edge(media, join_timeout_s=0.2)
    ws = FakeSocket()
    session = _session(
        redis, FakeVendor([Reply(text="Hi", audio_s=0.2)]), ws, [], edge=room_edge
    )
    assert await locks.take_user_lock(redis, USER, session.call_id) is None

    await asyncio.wait_for(session.run(), timeout=5)

    assert ws.types() == ["ended"]
    assert session.end_reason == "room_error"
    assert session.error == "room: TimeoutError"
    [room] = media.rooms
    assert room_edge.rooms == 1 and not room.connected
    media.connect_gate.set()
    await until(lambda: room.left)
    assert room_edge.rooms == 0


async def test_a_call_whose_vendor_refuses_while_its_room_is_joined_leaves_the_room_once_the_join_lands(
    redis,
):
    media = FakeMedia()
    media.connect_gate = asyncio.Event()
    room_edge = livekit_edge(media)
    session = _session(
        redis,
        FakeVendor(connect_delay_s=0.1, refuse=ConnectionError("the vendor is down")),
        FakeSocket(),
        [],
        edge=room_edge,
    )
    assert await locks.take_user_lock(redis, USER, session.call_id) is None

    await asyncio.wait_for(session.run(), timeout=5)

    assert session.end_reason == "vendor_error"
    # The call is over; the room's connect goes on under it.
    [room] = media.rooms
    assert room_edge.rooms == 1 and not room.connected
    media.connect_gate.set()
    await until(lambda: room.left)
    assert room_edge.rooms == 0


async def test_the_relay_losing_its_room_mid_call_ends_the_call_at_once(redis):
    media = FakeMedia()
    # Far longer than the test waits: the call must not wait out the window.
    call, _ = await _room_call(redis, media, timing=LONG_WINDOW)
    before = _calls("room_error")
    # Otto's last words would never finish playing in a room that is gone.
    media.sources[0].playing = asyncio.Event()

    media.rooms[0].lost()

    ended = await call.ended(timeout=LONG_WINDOW.reattach_s / 2)
    assert ended["reason"] == "room_error"
    assert _calls("room_error") - before == 1
    record = await locks.read_call(redis, call.session.call_id)
    assert record is not None and record.end_reason == "room_error"


async def test_the_persons_microphone_out_of_the_room_past_the_window_ends_the_call(
    redis,
):
    media = FakeMedia()
    call, vendor = await _room_call(redis, media)

    media.rooms[0].participant_leaves(_person(call))

    assert (await call.ended())["reason"] == "client_gone"
    # Their open turn was closed as their audio stopped.
    assert vendor.sessions[0].stream_ends >= 1


async def test_the_persons_microphone_back_within_the_window_keeps_the_call(redis):
    media = FakeMedia()
    call, _ = await _room_call(redis, media, timing=LONG_WINDOW)
    room = media.rooms[0]

    room.participant_leaves(_person(call))
    await until(lambda: call.session.away_since is not None)
    room.add_microphone(_person(call))
    await until(lambda: call.session.away_since is None)

    assert not call.session.ended.is_set()
    assert (await _hang_up(call))["reason"] == "user_hangup"


async def test_the_idle_clock_waits_while_the_persons_microphone_is_out_of_the_room(
    redis,
):
    media = FakeMedia()
    call, _ = await _room_call(redis, media, audio_s=0.04, timing=LONG_WINDOW)
    room = media.rooms[0]
    # Otto's few words have played out, so quiet time counts.
    await until(lambda: call.session.quiet_s > 0)

    room.participant_leaves(_person(call))
    await until(lambda: call.session.away_since is not None)
    quiet = call.session.quiet_s
    await asyncio.sleep(LONG_WINDOW.reattach_s / 20)
    assert call.session.quiet_s == quiet

    room.add_microphone(_person(call))
    await until(lambda: call.session.quiet_s > quiet + 0.1)
    await _hang_up(call)


async def test_the_idle_clock_waits_while_the_persons_audio_stalls_in_the_room(
    redis,
):
    media = FakeMedia()
    call, _ = await _room_call(redis, media, audio_s=0.04)
    room = media.rooms[0]
    microphone = room.microphone(_person(call))
    await until(lambda: call.session.quiet_s > 0)

    room.mute(_person(call), microphone, True)
    quiet = call.session.quiet_s
    await asyncio.sleep(0.3)
    assert call.session.quiet_s == quiet
    room.mute(_person(call), microphone, False)
    room.reconnecting()
    await asyncio.sleep(0.3)
    assert call.session.quiet_s == quiet

    room.reconnected()
    await until(lambda: call.session.quiet_s > quiet + 0.1)
    assert not call.session.ended.is_set()
    await _hang_up(call)


async def test_a_dropped_socket_leaves_the_audio_in_the_room(redis):
    media = FakeMedia()
    call, vendor = await _room_call(redis, media)
    source = media.sources[0]
    # Otto's opening has reached the room.
    await until(lambda: len(source.audio) == call.session.agent_out * 2)
    played = len(source.audio)

    call.ws.drop()
    await until(lambda: call.session.detached_at is not None)
    vendor.sessions[0].emit(AgentAudio(pcm=bytes(1920)))

    # Otto goes on through the room, and the person's turn stays open: their
    # microphone is still in it.
    await until(lambda: len(source.audio) == played + 1920)
    assert vendor.sessions[0].stream_ends == 0
    # The socket never came back.
    await asyncio.wait_for(call.task, timeout=5)
    assert call.session.end_reason == "client_gone"
    assert media.rooms[0].left


async def test_a_call_lets_ottos_last_words_play_out_before_leaving_its_room(redis):
    media = FakeMedia()
    room_edge = livekit_edge(media)
    vendor = FakeVendor([Reply(text="Hi, I'm Otto, an AI.", audio_s=0.1)])
    call = await open_call(redis, vendor, media=media, edge=room_edge)
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


async def test_a_room_that_fails_to_close_still_lets_the_call_end(redis, monkeypatch):
    media = FakeMedia()
    call, _ = await _room_call(redis, media)
    leg = call.session.audio
    assert leg is not None
    close = leg.close

    async def broken(*, play_out: bool) -> None:
        raise RuntimeError("the SDK went away")

    monkeypatch.setattr(leg, "close", broken)

    assert (await _hang_up(call))["reason"] == "user_hangup"
    assert call.session.closed.is_set()
    await close(play_out=False)


async def test_a_call_ended_from_another_device_leaves_its_room_at_once(redis):
    media = FakeMedia()
    call, _ = await _room_call(redis, media)
    # Otto's last words would never finish playing.
    media.sources[0].playing = asyncio.Event()

    await locks.request_end(redis, call.session.call_id)

    assert (await call.ended())["reason"] == "superseded"
    assert media.rooms[0].left
