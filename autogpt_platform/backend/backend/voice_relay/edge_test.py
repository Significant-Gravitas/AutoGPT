"""The LiveKit leg's room, against fakes of the SDK's room, source and
stream: what the relay joins and subscribes to, what it hears and plays, and
how it leaves, including a room whose connect lands after the join gave up."""

import asyncio

import pytest
from livekit import rtc
from prometheus_client import REGISTRY

from backend.voice_relay import edge
from backend.voice_relay.config import (
    IN_RATE,
    OUT_RATE,
    ROOM_CONNECT_TIMEOUT_S,
    ROOM_JOIN_TIMEOUT_S,
    ROOM_QUEUE_S,
)
from backend.voice_relay.conftest import LIVEKIT, FakeMedia, livekit_edge, until
from backend.voice_relay.edge import RoomLeg, RoomMedia

CALL_ID = "e" * 32
PERSON = "user:user-1"
# 10 ms of the person's audio, and the call's 40 ms frame.
TEN_MS = bytes(range(256)) + bytes(64)
FRAME_BYTES = 1280


class Listener:
    """What the room tells the call."""

    def __init__(self) -> None:
        self.heard: list[bytes] = []
        self.presence: list[bool] = []
        self.stalls: list[bool] = []

    async def hear(self, pcm: bytes) -> None:
        self.heard.append(pcm)

    def join(self, room_edge: edge.LiveKitEdge, *, user_id: str = "user-1"):
        return room_edge.join(
            call_id=CALL_ID,
            user_id=user_id,
            ttl_s=360,
            hear=self.hear,
            presence=self.presence.append,
            stall=self.stalls.append,
        )


class Joined:
    def __init__(
        self,
        media: FakeMedia,
        leg: RoomLeg,
        room_edge: edge.LiveKitEdge,
        listener: Listener,
    ) -> None:
        self.media = media
        self.leg = leg
        self.edge = room_edge
        self.heard = listener.heard
        self.presence = listener.presence
        self.stalls = listener.stalls

    @property
    def room(self):
        return self.media.rooms[-1]

    @property
    def source(self):
        return self.media.sources[-1]


async def join(media: FakeMedia, *, hearing: bool = True, **options) -> Joined:
    """Join a room; with *hearing*, until the person's microphone is heard."""
    room_edge = livekit_edge(media, **options)
    listener = Listener()
    joined = Joined(media, await listener.join(room_edge), room_edge, listener)
    if hearing:
        await until(lambda: joined.presence == [True])
    return joined


def _counter(name: str) -> float:
    return REGISTRY.get_sample_value(name) or 0.0


async def test_the_relay_joins_the_calls_room_as_otto_and_publishes_its_microphone():
    joined = await join(FakeMedia())

    room = joined.room
    assert room.url == LIVEKIT.url
    assert room.claims["sub"] == "otto"
    assert room.claims["video"]["room"] == f"voice-{CALL_ID}"
    [(track, options)] = room.local_participant.published
    assert track.source is joined.source
    assert options.source == rtc.TrackSource.SOURCE_MICROPHONE
    assert joined.edge.rooms == 1
    await joined.leg.close(play_out=False)


async def test_the_relay_subscribes_to_nothing_on_its_own_and_tries_each_connect_inside_its_deadline():
    joined = await join(FakeMedia())

    options = joined.room.options
    assert isinstance(options, rtc.RoomOptions)
    assert options.auto_subscribe is False
    assert options.connect_timeout == ROOM_CONNECT_TIMEOUT_S
    assert ROOM_CONNECT_TIMEOUT_S < ROOM_JOIN_TIMEOUT_S
    await joined.leg.close(play_out=False)


async def test_the_relay_subscribes_to_the_persons_microphone_and_nothing_else():
    joined = await join(FakeMedia())
    room = joined.room
    microphone = room.microphone(PERSON)

    camera = room.publish(
        PERSON, kind=rtc.TrackKind.KIND_VIDEO, source=rtc.TrackSource.SOURCE_CAMERA
    )
    video_called_a_microphone = room.publish(PERSON, kind=rtc.TrackKind.KIND_VIDEO)
    screen_audio = room.publish(PERSON, source=rtc.TrackSource.SOURCE_SCREENSHARE_AUDIO)
    unknown = room.publish(PERSON, source=rtc.TrackSource.SOURCE_UNKNOWN)
    someone_else = room.publish("user:someone-else")
    await asyncio.sleep(0.05)

    assert microphone.subscribe_calls == [True]
    for publication in (
        camera,
        video_called_a_microphone,
        screen_audio,
        unknown,
        someone_else,
    ):
        assert publication.subscribe_calls == []
    assert list(joined.media.streams) == [microphone.sid]
    await joined.leg.close(play_out=False)


async def test_a_microphone_already_subscribed_when_the_connect_lands_is_heard():
    joined = await join(FakeMedia(subscribed_at_connect=True))

    microphone = joined.room.microphone(PERSON)
    assert microphone.subscribe_calls == []
    joined.media.say([TEN_MS * 4])
    await until(lambda: len(joined.heard) == 1)
    await joined.leg.close(play_out=False)


async def test_the_persons_audio_reaches_the_call_in_forty_millisecond_frames():
    joined = await join(FakeMedia())

    joined.media.say([TEN_MS] * 4 + [TEN_MS * 10])
    await until(lambda: len(joined.heard) == 3)
    joined.media.say([TEN_MS * 2])
    await until(lambda: len(joined.heard) == 4)

    assert all(len(frame) == FRAME_BYTES for frame in joined.heard)
    assert b"".join(joined.heard) == TEN_MS * 16
    await joined.leg.close(play_out=False)


async def test_the_persons_microphone_published_after_the_join_is_heard():
    joined = await join(FakeMedia(person=None), hearing=False)
    assert joined.presence == []

    joined.room.add_microphone(PERSON)
    await until(lambda: joined.presence == [True])
    joined.media.say([TEN_MS] * 4)

    await until(lambda: len(joined.heard) == 1)
    await joined.leg.close(play_out=False)


@pytest.mark.parametrize(
    "leaves",
    ["microphone_unsubscribed", "participant_left", "room_lost", "stream_ended"],
)
async def test_the_call_is_told_when_the_persons_microphone_leaves_the_room(leaves):
    joined = await join(FakeMedia())
    room = joined.room
    [track_sid] = joined.media.streams
    track = joined.media.streams[track_sid].track

    if leaves == "microphone_unsubscribed":
        room.drop_microphone(PERSON, track)
    elif leaves == "participant_left":
        room.participant_leaves(PERSON)
    elif leaves == "room_lost":
        room.lost()
    else:
        joined.media.streams[track_sid].end()

    await until(lambda: joined.presence == [True, False])
    await until(lambda: joined.media.streams[track_sid].closed)
    await joined.leg.close(play_out=False)
    assert joined.presence == [True, False]


async def test_a_microphone_that_can_no_longer_be_read_counts_as_the_person_gone():
    joined = await join(FakeMedia())
    [track_sid] = joined.media.streams

    joined.media.streams[track_sid].fail(RuntimeError("the stream broke"))

    await until(lambda: joined.presence == [True, False])
    await until(lambda: joined.media.streams[track_sid].closed)
    await joined.leg.close(play_out=False)


async def test_a_reconnect_publishes_the_microphone_again_and_the_newer_one_replaces_the_old():
    joined = await join(FakeMedia())
    room = joined.room
    old = room.microphone(PERSON)
    old_stream = joined.media.streams[old.sid]
    old_track = old_stream.track

    new = room.add_microphone(PERSON)
    await until(lambda: new.sid in joined.media.streams)
    # The old track's unsubscribe can come after its replacement arrived.
    room.drop_microphone(PERSON, old_track)
    joined.media.streams[new.sid].push(TEN_MS * 4)

    await until(lambda: len(joined.heard) == 1)
    await until(lambda: old_stream.closed)
    assert old.subscribe_calls == [True, False]
    assert new.subscribe_calls == [True]
    assert joined.presence == [True]
    await joined.leg.close(play_out=False)


async def test_a_replaced_microphone_that_cannot_be_unsubscribed_still_gives_way(
    monkeypatch,
):
    joined = await join(FakeMedia())
    room = joined.room
    old = room.microphone(PERSON)

    def gone(subscribed: bool) -> None:
        raise RuntimeError("the publication is gone")

    monkeypatch.setattr(old, "set_subscribed", gone)
    new = room.add_microphone(PERSON)
    await until(lambda: new.sid in joined.media.streams)
    joined.media.streams[new.sid].push(TEN_MS * 4)

    await until(lambda: len(joined.heard) == 1)
    assert joined.presence == [True]
    await joined.leg.close(play_out=False)


async def test_a_microphone_announced_again_is_subscribed_once():
    joined = await join(FakeMedia())
    room = joined.room
    microphone = room.microphone(PERSON)

    room.emit("track_published", microphone, room.remote_participants[PERSON])
    await asyncio.sleep(0.05)

    assert microphone.subscribe_calls == [True]
    assert list(joined.media.streams) == [microphone.sid]
    await joined.leg.close(play_out=False)


async def test_the_call_is_told_while_the_persons_audio_stalls_in_the_room():
    joined = await join(FakeMedia())
    room = joined.room
    microphone = room.microphone(PERSON)

    room.mute(PERSON, microphone, True)
    room.mute(PERSON, microphone, False)
    room.quality(PERSON, rtc.ConnectionQuality.QUALITY_LOST)
    room.quality(PERSON, rtc.ConnectionQuality.QUALITY_GOOD)
    room.reconnecting()
    # Muted while the relay's own connection is still getting back.
    room.mute(PERSON, microphone, True)
    room.reconnected()
    room.mute(PERSON, microphone, False)

    assert joined.stalls == [True, False, True, False, True, False]
    await joined.leg.close(play_out=False)


async def test_a_microphone_published_again_is_not_held_muted_by_the_one_it_replaces():
    joined = await join(FakeMedia())
    room = joined.room
    old = room.microphone(PERSON)
    room.mute(PERSON, old, True)

    new = room.add_microphone(PERSON)
    await until(lambda: new.sid in joined.media.streams)

    assert joined.stalls == [True, False]
    await joined.leg.close(play_out=False)


async def test_another_participants_trouble_does_not_stall_the_call():
    joined = await join(FakeMedia())
    room = joined.room
    other = room.publish("user:someone-else")

    room.mute("user:someone-else", other, True)
    room.quality("user:someone-else", rtc.ConnectionQuality.QUALITY_LOST)

    assert joined.stalls == []
    await joined.leg.close(play_out=False)


async def test_otto_is_handed_to_the_room_in_order_without_waiting_for_it_to_play():
    joined = await join(FakeMedia())
    source = joined.source
    source.gate = asyncio.Event()
    chunks = [bytes([n]) * 960 for n in range(3)]

    for chunk in chunks:
        await asyncio.wait_for(joined.leg.play(chunk), timeout=1)
    source.gate.set()

    await until(lambda: source.frames == chunks)
    await joined.leg.close(play_out=False)


async def test_talking_over_otto_drops_what_the_room_has_not_played():
    joined = await join(FakeMedia())
    source = joined.source
    source.gate = asyncio.Event()
    chunks = [bytes([n]) * 960 for n in range(3)]
    for chunk in chunks:
        await joined.leg.play(chunk)
    # The first chunk is on its way into the source.
    await asyncio.sleep(0.05)

    joined.leg.flush()
    assert source.clears == 1
    source.gate.set()

    # The chunk that was on its way in is cleared once it lands, and the
    # rest never reach the source.
    await until(lambda: source.clears == 2)
    await asyncio.sleep(0.05)
    assert source.frames == chunks[:1]
    await joined.leg.close(play_out=False)


def _speech(seconds: float, fill: int) -> bytes:
    return bytes([fill]) * (round(seconds * OUT_RATE) * 2)


async def test_past_the_rooms_queue_the_oldest_of_otto_is_dropped_and_counted():
    dropped_before = _counter("voice_interview_livekit_dropped_bytes_total")
    joined = await join(FakeMedia())
    source = joined.source
    source.gate = asyncio.Event()
    first, older, newest = _speech(1, 1), _speech(ROOM_QUEUE_S - 1, 2), _speech(2, 3)
    await joined.leg.play(first)
    # The first chunk is on its way into the source; the rest wait.
    await asyncio.sleep(0.05)

    await joined.leg.play(older)
    await joined.leg.play(newest)
    source.gate.set()

    await until(lambda: source.frames == [first, newest])
    assert joined.leg.dropped_bytes == len(older)
    assert _counter(
        "voice_interview_livekit_dropped_bytes_total"
    ) - dropped_before == len(older)
    await joined.leg.close(play_out=False)


async def test_a_chunk_of_otto_longer_than_the_rooms_queue_keeps_its_end():
    joined = await join(FakeMedia())
    joined.source.gate = asyncio.Event()
    head, tail = _speech(1, 1), _speech(ROOM_QUEUE_S, 2)

    await joined.leg.play(head + tail)
    joined.source.gate.set()

    await until(lambda: joined.source.frames == [tail])
    assert joined.leg.dropped_bytes == len(head)
    await joined.leg.close(play_out=False)


async def test_otto_after_the_room_is_left_is_dropped():
    joined = await join(FakeMedia())
    await joined.leg.close(play_out=False)

    await joined.leg.play(b"\x01" * 960)

    assert joined.source.frames == []


async def test_leaving_waits_for_ottos_last_words_to_play_out_and_reach_the_browser(
    monkeypatch,
):
    monkeypatch.setattr(edge, "ROOM_PLAYOUT_TAIL_S", 0.3)
    joined = await join(FakeMedia())
    source = joined.source
    source.gate = asyncio.Event()
    source.playing = asyncio.Event()
    await joined.leg.play(b"\x01" * 960)

    leaving = asyncio.create_task(joined.leg.close(play_out=True))
    await asyncio.sleep(0.05)
    assert not joined.room.left
    source.gate.set()
    await until(lambda: source.frames == [b"\x01" * 960])
    await asyncio.sleep(0.05)
    assert not joined.room.left
    source.playing.set()
    # The source's last frame is handed to WebRTC; the room stays while it
    # crosses to the browser.
    await asyncio.sleep(0.1)
    assert not joined.room.left

    await asyncio.wait_for(leaving, timeout=2)
    assert joined.room.left and source.closed
    assert joined.edge.rooms == 0


async def test_leaving_at_once_drops_what_otto_has_not_said():
    joined = await join(FakeMedia())
    joined.source.gate = asyncio.Event()
    joined.source.playing = asyncio.Event()
    await joined.leg.play(b"\x01" * 960)

    await asyncio.wait_for(joined.leg.close(play_out=False), timeout=1)

    assert joined.room.left and joined.source.closed
    assert joined.source.frames == []
    assert joined.edge.rooms == 0


async def test_a_play_out_that_never_ends_is_cut_short(monkeypatch):
    monkeypatch.setattr(edge, "ROOM_PLAYOUT_S", 0.1)
    joined = await join(FakeMedia())
    joined.source.playing = asyncio.Event()
    await joined.leg.play(b"\x01" * 960)

    await asyncio.wait_for(joined.leg.close(play_out=True), timeout=2)

    assert joined.room.left


async def test_a_room_is_left_once_however_often_it_is_closed():
    joined = await join(FakeMedia())

    await joined.leg.close(play_out=False)
    await joined.leg.close(play_out=True)

    assert joined.edge.rooms == 0
    assert joined.presence == [True]


async def test_a_room_whose_teardown_fails_is_still_counted_left():
    joined = await join(FakeMedia())
    joined.source.close_fails = ConnectionError("the SDK went away")

    await joined.leg.close(play_out=False)

    assert joined.room.left
    assert joined.edge.rooms == 0


async def test_participant_seconds_count_the_relays_time_in_the_room_and_the_persons():
    now = [100.0]
    before = _counter("voice_interview_livekit_participant_seconds_total")
    joined = await join(FakeMedia(), clock=lambda: now[0])

    now[0] = 130.0
    joined.room.participant_leaves(PERSON)
    now[0] = 160.0
    await joined.leg.close(play_out=False)

    # The relay was in the room 60 s, the person 30 s of them.
    assert _counter(
        "voice_interview_livekit_participant_seconds_total"
    ) - before == pytest.approx(90.0)


async def test_a_refused_join_raises_and_leaves_no_room_behind():
    media = FakeMedia(refuse=ConnectionError("the plan's allowance is used up"))
    room_edge = livekit_edge(media)
    listener = Listener()

    with pytest.raises(ConnectionError):
        await listener.join(room_edge)

    assert room_edge.rooms == 0
    assert not media.rooms[0].connected
    assert listener.presence == []


async def test_a_join_past_its_deadline_leaves_the_room_once_the_connect_lands():
    media = FakeMedia()
    media.connect_gate = asyncio.Event()
    room_edge = livekit_edge(media, join_timeout_s=0.1)

    with pytest.raises(TimeoutError):
        await Listener().join(room_edge)

    # The SDK's connect goes on: the room is still counted, not yet made.
    room = media.rooms[0]
    assert room_edge.rooms == 1 and not room.connected
    media.connect_gate.set()

    await until(lambda: room.left)
    assert room_edge.rooms == 0


async def test_a_join_the_call_gave_up_on_leaves_the_room_once_the_connect_lands():
    media = FakeMedia()
    media.connect_gate = asyncio.Event()
    room_edge = livekit_edge(media)
    joining = asyncio.create_task(Listener().join(room_edge))
    await until(lambda: bool(media.rooms))

    joining.cancel()
    with pytest.raises(asyncio.CancelledError):
        await joining

    room = media.rooms[0]
    assert room_edge.rooms == 1 and not room.connected
    media.connect_gate.set()
    await until(lambda: room.left)
    assert room_edge.rooms == 0


async def test_a_connect_that_fails_after_the_join_gave_up_leaves_nothing_behind():
    media = FakeMedia(refuse=ConnectionError("the network is gone"))
    media.connect_gate = asyncio.Event()
    room_edge = livekit_edge(media, join_timeout_s=0.1)

    with pytest.raises(TimeoutError):
        await Listener().join(room_edge)
    media.connect_gate.set()

    await until(lambda: room_edge.rooms == 0)
    assert not media.rooms[0].connected and not media.rooms[0].left


async def test_the_relay_takes_the_persons_audio_at_16khz_mono_and_plays_otto_at_24khz(
    monkeypatch,
):
    opened: list[dict] = []

    def open_stream(track, **options):
        opened.append(options)
        return object()

    monkeypatch.setattr(edge.rtc, "AudioStream", open_stream)
    media = RoomMedia()

    media.stream(object())  # type: ignore[arg-type]
    frame = media.frame(bytes(1920))
    source = media.source()

    assert opened == [{"sample_rate": IN_RATE, "num_channels": 1, "frame_size_ms": 40}]
    assert (frame.sample_rate, frame.num_channels, frame.samples_per_channel) == (
        OUT_RATE,
        1,
        960,
    )
    assert (source.sample_rate, source.num_channels) == (OUT_RATE, 1)
    await source.aclose()


async def test_the_relay_makes_a_real_room_and_publishes_otto_as_otto():
    media = RoomMedia()
    source = media.source()

    room = media.room()
    track = media.track(source)

    assert isinstance(room, rtc.Room) and not room.isconnected()
    assert isinstance(track, rtc.LocalAudioTrack) and track.name == "otto"
    await source.aclose()
