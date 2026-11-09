"""The LiveKit leg's room, against fakes of the SDK's room, source and
stream: what the relay joins, what it hears and plays, and how it leaves."""

import asyncio

import pytest
from livekit import rtc
from prometheus_client import REGISTRY

from backend.voice_relay import edge
from backend.voice_relay.config import IN_RATE, OUT_RATE, ROOM_QUEUE_S
from backend.voice_relay.conftest import LIVEKIT, FakeMedia, livekit_edge, until
from backend.voice_relay.edge import RoomLeg, RoomMedia

CALL_ID = "e" * 32
PERSON = "user:user-1"
# 10 ms of the person's audio, and the call's 40 ms frame.
TEN_MS = bytes(range(256)) + bytes(64)
FRAME_BYTES = 1280


class Joined:
    def __init__(self, media: FakeMedia, leg: RoomLeg, edge_: edge.LiveKitEdge):
        self.media = media
        self.leg = leg
        self.edge = edge_
        self.heard: list[bytes] = []
        self.presence: list[bool] = []

    @property
    def room(self):
        return self.media.rooms[-1]

    @property
    def source(self):
        return self.media.sources[-1]


async def join(media: FakeMedia, **options) -> Joined:
    room_edge = livekit_edge(media, **options)
    heard: list[bytes] = []
    presence: list[bool] = []

    async def hear(pcm: bytes) -> None:
        heard.append(pcm)

    leg = await room_edge.join(
        call_id=CALL_ID,
        user_id="user-1",
        ttl_s=360,
        hear=hear,
        presence=presence.append,
    )
    joined = Joined(media, leg, room_edge)
    joined.heard, joined.presence = heard, presence
    return joined


def _participant_seconds() -> float:
    return (
        REGISTRY.get_sample_value("voice_interview_livekit_participant_seconds_total")
        or 0.0
    )


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
    # The person's browser joined first: their microphone is in the room.
    assert joined.presence == [True]
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
    joined = await join(FakeMedia(person=None))
    assert joined.presence == []

    joined.room.add_microphone(PERSON)
    joined.media.say([TEN_MS] * 4)

    await until(lambda: len(joined.heard) == 1)
    assert joined.presence == [True]
    await joined.leg.close(play_out=False)


async def test_tracks_that_are_not_the_persons_microphone_are_not_heard():
    joined = await join(FakeMedia(person=None))

    joined.room.add_microphone("user:someone-else")
    joined.room.add_microphone(PERSON, kind=rtc.TrackKind.KIND_VIDEO)

    assert joined.media.streams == {}
    assert joined.presence == []
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


async def test_a_reconnect_brings_the_microphone_back_as_a_new_track_that_replaces_the_old():
    joined = await join(FakeMedia())
    room = joined.room
    [old_sid] = joined.media.streams
    old = joined.media.streams[old_sid]

    new_track = room.add_microphone(PERSON)
    # The old track's unsubscribe can come after its replacement arrived.
    room.drop_microphone(PERSON, old.track)
    joined.media.streams[new_track.sid].push(TEN_MS * 4)

    await until(lambda: len(joined.heard) == 1)
    await until(lambda: old.closed)
    assert joined.presence == [True]
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


async def test_otto_past_the_rooms_queue_is_dropped():
    joined = await join(FakeMedia())
    joined.source.gate = asyncio.Event()
    too_long = bytes(round((ROOM_QUEUE_S + 1) * OUT_RATE) * 2)

    await joined.leg.play(too_long)

    assert joined.leg.dropped_bytes == len(too_long)
    await joined.leg.close(play_out=False)


async def test_leaving_waits_for_ottos_last_words_to_play_out():
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


async def test_otto_after_the_room_is_left_is_dropped():
    joined = await join(FakeMedia())
    await joined.leg.close(play_out=False)

    await joined.leg.play(b"\x01" * 960)

    assert joined.source.frames == []


async def test_a_room_whose_teardown_fails_is_still_counted_left():
    joined = await join(FakeMedia())
    joined.source.close_fails = ConnectionError("the SDK went away")

    await joined.leg.close(play_out=False)

    assert joined.room.left
    assert joined.edge.rooms == 0


async def test_a_room_is_left_once_however_often_it_is_closed():
    joined = await join(FakeMedia())

    await joined.leg.close(play_out=False)
    await joined.leg.close(play_out=True)

    assert joined.edge.rooms == 0
    assert joined.presence == [True]


async def test_participant_seconds_count_the_relays_time_in_the_room_and_the_persons():
    now = [100.0]
    before = _participant_seconds()
    joined = await join(FakeMedia(), clock=lambda: now[0])

    now[0] = 130.0
    joined.room.participant_leaves(PERSON)
    now[0] = 160.0
    await joined.leg.close(play_out=False)

    # The relay was in the room 60 s, the person 30 s of them.
    assert _participant_seconds() - before == pytest.approx(90.0)


async def test_a_refused_join_raises_and_leaves_no_room_behind():
    media = FakeMedia(refuse=ConnectionError("the plan's allowance is used up"))
    room_edge = livekit_edge(media)
    presence: list[bool] = []

    async def hear(pcm: bytes) -> None:
        raise AssertionError("nothing is heard from a room never joined")

    with pytest.raises(ConnectionError):
        await room_edge.join(
            call_id=CALL_ID,
            user_id="user-1",
            ttl_s=360,
            hear=hear,
            presence=presence.append,
        )

    assert room_edge.rooms == 0
    assert presence == []


async def test_a_join_that_never_answers_is_given_up():
    room_edge = livekit_edge(FakeMedia(hang=True), join_timeout_s=0.1)

    async def hear(pcm: bytes) -> None:
        return None

    with pytest.raises(TimeoutError):
        await room_edge.join(
            call_id=CALL_ID,
            user_id="user-1",
            ttl_s=360,
            hear=hear,
            presence=lambda present: None,
        )

    assert room_edge.rooms == 0


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
