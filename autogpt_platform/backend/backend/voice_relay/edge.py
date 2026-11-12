"""A call's audio, through the call's LiveKit room.

The person's browser joins the call's room while it opens the call's socket,
and the relay joins the same room as Otto while it opens the call, beside the
model session's connect (``rooms``). The relay hears one track, the person's
microphone: it subscribes to that publication alone, at 16 kHz mono, which the
SDK resamples to, and hands it to the call in frames of ``FRAME_S``. It
publishes Otto's speech from a 24 kHz source, which plays it out at real time,
as it arrives; when the person talks over Otto, what has not played yet is
dropped. The control socket stays open beside the room and carries everything
else.

The person's microphone can leave the room and come back, as LiveKit's own
reconnect does; the call is told each time (``presence``). While it is muted,
or a connection is getting back on its feet, the call is told its audio has
stalled (``stall``). The relay's own connection to the room, lost past the
SDK's reconnect, is the call's audio gone: the call is told at once
(``lost``). When the call ends, what Otto still has to say plays out, and the
relay leaves.

The SDK's connect always runs to its end, even when the join gives up on it
at its deadline or because the call ended first: the room it makes is left
once it lands, and counted until then.

LiveKit bills participant-minutes: each room adds the relay's time in it and
the person's to ``voice_interview_livekit_participant_seconds_total``.
"""

import asyncio
import collections
import logging
import time
from collections.abc import Awaitable, Callable

from livekit import rtc

from backend.voice_relay import rooms
from backend.voice_relay.config import (
    FRAME_S,
    IN_RATE,
    OUT_RATE,
    RELAY_IDENTITY,
    ROOM_CONNECT_TIMEOUT_S,
    ROOM_JOIN_TIMEOUT_S,
    ROOM_LEAVE_TIMEOUT_S,
    ROOM_PLAYOUT_S,
    ROOM_PLAYOUT_TAIL_S,
    ROOM_QUEUE_S,
)
from backend.voice_relay.metering import (
    LIVEKIT_DROPPED_BYTES,
    LIVEKIT_PARTICIPANT_SECONDS,
)
from backend.voice_relay.waits import bounded, stop

logger = logging.getLogger(__name__)

# The person's audio, in frames of FRAME_S.
Hear = Callable[[bytes], Awaitable[None]]
# Whether the person's microphone is in the room.
Presence = Callable[[bool], None]
# Whether the person's audio has stopped for now while they stay in the room.
Stall = Callable[[bool], None]
# The relay's own connection to the room is gone past the SDK's reconnect.
Lost = Callable[[], None]

_FRAME_BYTES = round(IN_RATE * FRAME_S) * 2
_FRAME_MS = round(FRAME_S * 1000)


class RoomMedia:
    """The LiveKit SDK objects a call's room is made of; tests stand in for
    them."""

    def room(self) -> rtc.Room:
        return rtc.Room()

    def source(self) -> rtc.AudioSource:
        return rtc.AudioSource(OUT_RATE, 1)

    def track(self, source: rtc.AudioSource) -> rtc.LocalAudioTrack:
        return rtc.LocalAudioTrack.create_audio_track(RELAY_IDENTITY, source)

    def stream(self, track: rtc.Track) -> rtc.AudioStream:
        return rtc.AudioStream(
            track, sample_rate=IN_RATE, num_channels=1, frame_size_ms=_FRAME_MS
        )

    def frame(self, pcm: bytes) -> rtc.AudioFrame:
        return rtc.AudioFrame(pcm, OUT_RATE, 1, len(pcm) // 2)


class LiveKitEdge:
    """Joins call rooms for this relay, and counts the rooms it is still in,
    so a drain waits for Otto's last words to play out and for any room a
    join gave up on to be left."""

    def __init__(
        self,
        project: rooms.LiveKitProject,
        *,
        media: RoomMedia | None = None,
        join_timeout_s: float = ROOM_JOIN_TIMEOUT_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._project = project
        self._media = media or RoomMedia()
        self._join_timeout_s = join_timeout_s
        self._clock = clock
        self._legs: set[RoomLeg] = set()

    @property
    def rooms(self) -> int:
        """Rooms joined, or being joined, and not yet left."""
        return len(self._legs)

    async def join(
        self,
        *,
        call_id: str,
        user_id: str,
        ttl_s: float,
        hear: Hear,
        presence: Presence,
        stall: Stall,
        lost: Lost,
    ) -> "RoomLeg":
        """Join call *call_id*'s room as Otto, for at most *ttl_s*.

        Raises whatever the join failed with, or ``TimeoutError``. The room
        is left either way, once the SDK's connect has landed.
        """
        token = rooms.relay_token(call_id=call_id, project=self._project, ttl_s=ttl_s)
        leg = RoomLeg(
            media=self._media,
            person=rooms.person_identity(user_id, call_id),
            hear=hear,
            presence=presence,
            stall=stall,
            lost=lost,
            on_left=self._legs.discard,
            clock=self._clock,
        )
        self._legs.add(leg)
        try:
            await bounded(
                leg.join(self._project.url, token), timeout=self._join_timeout_s
            )
        except BaseException:
            await leg.close(play_out=False)
            raise
        return leg


class RoomLeg:
    """One call's audio through its room: the call's ``AudioLeg``."""

    def __init__(
        self,
        *,
        media: RoomMedia,
        person: str,
        hear: Hear,
        presence: Presence,
        stall: Stall,
        lost: Lost,
        on_left: Callable[["RoomLeg"], None],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._media = media
        self._person = person
        self._hear = hear
        self._presence = presence
        self._stall = stall
        self._lost = lost
        self._on_left = on_left
        self._clock = clock
        self._room: rtc.Room | None = None
        self._connecting: asyncio.Task[None] | None = None
        self._late_leave: asyncio.Task[None] | None = None
        self._source: rtc.AudioSource | None = None
        # Otto's track, as the relay published it.
        self._otto: rtc.LocalTrackPublication | None = None
        # The person's microphone: the one publication the relay subscribes
        # to, its track, the stream over it, and the task reading the
        # stream. A stream is closed here, never by its reader, which a
        # cancel can stop before it ever runs.
        self._publication: rtc.RemoteTrackPublication | None = None
        self._track_sid: str | None = None
        self._stream: rtc.AudioStream | None = None
        self._reader: asyncio.Task | None = None
        self._closing: set[asyncio.Task] = set()
        self._feeder: asyncio.Task | None = None
        # What has stopped the person's audio while they stay in the room.
        self._muted = False
        self._connection_lost = False
        self._reconnecting = False
        self._stalled = False
        # Otto's speech the source has not taken yet.
        self._queue: collections.deque[bytes] = collections.deque()
        self._queued_bytes = 0
        self._max_queued_bytes = round(ROOM_QUEUE_S * OUT_RATE) * 2
        self._wake = asyncio.Event()
        # Set while the source has taken everything queued.
        self._fed = asyncio.Event()
        self._fed.set()
        # Bumped by each flush, so a frame in flight is dropped too.
        self._flushes = 0
        self._partial = b""
        self._joined_at: float | None = None
        self._person_since: float | None = None
        self._person_s = 0.0
        self._leaving = False
        self._left = False
        # Otto's speech dropped past the queue's length, in bytes.
        self.dropped_bytes = 0

    # --- joining and leaving ---------------------------------------------

    async def join(self, url: str, token: str) -> None:
        room = self._media.room()
        self._room = room
        room.on("track_published", self._on_track_published)
        room.on("track_subscribed", self._on_track_subscribed)
        room.on("track_unsubscribed", self._on_track_unsubscribed)
        room.on("track_muted", self._on_track_muted)
        room.on("track_unmuted", self._on_track_unmuted)
        room.on("connection_quality_changed", self._on_connection_quality)
        room.on("participant_disconnected", self._on_participant_disconnected)
        room.on("reconnecting", self._on_reconnecting)
        room.on("reconnected", self._on_reconnected)
        room.on("disconnected", self._on_disconnected)
        # Awaited through a shield: the SDK's connect runs to its end whatever
        # happens to the join, so that the room it makes can be left.
        connecting = asyncio.ensure_future(
            room.connect(
                url,
                token,
                rtc.RoomOptions(
                    auto_subscribe=False, connect_timeout=ROOM_CONNECT_TIMEOUT_S
                ),
            )
        )
        self._connecting = connecting
        await asyncio.shield(connecting)
        self._joined_at = self._clock()
        source = self._media.source()
        self._source = source
        self._otto = await room.local_participant.publish_track(
            self._media.track(source),
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
        )
        self._feeder = asyncio.create_task(self._feed(source), name="room-feed")
        # The person's microphone, published before the relay joined.
        person = room.remote_participants.get(self._person)
        if person is not None:
            for publication in list(person.track_publications.values()):
                self._consider(publication, person)

    async def otto_heard(self) -> None:
        """Return once someone in the room subscribes to Otto's track: from
        then what he says reaches the person."""
        if self._otto is not None:
            await self._otto.wait_for_subscription()

    async def close(self, *, play_out: bool) -> None:
        """Leave the room, once what Otto has said has played when
        *play_out*. Idempotent."""
        if self._leaving:
            return
        self._leaving = True
        try:
            if play_out and self._source is not None:
                await self._play_out(self._source)
        finally:
            await self._leave()

    async def _play_out(self, source: rtc.AudioSource) -> None:
        async def played() -> None:
            await self._fed.wait()
            await source.wait_for_playout()

        try:
            await bounded(played(), timeout=ROOM_PLAYOUT_S)
        except Exception:
            logger.info("Voice call: Otto's last words did not play out in time")
            return
        await asyncio.sleep(ROOM_PLAYOUT_TAIL_S)

    async def _leave(self) -> None:
        connecting = self._connecting
        if connecting is not None and not connecting.done():
            # The join gave up on a connect still under way: the room it makes
            # is left once it lands, and stays counted until then.
            self._late_leave = asyncio.create_task(
                self._leave_once_connected(connecting), name="room-late-leave"
            )
            return
        await self._leave_now()

    async def _leave_once_connected(self, connecting: asyncio.Task[None]) -> None:
        await asyncio.wait({connecting})
        if not connecting.cancelled() and connecting.exception() is not None:
            logger.info("Voice call: a room join given up on failed after all")
        await self._leave_now()

    async def _leave_now(self) -> None:
        try:
            reader, feeder = self._reader, self._feeder
            self._stop_reading()
            await stop(task for task in (reader, feeder) if task is not None)
            await asyncio.gather(*self._closing, return_exceptions=True)
            room, source = self._room, self._source
            if room is not None:
                await _bounded(room.disconnect(), "the room was not left cleanly")
            if source is not None:
                await _bounded(source.aclose(), "Otto's source did not close cleanly")
        finally:
            self._count_participant_seconds()
            if not self._left:
                self._left = True
                self._on_left(self)

    def _count_participant_seconds(self) -> None:
        now = self._clock()
        self._person_went(now)
        relay_s = now - self._joined_at if self._joined_at is not None else 0.0
        LIVEKIT_PARTICIPANT_SECONDS.inc(relay_s + self._person_s)
        self._joined_at = None

    # --- the person's microphone -----------------------------------------

    def _on_track_published(
        self,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        self._consider(publication, participant)

    def _consider(
        self,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        """Subscribe to *publication* when it is the person's microphone."""
        if (
            self._leaving
            or participant.identity != self._person
            or publication.kind != rtc.TrackKind.KIND_AUDIO
            or publication.source != rtc.TrackSource.SOURCE_MICROPHONE
        ):
            return
        current = self._publication
        if current is not None and current.sid == publication.sid:
            return
        # A full reconnect publishes the microphone again: the newer one
        # replaces the one before, and the person stays in the call while
        # it arrives. Muting belongs to a publication, so the newer one
        # counts as muted only if it is.
        self._publication = publication
        self._muted = publication.muted
        self._update_stall()
        if current is not None:
            self._stop_reading()
            _unsubscribe(current)
        if publication.track is not None:
            self._read(publication.track)
        else:
            publication.set_subscribed(True)

    def _on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if not self._leaving and self._is_the_microphone(publication):
            self._read(track)

    def _on_track_unsubscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        # Only the track being read: a track a reconnect replaced can be
        # unsubscribed after its replacement arrived.
        if participant.identity == self._person and track.sid == self._track_sid:
            self._person_left()

    def _on_participant_disconnected(self, participant: rtc.RemoteParticipant) -> None:
        if participant.identity == self._person:
            self._publication = None
            self._muted = self._connection_lost = False
            self._person_left()
            self._update_stall()

    def _on_disconnected(self, reason: object = None) -> None:
        # The relay's own connection to the room, lost past the SDK's own
        # reconnect; or the relay leaving, which needs nothing more.
        if not self._leaving:
            logger.warning(f"Voice call: the relay lost its room ({reason})")
            self._person_left()
            self._lost()

    def _is_the_microphone(self, publication: rtc.TrackPublication) -> bool:
        current = self._publication
        return current is not None and publication.sid == current.sid

    def _read(self, track: rtc.Track) -> None:
        self._stop_reading()
        stream = self._media.stream(track)
        self._track_sid = track.sid
        self._stream = stream
        self._partial = b""
        self._reader = asyncio.create_task(
            self._read_stream(track.sid, stream), name="room-read"
        )
        if self._person_since is None:
            self._person_since = self._clock()
            self._presence(True)

    def _person_left(self) -> None:
        self._stop_reading()
        if self._person_since is not None:
            self._person_went(self._clock())
            if not self._leaving:
                self._presence(False)

    def _stop_reading(self) -> None:
        reader, self._reader = self._reader, None
        stream, self._stream = self._stream, None
        self._track_sid = None
        if reader is not None and reader is not asyncio.current_task():
            reader.cancel()
        if stream is not None:
            closing = asyncio.create_task(self._close_stream(stream))
            self._closing.add(closing)
            closing.add_done_callback(self._closing.discard)

    async def _close_stream(self, stream: rtc.AudioStream) -> None:
        await _bounded(stream.aclose(), "the person's audio did not close cleanly")

    def _person_went(self, now: float) -> None:
        if self._person_since is not None:
            self._person_s += now - self._person_since
            self._person_since = None

    async def _read_stream(self, sid: str, stream: rtc.AudioStream) -> None:
        try:
            async for event in stream:
                for frame in self._frames(event.frame.data.tobytes()):
                    await self._hear(frame)
        except Exception:
            logger.warning("Voice call: the person's audio stopped", exc_info=True)
        # The track ended under the stream, or reading it failed: the relay
        # no longer hears them.
        if sid == self._track_sid:
            self._person_left()

    def _frames(self, pcm: bytes) -> list[bytes]:
        """*pcm* cut into the call's frames, keeping any remainder for the
        next."""
        buffered = self._partial + pcm
        whole = len(buffered) - len(buffered) % _FRAME_BYTES
        self._partial = buffered[whole:]
        return [
            buffered[start : start + _FRAME_BYTES]
            for start in range(0, whole, _FRAME_BYTES)
        ]

    # --- a stalled microphone ----------------------------------------------

    def _on_track_muted(
        self, participant: rtc.Participant, publication: rtc.TrackPublication
    ) -> None:
        if self._is_the_microphone(publication):
            self._muted = True
            self._update_stall()

    def _on_track_unmuted(
        self, participant: rtc.Participant, publication: rtc.TrackPublication
    ) -> None:
        if self._is_the_microphone(publication):
            self._muted = False
            self._update_stall()

    def _on_connection_quality(
        self, participant: rtc.Participant | None, quality: rtc.ConnectionQuality
    ) -> None:
        if participant is not None and participant.identity == self._person:
            self._connection_lost = quality == rtc.ConnectionQuality.QUALITY_LOST
            self._update_stall()

    def _on_reconnecting(self) -> None:
        self._reconnecting = True
        self._update_stall()

    def _on_reconnected(self) -> None:
        self._reconnecting = False
        self._update_stall()

    def _update_stall(self) -> None:
        stalled = self._muted or self._connection_lost or self._reconnecting
        if stalled != self._stalled and not self._leaving:
            self._stalled = stalled
            self._stall(stalled)

    # --- Otto's speech ---------------------------------------------------

    async def play(self, pcm: bytes) -> None:
        """Queue Otto's speech, 24 kHz PCM16, for the room; never waits for
        it to play."""
        usable = pcm[: len(pcm) - len(pcm) % 2]
        if not usable or self._leaving or self._source is None:
            return
        # Past the queue's length the oldest speech waiting is dropped, so
        # the room keeps to Otto's newest words.
        if len(usable) > self._max_queued_bytes:
            self._drop(len(usable) - self._max_queued_bytes)
            usable = usable[-self._max_queued_bytes :]
        self._queue.append(usable)
        self._queued_bytes += len(usable)
        while self._queued_bytes > self._max_queued_bytes:
            oldest = self._queue.popleft()
            self._queued_bytes -= len(oldest)
            self._drop(len(oldest))
        self._fed.clear()
        self._wake.set()

    def _drop(self, size: int) -> None:
        self.dropped_bytes += size
        LIVEKIT_DROPPED_BYTES.inc(size)

    def flush(self) -> None:
        """Drop what Otto has not said yet: the person talked over it."""
        self._queue.clear()
        self._queued_bytes = 0
        self._flushes += 1
        if self._source is not None:
            self._source.clear_queue()
        self._fed.set()

    async def _feed(self, source: rtc.AudioSource) -> None:
        """Hand the queue to the source, which takes it at real time."""
        while True:
            if not self._queue:
                self._fed.set()
                self._wake.clear()
                await self._wake.wait()
                continue
            pcm = self._queue.popleft()
            self._queued_bytes -= len(pcm)
            flushes = self._flushes
            await source.capture_frame(self._media.frame(pcm))
            if flushes != self._flushes:
                # Flushed while this frame was on its way in.
                source.clear_queue()


def _unsubscribe(publication: rtc.RemoteTrackPublication) -> None:
    try:
        publication.set_subscribed(False)
    except Exception:
        logger.debug("Voice call: a replaced microphone was gone already")


async def _bounded(operation: Awaitable[object], failed: str) -> None:
    """Await an SDK call that tears something down, for at most
    ``ROOM_LEAVE_TIMEOUT_S``; one that fails or hangs is only logged."""
    try:
        await bounded(operation, timeout=ROOM_LEAVE_TIMEOUT_S)
    except Exception:
        logger.info(f"Voice call: {failed}", exc_info=True)
