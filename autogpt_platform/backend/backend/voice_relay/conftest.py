import asyncio
import json
import socket
import tempfile
import time
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from types import SimpleNamespace

import jwt
import numpy as np
import pytest
import uvicorn
from fakeredis import FakeServer
from fakeredis.aioredis import FakeRedis
from livekit import api, rtc

from backend.api.features.onboarding_dump import rate_limit
from backend.api.utils.cors import build_cors_params
from backend.util.settings import AppEnvironment, BehaveAs
from backend.voice_relay import rooms, tokens
from backend.voice_relay.app import Relay, RelayDeps, create_app
from backend.voice_relay.config import (
    CAPS,
    IN_RATE,
    ROOM_PREFIX,
    TIMING,
    WS_MAX_MESSAGE_BYTES,
    Caps,
    Timing,
)
from backend.voice_relay.edge import LiveKitEdge
from backend.voice_relay.fake_vendor import FakeVendor
from backend.voice_relay.metering import Row
from backend.voice_relay.postcall import PostCall, Transcription
from backend.voice_relay.room_reaper import RoomReaper
from backend.voice_relay.rooms import LiveKitProject
from backend.voice_relay.serve import DrainingServer

BUCKET = "test-bucket"

SIGNING_KEY = "k" * 48  # pragma: allowlist secret
ORIGIN = "http://localhost:3000"


@pytest.fixture(autouse=True)
def signing_key(monkeypatch):
    monkeypatch.setattr(tokens.settings.secrets, "voice_relay_signing_key", SIGNING_KEY)
    return SIGNING_KEY


@pytest.fixture
async def redis():
    client = FakeRedis(server=FakeServer(), decode_responses=True)
    yield client
    await client.aclose()


@pytest.fixture
def counted(mocker, set_platform_payment) -> None:
    """Calls count against the person's allowance: the platform takes payment
    and runs as the cloud."""
    mocker.patch.object(rate_limit.settings.config, "behave_as", BehaveAs.CLOUD)
    set_platform_payment(True)


async def counted_calls(redis) -> list[int]:
    """The person's calls counted against their allowance, today's and this
    hour's."""
    keys = sorted(await redis.keys("onboarding:voice_interview:*"))
    return [int(await redis.get(key)) for key in keys]


def speech(seconds: float) -> list[bytes]:
    """40 ms frames of a tone loud enough to count as speech."""
    frame = int(IN_RATE * 0.04)
    samples = np.arange(frame)
    tone = (np.sin(samples * 2 * np.pi * 300 / IN_RATE) * 8000).astype(np.int16)
    return [tone.tobytes() for _ in range(max(1, round(seconds / 0.04)))]


async def until(condition: Callable[[], bool], timeout: float = 5.0) -> None:
    """Wait until *condition* holds."""

    async def poll() -> None:
        while not condition():
            await asyncio.sleep(0.02)

    await asyncio.wait_for(poll(), timeout=timeout)


def silence(seconds: float) -> list[bytes]:
    frame = bytes(int(IN_RATE * 0.04) * 2)
    return [frame for _ in range(max(1, round(seconds / 0.04)))]


LIVEKIT = LiveKitProject(
    url="wss://agpt-test.livekit.cloud",
    api_key="APItest",
    api_secret="r" * 48,  # pragma: allowlist secret
)


class FakeTrack:
    """A track in a call's room, as the relay's SDK hands it over."""

    def __init__(self, sid: str, kind: int = rtc.TrackKind.KIND_AUDIO) -> None:
        self.sid = sid
        self.kind = kind


class FakePublication:
    """A track someone published in the room. With auto-subscribe off, as
    the relay joins, nothing reaches the relay until it subscribes, and the
    track then comes a turn of the loop later, as the SDK's does."""

    def __init__(
        self,
        room: "FakeRoom",
        participant: "FakeParticipant",
        sid: str,
        *,
        kind: int,
        source: int,
        subscribed: bool = False,
    ) -> None:
        self.room = room
        self.participant = participant
        self.sid = sid
        self.kind = kind
        self.source = source
        self.track: FakeTrack | None = FakeTrack(sid, kind) if subscribed else None
        self.muted = False
        # Every set_subscribed the relay made, in order.
        self.subscribe_calls: list[bool] = []

    def set_subscribed(self, subscribed: bool) -> None:
        self.subscribe_calls.append(subscribed)
        loop = asyncio.get_running_loop()
        if subscribed and self.track is None:
            self.track = FakeTrack(self.sid, self.kind)
            loop.call_soon(
                self.room.emit, "track_subscribed", self.track, self, self.participant
            )
        elif not subscribed and self.track is not None:
            track, self.track = self.track, None
            loop.call_soon(
                self.room.emit, "track_unsubscribed", track, self, self.participant
            )


class FakeParticipant:
    def __init__(self, identity: str) -> None:
        self.identity = identity
        self.track_publications: dict[str, FakePublication] = {}


class FakeAudioStream:
    """A microphone in the room, read by the relay: what the test pushes."""

    def __init__(self, track: FakeTrack) -> None:
        self.track = track
        self.closed = False
        self._frames: asyncio.Queue[bytes | Exception | None] = asyncio.Queue()

    def push(self, pcm: bytes) -> None:
        self._frames.put_nowait(pcm)

    def end(self) -> None:
        """The track ends under the stream."""
        self._frames.put_nowait(None)

    def fail(self, error: Exception) -> None:
        """Reading the stream fails with *error*."""
        self._frames.put_nowait(error)

    def __aiter__(self) -> "FakeAudioStream":
        return self

    async def __anext__(self) -> SimpleNamespace:
        pcm = await self._frames.get()
        if pcm is None:
            raise StopAsyncIteration
        if isinstance(pcm, Exception):
            raise pcm
        return SimpleNamespace(frame=SimpleNamespace(data=memoryview(pcm)))

    async def aclose(self) -> None:
        self.closed = True


class FakeAudioSource:
    """Otto's microphone in the room: it keeps every frame it takes."""

    def __init__(self) -> None:
        self.frames: list[bytes] = []
        self.clears = 0
        self.closed = False
        # While set and clear, each capture waits on it.
        self.gate: asyncio.Event | None = None
        # While set and clear, the play-out waits on it.
        self.playing: asyncio.Event | None = None
        # Closing it fails with this, as an SDK call that tears down can.
        self.close_fails: Exception | None = None

    async def capture_frame(self, frame: bytes) -> None:
        if self.gate is not None:
            await self.gate.wait()
        self.frames.append(frame)

    def clear_queue(self) -> None:
        self.clears += 1

    async def wait_for_playout(self) -> None:
        if self.playing is not None:
            await self.playing.wait()

    async def aclose(self) -> None:
        if self.close_fails is not None:
            raise self.close_fails
        self.closed = True

    @property
    def audio(self) -> bytes:
        return b"".join(self.frames)


class FakeRoom:
    """A call's LiveKit room as the relay's SDK shows it. The person's
    browser joined first, so the room has them in it, their microphone
    published, when the relay connects. Like the SDK's, it is connected only
    once its connect has landed, and only a connected room can be left."""

    def __init__(self, media: "FakeMedia") -> None:
        self.media = media
        self.handlers: dict[str, list[Callable]] = {}
        self.remote_participants: dict[str, FakeParticipant] = {}
        self.local_participant = SimpleNamespace(
            published=[], publish_track=self._publish
        )
        self.url: str | None = None
        self.token: str | None = None
        self.options: object = None
        self.connected = False
        self.left = False
        # The identity the person joined this call's room under, once the
        # relay's connect names the call.
        self.person: str | None = None

    def on(self, event: str, callback: Callable) -> Callable:
        self.handlers.setdefault(event, []).append(callback)
        return callback

    def emit(self, event: str, *args: object) -> None:
        for callback in list(self.handlers.get(event, [])):
            callback(*args)

    async def connect(self, url: str, token: str, options: object = None) -> None:
        self.url, self.token, self.options = url, token, options
        call_id = self.claims["video"]["room"].removeprefix(ROOM_PREFIX)
        if self.media.user is not None:
            self.person = rooms.person_identity(self.media.user, call_id)
        if self.media.connect_gate is not None:
            await self.media.connect_gate.wait()
        if self.media.refuse is not None:
            raise self.media.refuse
        self.connected = True
        if self.person is not None:
            self.publish(
                self.person,
                announce=False,
                subscribed=self.media.subscribed_at_connect,
            )

    async def disconnect(self) -> None:
        if not self.connected:
            return
        self.connected = False
        self.left = True
        self.emit("disconnected", rtc.DisconnectReason.CLIENT_INITIATED)

    async def _publish(self, track: object, options: object) -> None:
        self.local_participant.published.append((track, options))

    # --- the room's side ---

    @property
    def claims(self) -> dict:
        """What the relay's token for this room says."""
        assert self.token is not None
        return jwt.decode(self.token, options={"verify_signature": False})

    def publish(
        self,
        identity: str,
        *,
        kind: int = rtc.TrackKind.KIND_AUDIO,
        source: int = rtc.TrackSource.SOURCE_MICROPHONE,
        announce: bool = True,
        subscribed: bool = False,
    ) -> FakePublication:
        """*identity* publishes a track; *announce* as the SDK does for one
        published after the relay joined."""
        participant = self.remote_participants.setdefault(
            identity, FakeParticipant(identity)
        )
        publication = FakePublication(
            self,
            participant,
            self.media.track_sid(),
            kind=kind,
            source=source,
            subscribed=subscribed,
        )
        participant.track_publications[publication.sid] = publication
        if announce:
            self.emit("track_published", publication, participant)
        return publication

    def add_microphone(self, identity: str) -> FakePublication:
        """*identity* publishes a microphone after the relay joined."""
        return self.publish(identity)

    def drop_microphone(self, identity: str, track: FakeTrack) -> None:
        participant = self.remote_participants[identity]
        publication = participant.track_publications.pop(track.sid)
        publication.track = None
        self.emit("track_unsubscribed", track, publication, participant)

    def mute(self, identity: str, publication: FakePublication, muted: bool) -> None:
        publication.muted = muted
        self.emit(
            "track_muted" if muted else "track_unmuted",
            self.remote_participants[identity],
            publication,
        )

    def quality(self, identity: str, quality: rtc.ConnectionQuality) -> None:
        self.emit(
            "connection_quality_changed", self.remote_participants[identity], quality
        )

    def participant_leaves(self, identity: str) -> None:
        self.emit("participant_disconnected", self.remote_participants.pop(identity))

    def reconnecting(self) -> None:
        """The relay's own connection to the room is getting back on its
        feet."""
        self.emit("reconnecting")

    def reconnected(self) -> None:
        self.emit("reconnected")

    def lost(self) -> None:
        """The relay's own connection to the room is gone."""
        self.emit("disconnected", rtc.DisconnectReason.SIGNAL_CLOSE)

    def say(self, frames: Sequence[bytes]) -> None:
        """The person speaks into their microphone in this room."""
        assert self.person is not None
        publication = self.microphone(self.person)
        for frame in frames:
            self.media.streams[publication.sid].push(frame)

    def microphone(self, identity: str) -> FakePublication:
        """*identity*'s latest microphone publication."""
        return [
            publication
            for publication in self.remote_participants[
                identity
            ].track_publications.values()
            if publication.source == rtc.TrackSource.SOURCE_MICROPHONE
        ][-1]


class FakeMedia:
    """The LiveKit SDK objects the relay makes, as fakes: every room it joins
    and every source and stream it opens."""

    def __init__(
        self,
        *,
        user: str | None = "user-1",
        refuse: Exception | None = None,
        subscribed_at_connect: bool = False,
    ) -> None:
        # Who is in each room the relay joins, with their microphone, under
        # the identity of that room's call; None for nobody.
        self.user = user
        # A join that is refused with this.
        self.refuse = refuse
        # The person's microphone already subscribed when the connect lands.
        self.subscribed_at_connect = subscribed_at_connect
        # While set and clear, each connect waits on it: a connect that lands
        # only once the test lets it.
        self.connect_gate: asyncio.Event | None = None
        self.rooms: list[FakeRoom] = []
        self.sources: list[FakeAudioSource] = []
        self.streams: dict[str, FakeAudioStream] = {}
        self._tracks = 0

    def track_sid(self) -> str:
        """A sid no other track in any of these rooms has."""
        self._tracks += 1
        return f"TR_{self._tracks}"

    def room(self) -> FakeRoom:
        room = FakeRoom(self)
        self.rooms.append(room)
        return room

    def source(self) -> FakeAudioSource:
        source = FakeAudioSource()
        self.sources.append(source)
        return source

    def track(self, source: FakeAudioSource) -> SimpleNamespace:
        return SimpleNamespace(source=source)

    def stream(self, track: FakeTrack) -> FakeAudioStream:
        stream = FakeAudioStream(track)
        self.streams[track.sid] = stream
        return stream

    def frame(self, pcm: bytes) -> bytes:
        return pcm

    def say(self, frames: Sequence[bytes]) -> None:
        """The person speaks into the latest room's microphone."""
        stream = list(self.streams.values())[-1]
        for frame in frames:
            stream.push(frame)

    @property
    def otto(self) -> bytes:
        """Everything Otto said in the latest room."""
        return self.sources[-1].audio if self.sources else b""

    async def hearing(self) -> FakeRoom:
        """The latest room, once the relay reads the person's microphone in
        it."""

        def heard() -> bool:
            room = self.rooms[-1] if self.rooms else None
            if room is None or room.person not in room.remote_participants:
                return False
            return room.microphone(room.person).sid in self.streams

        await until(heard)
        return self.rooms[-1]


def livekit_edge(media: FakeMedia, **options) -> LiveKitEdge:
    """The relay's LiveKit edge, on fake rooms."""
    return LiveKitEdge(LIVEKIT, media=media, **options)  # type: ignore[arg-type]


class ServiceRoom:
    def __init__(self, participants: set[str], creation_time: int) -> None:
        self.participants = participants
        self.creation_time = creation_time


class FakeRoomService:
    """LiveKit's room service for one project, in memory: its rooms, who is
    in each, and every request it took, in order."""

    def __init__(self) -> None:
        self.rooms: dict[str, ServiceRoom] = {}
        self.requests: list[tuple[str, object]] = []
        # A request of this kind fails with the exception, or never answers.
        self.fail: dict[str, BaseException] = {}
        self.hang: set[str] = set()
        # Told the kind of each request as it arrives.
        self.watch: Callable[[str], None] | None = None

    def open(self, name: str, *participants: str, created: int = 0) -> None:
        """A room open in the project, with *participants* in it."""
        self.rooms[name] = ServiceRoom(set(participants), created)

    def kinds(self) -> list[str]:
        return [kind for kind, _ in self.requests]

    async def _take(self, kind: str, request: object) -> None:
        self.requests.append((kind, request))
        if self.watch is not None:
            self.watch(kind)
        if kind in self.hang:
            await asyncio.Event().wait()
        failure = self.fail.get(kind)
        if failure is not None:
            raise failure

    async def remove_participant(
        self, remove: api.RoomParticipantIdentity
    ) -> api.RemoveParticipantResponse:
        await self._take("remove", remove)
        room = self.rooms.get(remove.room)
        if room is None or remove.identity not in room.participants:
            raise api.TwirpError("not_found", "participant does not exist", status=404)
        room.participants.discard(remove.identity)
        return api.RemoveParticipantResponse()

    async def delete_room(
        self, delete: api.DeleteRoomRequest
    ) -> api.DeleteRoomResponse:
        await self._take("delete", delete)
        if self.rooms.pop(delete.room, None) is None:
            raise api.TwirpError("not_found", "room not found", status=404)
        return api.DeleteRoomResponse()

    async def list_participants(
        self, request: api.ListParticipantsRequest
    ) -> api.ListParticipantsResponse:
        await self._take("participants", request)
        room = self.rooms.get(request.room)
        if room is None:
            raise api.TwirpError("not_found", "room not found", status=404)
        return api.ListParticipantsResponse(
            participants=[
                api.ParticipantInfo(identity=identity)
                for identity in sorted(room.participants)
            ]
        )

    async def list_rooms(self, request: api.ListRoomsRequest) -> api.ListRoomsResponse:
        await self._take("list", request)
        return api.ListRoomsResponse(
            rooms=[
                api.Room(
                    name=name,
                    num_participants=len(room.participants),
                    creation_time=room.creation_time,
                )
                for name, room in self.rooms.items()
            ]
        )


def room_reaper(redis, service: FakeRoomService, **options) -> RoomReaper:
    """The relay's room reaper, on a fake room service."""

    async def get_redis():
        return redis

    return RoomReaper(LIVEKIT, redis=get_redis, service=service, **options)


class MemoryRecordings:
    """Cloud storage for recordings, in memory."""

    def __init__(self, *, fail_stores: int = 0) -> None:
        self.objects: dict[str, bytes] = {}
        self.stored: list[str] = []
        self.deleted: list[str] = []
        # The next this-many stores fail, as an outage does.
        self.fail_stores = fail_stores

    async def store(self, user_id: str, wav: bytes, filename: str) -> str:
        if self.fail_stores:
            self.fail_stores -= 1
            raise ConnectionError("storage is down")
        path = f"gcs://{BUCKET}/uploads/users/{user_id}/{len(self.stored)}/{filename}"
        self.objects[path] = wav
        self.stored.append(path)
        return path

    async def fetch(self, path: str, user_id: str) -> bytes:
        if f"/users/{user_id}/" not in path:
            raise PermissionError("not this user's file")
        return self.objects[path]

    async def delete(self, path: str) -> None:
        self.objects.pop(path, None)
        self.deleted.append(path)


class ScriptedTranscriber:
    """Transcribes each turn with the next of *texts*; an exception in them is
    raised instead."""

    def __init__(self, texts: Sequence[str | Exception] = ()) -> None:
        self.texts = list(texts)
        self.calls: list[tuple[int, str | None]] = []
        # Each clip sent, as the WAV it was sent as.
        self.wavs: list[bytes] = []

    async def __call__(self, wav: bytes, *, first_name: str | None) -> Transcription:
        self.calls.append((len(wav), first_name))
        self.wavs.append(wav)
        text = self.texts.pop(0) if self.texts else "I run a bakery."
        if isinstance(text, Exception):
            raise text
        return Transcription(text=text, tokens_in=100, tokens_out=10)


class FakeSocket:
    """The browser's end of a call socket."""

    def __init__(self) -> None:
        self.inbox: asyncio.Queue[dict] = asyncio.Queue()
        self.sent: list[dict] = []
        # The type of everything sent, in order.
        self.log: list[str] = []
        self.closed_with: int | None = None
        self.broken = False
        self._changed = asyncio.Event()

    # The relay's side.

    async def receive(self) -> dict:
        return await self.inbox.get()

    async def send_text(self, data: str) -> None:
        self._check()
        message = json.loads(data)
        self.sent.append(message)
        self.log.append(message["type"])
        self._notify()

    async def close(self, code: int = 1000, reason: str | None = None) -> None:
        self.closed_with = code
        self.inbox.put_nowait({"type": "websocket.disconnect", "code": code})
        self._notify()

    # The browser's side.

    def send_json(self, message: dict) -> None:
        self.inbox.put_nowait(
            {"type": "websocket.receive", "text": json.dumps(message)}
        )

    def send_raw_text(self, text: str) -> None:
        self.inbox.put_nowait({"type": "websocket.receive", "text": text})

    def send_binary(self, frames: list[bytes]) -> None:
        for frame in frames:
            self.inbox.put_nowait({"type": "websocket.receive", "bytes": frame})

    def drop(self) -> None:
        """The network goes away under the socket."""
        self.broken = True
        self.inbox.put_nowait({"type": "websocket.disconnect", "code": 1006})

    def types(self) -> list[str]:
        return [message["type"] for message in self.sent]

    async def wait_for(
        self, kind: str, *, count: int = 1, timeout: float = 5.0
    ) -> dict:
        """The *count*-th message of type *kind* the relay sent."""

        async def seen() -> dict:
            while True:
                matching = [m for m in self.sent if m["type"] == kind]
                if len(matching) >= count:
                    return matching[count - 1]
                self._changed.clear()
                await self._changed.wait()

        return await asyncio.wait_for(seen(), timeout=timeout)

    def _check(self) -> None:
        if self.broken or self.closed_with is not None:
            raise RuntimeError("socket is closed")

    def _notify(self) -> None:
        self._changed.set()


class RunningRelay:
    def __init__(
        self,
        relay: Relay,
        server: DrainingServer,
        serving: asyncio.Task,
        port: int,
        *,
        media: FakeMedia,
        rows: list[Row],
        recorded: dict[str, str],
        postcall: PostCall,
        recordings: MemoryRecordings,
    ) -> None:
        self.relay = relay
        self.server = server
        self.serving = serving
        self.port = port
        # The calls' rooms, and what the person says and hears in them.
        self.media = media
        self.rows = rows
        # The calls recorded on onboarding rows: user id -> call id.
        self.recorded = recorded
        self.postcall = postcall
        self.recordings = recordings

    async def answer(self, call_id: str, frames: Sequence[bytes]) -> None:
        """The person on call *call_id* answers Otto once his turn has played
        out, as someone does on hearing him finish."""
        session = self.relay.sessions[call_id]

        def hears() -> bool:
            now = time.monotonic()
            return session.answer.is_open(now) and session.mic_open(now)

        await until(hears)
        (await self.media.hearing()).say(frames)

    async def postcalls_done(self, timeout: float = 10.0) -> None:
        """Wait for the post-call jobs running on the relay."""

        async def idle() -> None:
            while self.postcall.jobs:
                await asyncio.gather(
                    *self.postcall.jobs.values(), return_exceptions=True
                )

        await asyncio.wait_for(idle(), timeout=timeout)

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}/voice/call"

    @property
    def http(self) -> str:
        return f"http://127.0.0.1:{self.port}"


@asynccontextmanager
async def running_relay(
    redis,
    vendor: FakeVendor,
    *,
    max_calls: int = 5,
    drain_max_s: float = 30.0,
    settle_s: float = 5.0,
    caps: Caps = CAPS,
    timing: Timing = TIMING,
    record_failures: int = 0,
    record_delay_s: float = 0.0,
    max_concurrent: int = 25,
    transcriber: ScriptedTranscriber | None = None,
    redis_delay_s: float = 0.0,
    media: FakeMedia | None = None,
    reaper: RoomReaper | None = None,
) -> AsyncIterator[RunningRelay]:
    """The relay app under a real uvicorn server on a free local port. Its
    calls' rooms are *media*'s, and *reaper*, on a fake room service unless
    given, closes and sweeps them."""
    media = media or FakeMedia()
    rows: list[Row] = []
    recorded: dict[str, str] = {}
    recordings = MemoryRecordings()

    async def get_redis():
        if redis_delay_s:
            await asyncio.sleep(redis_delay_s)
        return redis

    async def account_name(user_id: str) -> str | None:
        return "Dana Smith"

    async def first_name(user_id: str) -> str | None:
        return "Dana"

    async def sink(row: Row) -> None:
        rows.append(row)

    failures = [record_failures]

    async def record_call(user_id: str, call_id: str) -> str | None:
        if record_delay_s:
            await asyncio.sleep(record_delay_s)
        if failures[0] > 0:
            failures[0] -= 1
            raise ConnectionError("database is down")
        replaced = recorded.get(user_id)
        recorded[user_id] = call_id
        return replaced

    spool = tempfile.TemporaryDirectory()
    postcall = PostCall(
        redis=get_redis,
        recordings=recordings,
        transcriber=transcriber or ScriptedTranscriber(),
        first_name=first_name,
        live_model="gemini-3.8-live",
        pod="relay-test",
        cost_sink=sink,
    )
    relay = Relay(
        max_calls=max_calls,
        origins=build_cors_params([ORIGIN], AppEnvironment.LOCAL),
        deps=RelayDeps(
            vendor=vendor,
            classifier=None,
            redis=get_redis,
            account_name=account_name,
            record_call=record_call,
            live_model="gemini-3.8-live",
            max_concurrent=max_concurrent,
            postcall=postcall,
            spool_dir=spool.name,
            cost_sink=sink,
            caps=caps,
            timing=timing,
            edge=livekit_edge(media),
            reaper=reaper or room_reaper(redis, FakeRoomService()),
        ),
    )
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    server = DrainingServer(
        uvicorn.Config(
            create_app(relay),
            ws="websockets-sansio",
            ws_max_size=WS_MAX_MESSAGE_BYTES,
            ws_per_message_deflate=False,
            log_config=None,
        ),
        relay=relay,
        drain_max_s=drain_max_s,
        settle_s=settle_s,
    )
    serving = asyncio.create_task(server.serve(sockets=[listener]))
    while not server.started:
        if serving.done():
            serving.result()
        await asyncio.sleep(0.01)
    running = RunningRelay(
        relay,
        server,
        serving,
        listener.getsockname()[1],
        media=media,
        rows=rows,
        recorded=recorded,
        postcall=postcall,
        recordings=recordings,
    )
    try:
        yield running
    finally:
        server.should_exit = True
        for session in list(relay.sessions.values()):
            session.end("server_restart")
        await asyncio.wait_for(serving, timeout=15)
        await asyncio.wait_for(
            asyncio.gather(*postcall.jobs.values(), return_exceptions=True),
            timeout=15,
        )
        listener.close()
        spool.cleanup()
