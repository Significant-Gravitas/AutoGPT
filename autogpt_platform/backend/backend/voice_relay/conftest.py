import asyncio
import json
import socket
import tempfile
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from types import SimpleNamespace

import jwt
import numpy as np
import pytest
import uvicorn
from fakeredis import FakeServer
from fakeredis.aioredis import FakeRedis
from livekit import rtc

from backend.api.utils.cors import build_cors_params
from backend.util.settings import AppEnvironment
from backend.voice_relay import tokens
from backend.voice_relay.app import Relay, RelayDeps, create_app
from backend.voice_relay.config import (
    CAPS,
    IN_RATE,
    TIMING,
    WS_MAX_MESSAGE_BYTES,
    Caps,
    Timing,
)
from backend.voice_relay.edge import LiveKitEdge
from backend.voice_relay.fake_vendor import FakeVendor
from backend.voice_relay.metering import Row
from backend.voice_relay.postcall import PostCall, Transcription
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


class FakeParticipant:
    def __init__(self, identity: str) -> None:
        self.identity = identity
        self.track_publications: dict[str, SimpleNamespace] = {}


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
    subscribed, when the relay connects."""

    def __init__(self, media: "FakeMedia") -> None:
        self.media = media
        self.handlers: dict[str, list[Callable]] = {}
        self.remote_participants: dict[str, FakeParticipant] = {}
        self.local_participant = SimpleNamespace(
            published=[], publish_track=self._publish
        )
        self.url: str | None = None
        self.token: str | None = None
        self.left = False
        self._tracks = 0

    def on(self, event: str, callback: Callable) -> Callable:
        self.handlers.setdefault(event, []).append(callback)
        return callback

    def emit(self, event: str, *args: object) -> None:
        for callback in list(self.handlers.get(event, [])):
            callback(*args)

    async def connect(self, url: str, token: str, options: object = None) -> None:
        self.url, self.token = url, token
        if self.media.hang:
            await asyncio.Event().wait()
        if self.media.refuse is not None:
            raise self.media.refuse
        if self.media.person is not None:
            self.add_microphone(self.media.person, announce=False)

    async def disconnect(self) -> None:
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

    def add_microphone(
        self, identity: str, *, announce: bool = True, kind: int | None = None
    ) -> FakeTrack:
        """*identity* publishes a microphone, which the relay is subscribed
        to; *announce* as the SDK does for a track that comes after the
        join."""
        self._tracks += 1
        track = FakeTrack(f"TR_{self._tracks}", kind or rtc.TrackKind.KIND_AUDIO)
        participant = self.remote_participants.setdefault(
            identity, FakeParticipant(identity)
        )
        publication = SimpleNamespace(track=track)
        participant.track_publications[track.sid] = publication
        if announce:
            self.emit("track_subscribed", track, publication, participant)
        return track

    def drop_microphone(self, identity: str, track: FakeTrack) -> None:
        participant = self.remote_participants[identity]
        publication = participant.track_publications.pop(track.sid)
        self.emit("track_unsubscribed", track, publication, participant)

    def participant_leaves(self, identity: str) -> None:
        self.emit("participant_disconnected", self.remote_participants.pop(identity))

    def lost(self) -> None:
        """The relay's own connection to the room is gone."""
        self.emit("disconnected", rtc.DisconnectReason.SIGNAL_CLOSE)


class FakeMedia:
    """The LiveKit SDK objects the relay makes, as fakes: every room it joins
    and every source and stream it opens."""

    def __init__(
        self,
        *,
        person: str | None = "user:user-1",
        refuse: Exception | None = None,
        hang: bool = False,
    ) -> None:
        # Who is in each room the relay joins, with their microphone.
        self.person = person
        # A join that is refused with this, or never answers.
        self.refuse = refuse
        self.hang = hang
        self.rooms: list[FakeRoom] = []
        self.sources: list[FakeAudioSource] = []
        self.streams: dict[str, FakeAudioStream] = {}

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


def livekit_edge(media: FakeMedia, **options) -> LiveKitEdge:
    """The relay's LiveKit edge, on fake rooms."""
    return LiveKitEdge(LIVEKIT, media=media, **options)  # type: ignore[arg-type]


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

    async def __call__(self, wav: bytes, *, first_name: str | None) -> Transcription:
        self.calls.append((len(wav), first_name))
        text = self.texts.pop(0) if self.texts else "I run a bakery."
        if isinstance(text, Exception):
            raise text
        return Transcription(text=text, tokens_in=100, tokens_out=10)


class FakeSocket:
    """The browser's end of a call socket."""

    def __init__(self) -> None:
        self.inbox: asyncio.Queue[dict] = asyncio.Queue()
        self.sent: list[dict] = []
        self.audio: list[bytes] = []
        # Everything sent, in order: ("text", its type) or ("bytes", its size).
        self.log: list[tuple[str, object]] = []
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
        self.log.append(("text", message["type"]))
        self._notify()

    async def send_bytes(self, data: bytes) -> None:
        self._check()
        self.audio.append(data)
        self.log.append(("bytes", len(data)))
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

    def send_audio(self, frames: list[bytes]) -> None:
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
        rows: list[Row],
        recorded: dict[str, str],
        postcall: PostCall,
        recordings: MemoryRecordings,
    ) -> None:
        self.relay = relay
        self.server = server
        self.serving = serving
        self.port = port
        self.rows = rows
        # The calls recorded on onboarding rows: user id -> call id.
        self.recorded = recorded
        self.postcall = postcall
        self.recordings = recordings

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
    edge: LiveKitEdge | None = None,
) -> AsyncIterator[RunningRelay]:
    """The relay app under a real uvicorn server on a free local port; with
    *edge*, it joins the rooms of calls on the LiveKit leg."""
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
            edge=edge,
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
