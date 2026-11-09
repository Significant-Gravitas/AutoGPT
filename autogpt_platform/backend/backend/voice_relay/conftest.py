import asyncio
import json
import socket
import tempfile
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager

import numpy as np
import pytest
import uvicorn
from fakeredis import FakeServer
from fakeredis.aioredis import FakeRedis
from fastapi import BackgroundTasks
from prisma.enums import BrainDumpInputMode, BrainDumpStatus

from backend.api.features.onboarding_dump.interview import InterviewClosure
from backend.api.features.onboarding_dump.models import FinalizeResponse
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
from backend.voice_relay.fake_vendor import FakeVendor
from backend.voice_relay.handoff import record_hand_off
from backend.voice_relay.locks import CallRecord
from backend.voice_relay.metering import Row
from backend.voice_relay.postcall import PostCall, Transcription
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


class HandOffs:
    """Hands every call off as an accepted take and remembers each."""

    def __init__(self) -> None:
        self.calls: list[CallRecord] = []

    async def __call__(
        self, redis, record: CallRecord, background: BackgroundTasks
    ) -> InterviewClosure:
        self.calls.append(record)
        closure = InterviewClosure(
            ending=None,
            hand_off=FinalizeResponse(
                status=BrainDumpStatus.transcribed,
                input_mode=BrainDumpInputMode.typed,
            ),
        )
        await record_hand_off(redis, record.call_id, closure)
        return closure


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
        hand_offs: HandOffs,
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
        self.hand_offs = hand_offs

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
) -> AsyncIterator[RunningRelay]:
    """The relay app under a real uvicorn server on a free local port."""
    rows: list[Row] = []
    recorded: dict[str, str] = {}
    recordings = MemoryRecordings()
    hand_offs = HandOffs()

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

    async def record_call(user_id: str, call_id: str) -> None:
        if record_delay_s:
            await asyncio.sleep(record_delay_s)
        if failures[0] > 0:
            failures[0] -= 1
            raise ConnectionError("database is down")
        recorded[user_id] = call_id

    spool = tempfile.TemporaryDirectory()
    postcall = PostCall(
        redis=get_redis,
        recordings=recordings,
        transcriber=transcriber or ScriptedTranscriber(),
        first_name=first_name,
        live_model="gemini-3.8-live",
        pod="relay-test",
        cost_sink=sink,
        hand_off=hand_offs,
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
        hand_offs=hand_offs,
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
