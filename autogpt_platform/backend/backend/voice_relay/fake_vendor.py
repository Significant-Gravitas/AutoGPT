"""A scripted stand-in for Gemini Live, for tests.

It speaks the relay's own vendor events, so a call runs unchanged against it
with no network and no bill. Every session it opens takes the script's next
reply as its answer: to its opening, to the person falling silent after
speaking, and to a server note sent with ``turn_complete`` (a goodbye). A
reply can call the end tool before it speaks, and reports its usage at the
end. A session can be told to drop partway through a reply, the way a vendor
socket dies with 1011.
"""

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Collection, Sequence
from dataclasses import dataclass, field

import numpy as np

from backend.voice_relay import vad
from backend.voice_relay.config import OUT_RATE
from backend.voice_relay.vendor import (
    AgentAudio,
    AgentText,
    FunctionCall,
    ToolCall,
    ToolResult,
    TurnComplete,
    Usage,
    UserText,
    VendorEvent,
)

END_TOOL = "finish_onboarding"
CHUNK_S = 0.04
TOOL_RESULT_WAIT_S = 2.0

REPLY_USAGE = Usage(
    prompt_tokens=900,
    response_tokens=100,
    thoughts_tokens=20,
    prompt_by_modality={"text": 700, "audio": 200},
    response_by_modality={"audio": 100},
)


@dataclass(frozen=True)
class Reply:
    """One turn of Otto's."""

    text: str = "Hi, I'm Otto, an AI."
    audio_s: float = 1.0
    # The caption of the person's turn this reply answers.
    heard: str = ""
    # When set, Otto calls the end tool with this first task before speaking.
    first_task: str | None = None
    usage: Usage = REPLY_USAGE


class _Drop:
    pass


@dataclass
class FakeVendor:
    """Opens scripted sessions; the script runs on across reconnects."""

    replies: Sequence[Reply] = ()
    # Sessions (counted from 1) that drop halfway through their first reply.
    drop_sessions: Collection[int] = ()
    # Seconds of silence after speech that end the person's turn.
    end_of_speech_s: float = 0.3
    # Seconds between audio chunks; 0 streams a reply as fast as it can.
    pace_s: float = 0.0
    connect_delay_s: float = 0.0
    sessions: list["FakeVendorSession"] = field(default_factory=list)
    _next: int = 0

    async def connect(self, *, opening: str) -> "FakeVendorSession":
        if self.connect_delay_s:
            await asyncio.sleep(self.connect_delay_s)
        session = FakeVendorSession(
            self, number=len(self.sessions) + 1, opening=opening
        )
        self.sessions.append(session)
        session.start()
        return session

    def next_reply(self) -> Reply | None:
        if self._next >= len(self.replies):
            return None
        reply = self.replies[self._next]
        self._next += 1
        return reply

    @property
    def openings(self) -> list[str]:
        return [session.opening for session in self.sessions]


class FakeVendorSession:
    def __init__(self, vendor: FakeVendor, *, number: int, opening: str) -> None:
        self.vendor = vendor
        self.number = number
        self.opening = opening
        self.notes: list[tuple[str, bool]] = []
        self.tool_results: list[ToolResult] = []
        self.audio_in_bytes = 0
        self.stream_ends = 0
        self.closed = False
        self._queue: asyncio.Queue[VendorEvent | _Drop | None] = asyncio.Queue()
        self._speaking = False
        self._last_speech = 0.0
        self._tool_answered = asyncio.Event()
        self._replies_spoken = 0
        # Otto finishes one reply before starting the next.
        self._speaking_lock = asyncio.Lock()
        self._tasks: set[asyncio.Task] = set()

    def start(self) -> None:
        self._spawn(self._answer())
        self._spawn(self._listen())

    def emit(self, *events: VendorEvent) -> None:
        """Put events on the stream as if the vendor had sent them."""
        for event in events:
            self._queue.put_nowait(event)

    def drop(self) -> None:
        """Kill the stream now, as a vendor socket error does."""
        self._queue.put_nowait(_Drop())

    async def send_audio(self, pcm: bytes) -> None:
        self._check_open()
        self.audio_in_bytes += len(pcm)
        if vad.heard(pcm):
            self._speaking = True
            self._last_speech = time.monotonic()

    async def end_audio_stream(self) -> None:
        self._check_open()
        self.stream_ends += 1

    async def send_note(self, text: str, *, turn_complete: bool) -> None:
        self._check_open()
        self.notes.append((text, turn_complete))
        if turn_complete:
            self._spawn(self._answer())

    async def send_tool_results(self, results: Sequence[ToolResult]) -> None:
        self._check_open()
        self.tool_results.extend(results)
        self._tool_answered.set()

    async def events(self) -> AsyncIterator[VendorEvent]:
        while True:
            event = await self._queue.get()
            if event is None:
                return
            if isinstance(event, _Drop):
                raise ConnectionError("fake vendor dropped the session")
            yield event

    async def close(self) -> None:
        self.closed = True
        for task in list(self._tasks):
            task.cancel()
        self._queue.put_nowait(None)

    def _check_open(self) -> None:
        if self.closed:
            raise ConnectionError("fake vendor session is closed")

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _listen(self) -> None:
        while not self.closed:
            await asyncio.sleep(0.02)
            silent_for = time.monotonic() - self._last_speech
            if self._speaking and silent_for > self.vendor.end_of_speech_s:
                self._speaking = False
                self._spawn(self._answer(after_speech=True))

    async def _answer(self, *, after_speech: bool = False) -> None:
        async with self._speaking_lock:
            await self._speak(after_speech=after_speech)

    async def _speak(self, *, after_speech: bool) -> None:
        reply = self.vendor.next_reply()
        if reply is None:
            return
        if after_speech and reply.heard:
            self.emit(UserText(reply.heard))
        if reply.first_task is not None:
            self._tool_answered.clear()
            self.emit(
                ToolCall(
                    calls=(
                        FunctionCall(
                            call_id=f"call-{self.number}-{self._replies_spoken}",
                            name=END_TOOL,
                            args={"first_task": reply.first_task},
                        ),
                    )
                )
            )
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(
                    self._tool_answered.wait(), timeout=TOOL_RESULT_WAIT_S
                )
        drops = self.number in self.vendor.drop_sessions and self._replies_spoken == 0
        self._replies_spoken += 1
        chunks = max(1, round(reply.audio_s / CHUNK_S))
        self.emit(AgentText(reply.text))
        for index in range(chunks):
            if drops and index >= chunks // 2:
                self.drop()
                return
            self.emit(AgentAudio(_tone(CHUNK_S)))
            await asyncio.sleep(self.vendor.pace_s)
        self.emit(reply.usage, TurnComplete())


def _tone(seconds: float) -> bytes:
    samples = np.arange(int(OUT_RATE * seconds))
    wave = np.sin(samples * 2 * np.pi * 220 / OUT_RATE) * 3000
    return wave.astype(np.int16).tobytes()
