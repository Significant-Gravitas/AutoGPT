"""A scripted stand-in for Gemini Live, for tests.

It speaks the relay's own vendor events, so a call runs unchanged against it
with no network and no bill. A session it opens says nothing until it is
sent a note, as a live one does. Every session takes the script's next reply
as its answer: to a server note sent with ``turn_complete`` (the call's
opening, a reconnect's, a goodbye), and to the person falling silent after
speaking. A
reply can call the end tool before it speaks, and reports its usage at the
end. A session can be told to drop partway through a reply, the way a vendor
socket dies with 1011.
"""

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Collection, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

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
from backend.voice_relay.waits import bounded

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


class Reply(BaseModel):
    """One turn of Otto's."""

    model_config = ConfigDict(frozen=True)

    text: str = "Hi, I'm Otto, an AI."
    audio_s: float = 1.0
    # The caption of the person's turn this reply answers.
    heard: str = ""
    # When set, Otto calls the end tool with this first task before speaking.
    first_task: str | None = None
    usage: Usage = REPLY_USAGE


class _Drop:
    pass


class FakeVendor:
    """Opens scripted sessions; the script runs on across reconnects."""

    def __init__(
        self,
        replies: Sequence[Reply] = (),
        *,
        drop_sessions: Collection[int] = (),
        end_of_speech_s: float = 0.3,
        pace_s: float = 0.0,
        connect_delay_s: float = 0.0,
        stall_notes: bool = False,
        stall_opening: bool = False,
        refuse: Exception | None = None,
        loses_a_cancellation: bool = False,
    ) -> None:
        self.replies = list(replies)
        # Sessions (counted from 1) that drop halfway through their first reply.
        self.drop_sessions = set(drop_sessions)
        # Seconds of silence after speech that end the person's turn.
        self.end_of_speech_s = end_of_speech_s
        # Seconds between audio chunks; 0 streams a reply as fast as it can.
        self.pace_s = pace_s
        self.connect_delay_s = connect_delay_s
        # Sending a server note after a session's opening never finishes, the
        # way a send to a vendor socket that has stopped reading never does.
        self.stall_notes = stall_notes
        # Sending the first session its first note, the call's opening, never
        # finishes, while the session itself stays up.
        self.stall_opening = stall_opening
        # Each connect is refused with this.
        self.refuse = refuse
        # The task reading a session's events loses its first cancellation
        # there and waits on, as one can inside a library's
        # ``asyncio.wait_for`` on Python 3.11.
        self.loses_a_cancellation = loses_a_cancellation
        # While set and clear, each connect waits on it: a connect that lands
        # only once the test lets it.
        self.connect_gate: asyncio.Event | None = None
        self.sessions: list[FakeVendorSession] = []
        self._next = 0

    async def connect(self) -> "FakeVendorSession":
        if self.connect_delay_s:
            await asyncio.sleep(self.connect_delay_s)
        if self.connect_gate is not None:
            await self.connect_gate.wait()
        if self.refuse is not None:
            raise self.refuse
        session = FakeVendorSession(self, number=len(self.sessions) + 1)
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
    def __init__(self, vendor: FakeVendor, *, number: int) -> None:
        self.vendor = vendor
        self.number = number
        self.notes: list[tuple[str, bool]] = []
        self.tool_results: list[ToolResult] = []
        self.audio_in_bytes = 0
        # The audio sent to it that the voice detector calls speech.
        self.speech_in_bytes = 0
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
        # Whether the task reading its events lost a cancellation there.
        self.lost_a_cancellation = False

    @property
    def opening(self) -> str:
        """The first note the session was sent: the call's opening, or a
        reconnect's; empty before one."""
        return self.notes[0][0] if self.notes else ""

    @property
    def notes_after_opening(self) -> list[tuple[str, bool]]:
        """The server's notes since the session's opening."""
        return self.notes[1:]

    def start(self) -> None:
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
            self.speech_in_bytes += len(pcm)
            self._speaking = True
            self._last_speech = time.monotonic()

    async def end_audio_stream(self) -> None:
        self._check_open()
        self.stream_ends += 1

    async def send_note(self, text: str, *, turn_complete: bool) -> None:
        self._check_open()
        self.notes.append((text, turn_complete))
        if self.vendor.stall_notes and len(self.notes) > 1:
            await asyncio.Event().wait()
        if self.vendor.stall_opening and self.number == 1 and len(self.notes) == 1:
            await asyncio.Event().wait()
        if turn_complete:
            self._spawn(self._answer())

    async def send_tool_results(self, results: Sequence[ToolResult]) -> None:
        self._check_open()
        self.tool_results.extend(results)
        self._tool_answered.set()

    async def events(self) -> AsyncIterator[VendorEvent]:
        while True:
            try:
                event = await self._queue.get()
            except asyncio.CancelledError:
                if not self.vendor.loses_a_cancellation or self.lost_a_cancellation:
                    raise
                self.lost_a_cancellation = True
                continue
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
            self.emit(UserText(text=reply.heard))
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
                await bounded(self._tool_answered.wait(), timeout=TOOL_RESULT_WAIT_S)
        drops = self.number in self.vendor.drop_sessions and self._replies_spoken == 0
        self._replies_spoken += 1
        chunks = max(1, round(reply.audio_s / CHUNK_S))
        self.emit(AgentText(text=reply.text))
        for index in range(chunks):
            if drops and index >= chunks // 2:
                self.drop()
                return
            self.emit(AgentAudio(pcm=_tone(CHUNK_S)))
            await asyncio.sleep(self.vendor.pace_s)
        self.emit(reply.usage, TurnComplete())


def _tone(seconds: float) -> bytes:
    samples = np.arange(int(OUT_RATE * seconds))
    wave = np.sin(samples * 2 * np.pi * 220 / OUT_RATE) * 3000
    return wave.astype(np.int16).tobytes()
