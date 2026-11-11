"""One voice interview call: the browser's socket on one side, the vendor's
live session on the other.

The relay forwards audio both ways and owns every control of the call; the
browser only displays them:

- the wall clock and the turn cap, each with a nudge to sum up before it;
- the idle cutoff: nothing heard for ``idle_s``, Otto's own speech excluded;
- the scope guard, from a label per user turn and the interview's rule;
- the spend cap, from every priced usage row;
- the end tool: once Otto calls it, the call ends after its goodbye;
- re-attach: a dropped browser socket comes back on its resume ticket within
  ``reattach_s``, and what the call said on it meanwhile is replayed;
- vendor reconnect: a dropped vendor session is replaced by one seeded with
  the conversation so far, at most ``max_vendor_reconnects`` times;
- an end request from the person's other device, and the drain on deploys.

It also records the person's audio (``record``) and marks on the call's
record where each of Otto's turns ended in it, which is what the post-call
job cuts the recording into turns by. The call is pending a post-call job
from its start, and the call claims that job before it stops counting as
live (``postcall``).

The call's audio travels through its LiveKit room (``edge``), which the
relay joins while the vendor connects; the control socket carries the call's
events, its captions and the resume ticket, and no audio. A call whose room
the relay cannot join, or that loses it, ends ``room_error``: before
``ready``, the browser reads that as a call that never started. The call
counts against the person's allowance once it has both its room and its
model session, the first moment it costs anything.

Browser to relay, after ``start``: JSON ``hangup`` and ``ping {id}``; a
binary frame is counted and dropped. Relay to browser: JSON ``ready``,
``user_text``, ``agent_text``, ``interrupted``, ``turn_complete``,
``wrap_up``, ``resumed``, ``ended`` and ``pong``. The browser has no way to
put text into the conversation: the server's notes are the server's alone.
"""

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from backend.copilot.onboarding_interview import MAX_FIRST_TASK_CHARS
from backend.copilot.onboarding_interview_goals import END_SIGNAL_FIRST_TASK
from backend.copilot.onboarding_interview_scope import ScopeLabel, scope_verdict
from backend.copilot.onboarding_interview_voice import (
    VOICE_INTERVIEW_END_TOOL,
    voice_interview_kickoff,
)
from backend.voice_relay import tokens, vad
from backend.voice_relay.config import (
    BURST_S,
    CAPS,
    IN_RATE,
    LIVE_CALL_REFRESH_S,
    MAX_FRAME_BYTES,
    MAX_TEXT_BYTES,
    OUT_RATE,
    RELAY_ROOM_GRACE_S,
    RESUME_GRACE_S,
    TIMING,
    Caps,
    RoomCloseWhy,
    Timing,
)
from backend.voice_relay.locks import (
    CONNECTING,
    ENDED,
    LIVE,
    RUNNING,
    CallStore,
    TurnMark,
)
from backend.voice_relay.metering import (
    CALL_SECONDS,
    CALLS,
    FIRST_AUDIO_SECONDS,
    REPLY_SECONDS,
    VENDOR_FAILURES,
    CallMeter,
)
from backend.voice_relay.record import CallRecording
from backend.voice_relay.scope import ScopeClassifier
from backend.voice_relay.vendor import (
    AgentAudio,
    AgentText,
    GoingAway,
    Interrupted,
    ToolCall,
    ToolResult,
    TurnComplete,
    Usage,
    UserText,
    VendorConnector,
    VendorEvent,
    VendorSession,
)
from backend.voice_relay.vendor_failure import Stage as VendorStage
from backend.voice_relay.vendor_failure import VendorFailure, vendor_failure

logger = logging.getLogger(__name__)

EndReason = Literal[
    # Otto called the end tool and its goodbye played.
    "finished",
    "wall_clock",
    "turn_cap",
    "idle",
    "scope",
    "spend_cap",
    "user_hangup",
    # Ended from the person's other device.
    "superseded",
    # The browser's socket dropped and did not come back.
    "client_gone",
    "vendor_error",
    # The relay could not join the call's room, or lost it.
    "room_error",
    # The pod's drain ran past its limit.
    "server_restart",
]

SERVER_NOTE = "[Server note, not said by the user] "
NUDGES = {
    "time": SERVER_NOTE
    + "About thirty seconds are left. In your next turn, sum up what you heard "
    "in one or two sentences and ask if you got it right. Ask nothing new.",
    "turns": SERVER_NOTE
    + "Only two turns are left. In your next turn, sum up what you heard in one "
    "or two sentences and ask if you got it right. Ask nothing new.",
}
GOODBYES: dict[EndReason, str] = {
    "scope": SERVER_NOTE
    + "End the call now: the user keeps asking for things outside this call. In "
    "one short, friendly sentence, say you'll stop here and they can finish "
    "setting up by typing. Ask nothing.",
}

# One send to the vendor: past it the send is given up, and a session that
# has died is replaced by the reconnect.
VENDOR_SEND_TIMEOUT_S = 2.0

# Close codes: a normal end, and a server going away for a restart.
CLOSE_NORMAL = 1000
CLOSE_SERVICE_RESTART = 1012
# The browser's old socket, replaced by a re-attach.
CLOSE_REPLACED = 4000

# How a failed Live connect or session reads in the log.
_VENDOR_STAGES: dict[VendorStage, str] = {
    "connect": "did not connect",
    "reconnect": "did not reconnect",
    "session": "session dropped",
}


class ClientSocket(Protocol):
    """The browser's socket, as the call uses it (Starlette's ``WebSocket``)."""

    async def receive(self) -> Mapping[str, Any]: ...

    async def send_text(self, data: str) -> None: ...

    async def close(self, code: int = 1000, reason: str | None = None) -> None: ...


class AudioLeg(Protocol):
    """The call's audio in its room (``edge.RoomLeg``).

    The person's audio reaches the call through ``Session.on_mic``; this is
    the rest of what the call does with its audio.
    """

    async def play(self, pcm: bytes) -> None:
        """Pass on Otto's speech, 24 kHz PCM16, as the vendor sends it."""
        ...

    def flush(self) -> None:
        """Drop what Otto has not said yet: the person talked over it."""
        ...

    async def close(self, *, play_out: bool) -> None:
        """Stop carrying the call's audio, once what Otto said has played
        when *play_out*."""
        ...


# What a room hands the call: the person's audio, whether their microphone
# is in the room, whether their audio has stalled while it stays there, and
# that the relay's own connection to the room is gone.
Hear = Callable[[bytes], Awaitable[None]]
Presence = Callable[[bool], None]
Stall = Callable[[bool], None]
Lost = Callable[[], None]


@runtime_checkable
class RoomEdge(Protocol):
    """Joins a call's LiveKit room for the relay (``edge.LiveKitEdge``)."""

    @property
    def rooms(self) -> int:
        """Rooms joined, or being joined, and not yet left."""
        ...

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
    ) -> AudioLeg: ...


@runtime_checkable
class RoomCloser(Protocol):
    """Closes a call's LiveKit room for good (``room_reaper.RoomReaper``)."""

    async def close(self, call_id: str, user_id: str, why: RoomCloseWhy) -> bool:
        """True once the room is closed; a close that fails is left to the
        sweep."""
        ...


# Ends after which nobody is left to hear Otto's last words, or the server
# must stop now: the call's room is left at once.
_LEAVE_AT_ONCE: frozenset[str] = frozenset(
    {"client_gone", "superseded", "server_restart", "room_error"}
)


class _StartFailed(Exception):
    """The call could not open: its room, or its model session."""

    def __init__(self, reason: EndReason, error: str) -> None:
        super().__init__(error)
        self.reason: EndReason = reason
        self.error = error


class AgentTurn(BaseModel):
    index: int = 0
    text: str = ""
    audio_s: float = 0.0
    # When its first audio arrived (monotonic), or None while silent.
    first_audio: float | None = None
    interrupted: bool = False


class UserTurn(BaseModel):
    model_config = ConfigDict(frozen=True)

    # The agent turn this answers; -1 for words said before Otto's first.
    after: int
    text: str


class Session:
    def __init__(
        self,
        *,
        call_id: str,
        user_id: str,
        ws: ClientSocket,
        vendor: VendorConnector,
        store: CallStore,
        meter: CallMeter,
        classifier: ScopeClassifier | None,
        edge: RoomEdge,
        first_name: str | None = None,
        caps: Caps = CAPS,
        timing: Timing = TIMING,
        started: float | None = None,
        on_end: Callable[[], None] | None = None,
        recording: CallRecording | None = None,
        reaper: RoomCloser | None = None,
    ) -> None:
        self.call_id = call_id
        self.user_id = user_id
        self.edge = edge
        # The REST API registered the call's room before it minted its token,
        # and the call closes it (``reaper``) once it is done with it.
        self.reaper = reaper
        self._room_closed = False
        # The call's audio in its room, once the relay has joined it.
        self.audio: AudioLeg | None = None
        self.ws = ws
        self.vendor = vendor
        self.store = store
        self.meter = meter
        self.classifier = classifier
        self.first_name = first_name
        self.caps = caps
        self.timing = timing
        # The person's audio, and where each of Otto's turns ended in it.
        self.recording = recording
        self.turn_marks: list[TurnMark] = []
        # Whether this process holds the call's post-call job.
        self.postcall_claimed = False
        # When the start message arrived (monotonic): the first audio is
        # measured from here.
        self.started = time.monotonic() if started is None else started
        # Called once the call's shared state is released, before the relay
        # waits on the browser's socket again.
        self.on_end = on_end
        self.ended = asyncio.Event()
        self.closed = asyncio.Event()
        self.end_reason: EndReason | None = None
        self.error: str | None = None
        self.tasks: set[asyncio.Task] = set()
        self.t0: float | None = None
        self.first_audio_s: float | None = None
        # The browser side.
        self._send_lock = asyncio.Lock()
        # Held through a re-attach, and by the call's end while it tells the
        # browser, so the end reaches the socket a re-attach brings.
        self._attach_lock = asyncio.Lock()
        self._socket_done = asyncio.Event()
        self.upstream_task: asyncio.Task | None = None
        self.client_gone = False
        self.detached_at: float | None = None
        self.outbox: list[str] = []
        self.outbox_bytes = 0
        self.reattaches = 0
        # The vendor side.
        self.live: VendorSession | None = None
        self.reconnecting = False
        self.vendor_reconnects = 0
        # Whether the running session sent its closing notice (GoAway): a
        # drop after it is the reset the vendor warned of.
        self.vendor_notice = False
        # Audio, in samples.
        self.received = self.forwarded = self.dropped = self.agent_out = 0
        self.bad_frames = 0
        self.bucket = BURST_S * IN_RATE
        self.bucket_at = time.monotonic()
        # The idle cutoff counts quiet time while Otto's audio is not playing,
        # which it is estimated to do until playout_end.
        self.playout_end = 0.0
        self.quiet_s = 0.0
        # When the voice detector last heard the person (monotonic), until
        # Otto's next turn answers it: the reply's latency runs from there.
        self.last_heard: float | None = None
        # Since when the person's microphone has been out of the room, or
        # None while it is in, and whether its audio has stalled while it
        # stays there (muted, or a connection reconnecting). The room's audio
        # reaches the call once the call is live.
        self.away_since: float | None = None
        self.stalled = False
        self.hearing_room = False
        # The conversation.
        self.turns: list[AgentTurn] = []
        self.cur = AgentTurn()
        self.generating = False
        self.user_buf = ""
        self.user_turns: list[UserTurn] = []
        # The controls.
        self.labels: list[ScopeLabel] = []
        self.scope_queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        self.wrapping: EndReason | None = None
        self.wrap_started = 0.0
        self.wrap_deadline = 0.0
        self.finish_called = False
        self.finish_at = 0.0
        self.first_task: str | None = None
        self.nudged: set[str] = set()

    # --- lifecycle --------------------------------------------------------

    def elapsed(self) -> float:
        return time.monotonic() - self.t0 if self.t0 is not None else 0.0

    def end(self, reason: EndReason) -> None:
        if self.end_reason is None:
            self.end_reason = reason
            logger.info(
                f"Voice call {self.call_id} ending: {reason} after "
                f"{len(self.turns)} turns"
            )
        self.ended.set()

    def spawn(self, coro: Coroutine[Any, Any, object], name: str) -> asyncio.Task:
        task = asyncio.create_task(self._guard(coro, name), name=name)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    async def _guard(self, coro: Coroutine[Any, Any, object], name: str) -> None:
        try:
            await coro
        except Exception as error:
            if not self.ended.is_set():
                logger.exception(f"Voice call {self.call_id}: {name} failed")
                self.error = f"{name}: {type(error).__name__}"
                self.end("vendor_error")

    async def run(self) -> None:
        """Run the call to its end, then report how it ended."""
        await self.store.pend_postcall()
        # Held from here, so no other process's sweep takes the job of a call
        # still running; the call's refresh holds it on.
        self.postcall_claimed = await self.store.claim_postcall()
        try:
            live = await self._open()
            self.live = live
            if self.ended.is_set():
                # The room was lost while the call opened.
                return
            self.t0 = time.monotonic()
            self.bucket_at = self.t0
            # The call costs us from here.
            await self.store.count_call()
            await self.store.advance(
                LIVE,
                moves_from=(None, CONNECTING),
                connect_s=round(self.t0 - self.started, 3),
            )
            ticket = tokens.mint_resume(
                self.user_id, self.call_id, self.caps.max_call_s + RESUME_GRACE_S
            )
            await self.send_json(
                {
                    "type": "ready",
                    "call_id": self.call_id,
                    "caps": self.caps.as_message(),
                    "resume": ticket,
                }
            )
            self.hearing_room = True
            self.spawn(self.downstream(live), "downstream")
            self.upstream_task = self.spawn(self.upstream(self.ws), "upstream")
            self.spawn(self.watchdog(), "watchdog")
            if self.classifier is not None:
                self.spawn(self.scope_worker(self.classifier), "scope")
            await self.ended.wait()
        except asyncio.CancelledError:
            # The server is shutting down under the call.
            self.end("server_restart")
            raise
        except _StartFailed as failed:
            self.error = failed.error
            self.end(failed.reason)
        finally:
            for task in list(self.tasks):
                task.cancel()
            await asyncio.gather(*self.tasks, return_exceptions=True)
            await self.close_vendor()
            await self.finish()

    async def _open(self) -> VendorSession:
        """The call's model session, with its room joined: the room is joined
        while the vendor connects, so the audio has its way by Otto's first
        words.

        Raises ``_StartFailed`` naming the side that failed. A model session
        opened for a room the relay could not join is closed.
        """
        joining = asyncio.create_task(self.join_room(), name="room_join")
        self.tasks.add(joining)
        joining.add_done_callback(self.tasks.discard)
        try:
            live = await asyncio.wait_for(
                self.vendor.connect(opening=voice_interview_kickoff(self.first_name)),
                timeout=self.timing.vendor_connect_timeout_s,
            )
        except Exception as error:
            failure = self._vendor_failed("connect", error)
            raise _StartFailed(
                "vendor_error", f"connect: {failure.summary()}"
            ) from error
        try:
            await joining
        except Exception as error:
            await live.close()
            logger.warning(
                f"Voice call {self.call_id}: the relay could not join its room "
                f"({type(error).__name__})"
            )
            raise _StartFailed("room_error", f"room: {type(error).__name__}") from error
        except asyncio.CancelledError:
            await live.close()
            raise
        return live

    async def finish(self) -> None:
        reason: EndReason = self.end_reason or "vendor_error"
        elapsed = self.elapsed()
        try:
            await self.meter.unreported(
                sent_samples=self.forwarded,
                received_samples=self.agent_out,
                cut_off_reply=self.generating,
                why="listening after the last usage report",
            )
            await self.meter.flush()
            # Before the call stops counting as live, so no other process's
            # sweep takes the job from under this one, when the claim at the
            # start did not go through.
            self.postcall_claimed = await self.store.claim_postcall()
            # A post-call job that took the call meanwhile keeps the status it
            # wrote, and the end reason it ran on.
            await self.store.advance(
                ENDED,
                moves_from=(None, *RUNNING),
                guarded={"end_reason": reason},
                ended_at=round(time.time(), 3),
                elapsed_s=round(elapsed, 1),
                agent_turns=len(self.turns),
                usd=round(self.meter.usd, 6),
                first_task=self.first_task,
                error=self.error,
                reattaches=self.reattaches,
                vendor_reconnects=self.vendor_reconnects,
                transcript=self.transcript(),
            )
            # Before the browser hears the call ended, so it can start another.
            await self.store.release()
        finally:
            if self.on_end is not None:
                self.on_end()
        CALLS.labels(reason=reason).inc()
        CALL_SECONDS.observe(elapsed)
        try:
            async with self._attach_lock:
                if self.detached_at is not None:
                    self.client_gone = True
                await self.send_json(
                    {"type": "ended", "reason": reason, "elapsed_s": round(elapsed, 1)}
                )
                if not self.client_gone:
                    code = (
                        CLOSE_SERVICE_RESTART
                        if reason == "server_restart"
                        else CLOSE_NORMAL
                    )
                    await self._close(self.ws, code)
        finally:
            try:
                await self.leave_room(reason)
            finally:
                self.closed.set()
                self._socket_done.set()

    async def leave_room(self, reason: EndReason) -> None:
        """Stop carrying the call's audio: Otto's last words play out in the
        room first, unless nobody is left to hear them or the server must
        stop. Then the call's room is closed, joined or not."""
        audio = self.audio
        try:
            if audio is not None:
                await audio.close(play_out=reason not in _LEAVE_AT_ONCE)
        except Exception:
            logger.warning(
                f"Voice call {self.call_id}: its audio did not close cleanly",
                exc_info=True,
            )
        await self._close_room("call_end")

    async def _close_room(self, why: RoomCloseWhy) -> None:
        reaper = self.reaper
        if reaper is None or self._room_closed:
            return
        try:
            self._room_closed = await reaper.close(self.call_id, self.user_id, why)
        except Exception:
            logger.warning(
                f"Voice call {self.call_id}: its room was not closed", exc_info=True
            )

    async def close_vendor(self) -> None:
        live, self.live = self.live, None
        if live is not None:
            await live.close()

    # --- the call's room --------------------------------------------------

    async def join_room(self) -> None:
        """Take the call's audio into its LiveKit room. Raises whatever the
        join failed with."""
        # Away until the person's microphone is found in the room.
        self.away_since = time.monotonic()
        self.audio = await self.edge.join(
            call_id=self.call_id,
            user_id=self.user_id,
            ttl_s=self.caps.max_call_s + RELAY_ROOM_GRACE_S,
            hear=self.hear_room,
            presence=self.on_presence,
            stall=self.on_stall,
            lost=self.on_room_lost,
        )

    def on_room_lost(self) -> None:
        """The relay's own connection to the call's room is gone past the
        SDK's reconnect: nothing the person says reaches the call any more,
        so it ends at once."""
        if not self.ended.is_set():
            self.error = "room: lost"
            self.end("room_error")

    async def hear_room(self, pcm: bytes) -> None:
        """The person's audio from the room, taken once the call is live."""
        if self.hearing_room and not self.ended.is_set():
            await self.on_mic(pcm)

    def on_presence(self, present: bool) -> None:
        """The person's microphone came into the room, or left it, as
        LiveKit's own reconnect can make it do."""
        if present:
            self.away_since = None
            return
        if self.away_since is None:
            self.away_since = time.monotonic()
        if not self.ended.is_set():
            # Their audio stopped, maybe mid-sentence: close their open turn
            # so it does not hang the vendor's turn detection.
            self.spawn(
                self._vendor_call("stream end", lambda live: live.end_audio_stream()),
                "stream_end",
            )

    def on_stall(self, stalled: bool) -> None:
        """The person's audio stopped, or came back, while they stayed in the
        room: nothing they say can be heard meanwhile, so the idle clock
        waits."""
        self.stalled = stalled

    # --- the browser's socket ---------------------------------------------

    async def send_json(self, message: Mapping[str, object]) -> None:
        if self.client_gone:
            return
        async with self._send_lock:
            await self._send_locked(json.dumps(message))

    async def _send_locked(self, text: str) -> None:
        """Send on the call's socket, or keep it for a re-attach while the
        browser is away. The send lock is held."""
        if self.client_gone:
            return
        if self.detached_at is not None:
            if self.outbox_bytes + len(text) <= self.timing.outbox_bytes:
                self.outbox.append(text)
                self.outbox_bytes += len(text)
            return
        ws = self.ws
        try:
            await asyncio.wait_for(
                ws.send_text(text), timeout=self.timing.socket_timeout_s
            )
        except Exception:
            # Failed or stalled: the browser is no longer reading.
            self.detach("send failed", ws)

    async def _close(self, ws: ClientSocket, code: int) -> None:
        try:
            await asyncio.wait_for(
                ws.close(code=code), timeout=self.timing.socket_timeout_s
            )
        except Exception:
            logger.debug(f"Voice call {self.call_id}: socket did not close cleanly")

    def detach(self, why: str, ws: ClientSocket | None = None) -> None:
        """The browser's socket is gone: hold the call for a re-attach."""
        if ws is not None and ws is not self.ws:
            return
        if self.detached_at is not None or self.ended.is_set():
            return
        self.detached_at = time.monotonic()
        self._socket_done.set()
        # The person's audio goes on through the room meanwhile.
        logger.info(f"Voice call {self.call_id}: browser detached ({why})")

    async def attach(self, ws: ClientSocket) -> asyncio.Event:
        """Make *ws* the call's socket (a re-attach after a drop or a reload).

        Returns an event set once the call is done with *ws*.
        """
        async with self._attach_lock:
            old = self.ws
            # From here on the old socket hears nothing: what the call says is
            # kept for *ws*.
            if self.detached_at is None:
                self.detached_at = time.monotonic()
            away_since = self.detached_at
            if self.upstream_task is not None and not self.upstream_task.done():
                self.upstream_task.cancel()
            if old is not ws:
                await self._close(old, CLOSE_REPLACED)
            # Switched once the old socket is closed, under the send lock, so
            # nothing the call says lands ahead of ``resumed`` and what was
            # kept.
            async with self._send_lock:
                self._socket_done.set()
                self._socket_done = asyncio.Event()
                self.ws = ws
                gap = time.monotonic() - away_since
                outbox, self.outbox, self.outbox_bytes = self.outbox, [], 0
                self.detached_at = None
                self.reattaches += 1
                logger.info(
                    f"Voice call {self.call_id}: browser re-attached after "
                    f"{gap:.1f}s, {len(outbox)} messages replayed"
                )
                resumed = {"type": "resumed", "gap_s": round(gap, 3)}
                await self._send_locked(json.dumps(resumed))
                for text in outbox:
                    await self._send_locked(text)
            self.upstream_task = self.spawn(self.upstream(ws), "upstream")
            return self._socket_done

    async def upstream(self, ws: ClientSocket) -> None:
        while not self.ended.is_set():
            message: Mapping[str, Any]
            try:
                message = await ws.receive()
            except Exception:
                message = {"type": "websocket.disconnect"}
            if message.get("type") == "websocket.disconnect":
                self.detach(f"disconnect {message.get('code')}", ws)
                return
            if message.get("bytes") is not None:
                # The person's audio takes the room, never the socket.
                self.bad_frames += 1
                continue
            text = message.get("text")
            if text:
                await self.on_control(text)

    async def on_mic(self, pcm: bytes) -> None:
        now = time.monotonic()
        if not pcm or len(pcm) > MAX_FRAME_BYTES or len(pcm) % 2:
            self.bad_frames += 1
            return
        samples = len(pcm) // 2
        # No client can push audio into the vendor's context faster than real
        # time: past the burst, frames are dropped, not buffered.
        self.bucket = min(
            self.bucket + (now - self.bucket_at) * IN_RATE, BURST_S * IN_RATE
        )
        self.bucket_at = now
        if samples > self.bucket:
            self.dropped += samples
            return
        self.bucket -= samples
        self.received += samples
        if self.recording is not None:
            self.recording.append(pcm)
        if vad.heard(pcm):
            self.quiet_s = 0.0
            self.last_heard = now
        live = self.live
        if self.wrapping or self.reconnecting or live is None:
            return
        try:
            await asyncio.wait_for(live.send_audio(pcm), timeout=VENDOR_SEND_TIMEOUT_S)
        except Exception:
            # The vendor socket is going down; downstream reconnects it.
            return
        self.forwarded += samples

    async def on_control(self, text: str) -> None:
        if len(text.encode()) > MAX_TEXT_BYTES:
            return
        try:
            message = json.loads(text)
        except ValueError:
            return
        if not isinstance(message, dict):
            return
        kind = message.get("type")
        if kind == "hangup":
            self.end("user_hangup")
        elif kind == "ping":
            ping_id = message.get("id")
            echoable = isinstance(ping_id, int) and not isinstance(ping_id, bool)
            echoable = echoable or (isinstance(ping_id, str) and len(ping_id) <= 64)
            await self.send_json({"type": "pong", "id": ping_id if echoable else None})

    # --- the vendor's session ---------------------------------------------

    async def _vendor_call(
        self, what: str, call: Callable[[VendorSession], Awaitable[None]]
    ) -> bool:
        """Send something to the vendor, tolerating a session going down."""
        live = self.live
        if live is None or self.reconnecting:
            return False
        try:
            await asyncio.wait_for(call(live), timeout=VENDOR_SEND_TIMEOUT_S)
            return True
        except Exception as error:
            logger.info(
                f"Voice call {self.call_id}: {what} not sent ({type(error).__name__})"
            )
            return False

    def _vendor_failed(self, stage: VendorStage, error: BaseException) -> VendorFailure:
        """Log and count a failed Live connect or session with what the
        vendor said about why: its HTTP status, or its close code and reason,
        and never a key nor anything said on the call."""
        failure = vendor_failure(
            error, after_notice=stage == "session" and self.vendor_notice
        )
        VENDOR_FAILURES.labels(stage=stage, reason=failure.reason).inc()
        logger.warning(
            f"Voice call {self.call_id}: Gemini Live {_VENDOR_STAGES[stage]}: "
            f"{failure.describe()}"
        )
        return failure

    async def downstream(self, live: VendorSession) -> None:
        try:
            async for event in live.events():
                await self.on_event(event)
            raise ConnectionError("the vendor's stream ended")
        except Exception as error:
            if live is self.live and not self.ended.is_set() and not self.reconnecting:
                self.spawn(self.reconnect_vendor(error), "vendor_reconnect")

    async def reconnect_vendor(self, error: Exception) -> None:
        """Replace a dropped vendor session with one that knows the call so far.

        Only the relay can: it holds the conversation, the timers and the
        browser's socket, so the person hears a short apology, not a dead line.
        """
        dropped_at = time.monotonic()
        self.reconnecting = True
        failure = self._vendor_failed("session", error)
        await self.meter.unreported(
            sent_samples=self.forwarded,
            received_samples=self.agent_out,
            cut_off_reply=self.generating,
            why="vendor session dropped",
        )
        if self.cur.first_audio is not None or self.cur.text:
            self.cur.interrupted = True
            self.playout_end = dropped_at
            self._flush()
            await self.send_json({"type": "interrupted"})
            await self.on_turn_complete()
        await self.close_vendor()
        if self.ended.is_set():
            return
        if self.vendor_reconnects >= self.timing.max_vendor_reconnects:
            self.error = f"vendor: {failure.summary()}"
            self.end("vendor_error")
            return
        self.vendor_reconnects += 1
        try:
            live = await asyncio.wait_for(
                self.vendor.connect(opening=self.reseed_text()),
                timeout=self.timing.vendor_connect_timeout_s,
            )
        except Exception as reconnect_error:
            refused = self._vendor_failed("reconnect", reconnect_error)
            self.error = f"vendor reconnect: {refused.summary()}"
            self.end("vendor_error")
            return
        self.live = live
        self.vendor_notice = False
        self.reconnecting = False
        self.spawn(self.downstream(live), "downstream")
        logger.info(
            f"Voice call {self.call_id}: vendor reconnected in "
            f"{time.monotonic() - dropped_at:.2f}s"
        )

    def reseed_text(self) -> str:
        """The opening of a replacement session: the call so far, as text."""
        lines = [
            f"{'Otto' if line['who'] == 'otto' else 'User'}: {_as_heard(line['text'])}"
            for line in self.transcript()
        ]
        if not lines:
            return voice_interview_kickoff(self.first_name)
        name = f" Their first name is {self.first_name}." if self.first_name else ""
        return (
            SERVER_NOTE
            + "The call's connection dropped for a moment and is back."
            + name
            + " The conversation so far:\n"
            + "\n".join(lines)
            + "\nDo not greet again. Say in a few words that you're back, then "
            "carry on from where the conversation stopped."
        )

    def transcript(self) -> list[dict[str, str]]:
        """The call as heard live: Otto's turns and the person's captions."""
        lines: list[dict[str, str]] = []

        def heard_after(index: int) -> None:
            for turn in self.user_turns:
                if turn.after == index:
                    lines.append({"who": "user", "text": turn.text})

        heard_after(-1)
        for turn in self.turns:
            if turn.text:
                lines.append({"who": "otto", "text": turn.text})
            heard_after(turn.index)
        if self.cur.text.strip():
            lines.append({"who": "otto", "text": self.cur.text.strip()})
        if self.user_buf.strip():
            lines.append({"who": "user", "text": self.user_buf.strip()})
        return lines

    async def on_event(self, event: VendorEvent) -> None:
        now = time.monotonic()
        match event:
            case Usage():
                await self.meter.live_usage(
                    event, sent_samples=self.forwarded, received_samples=self.agent_out
                )
                if self.meter.over_cap:
                    self.end("spend_cap")
            case GoingAway(time_left=time_left):
                self.vendor_notice = True
                logger.info(
                    f"Voice call {self.call_id}: vendor going away in "
                    f"{time_left or 'an unknown time'}"
                )
            case ToolCall():
                await self.on_tool_call(event)
            case UserText(text=text):
                self.user_buf += text
                await self.send_json({"type": "user_text", "text": text})
            case AgentText(text=text):
                self.cur.text += text
                await self.send_json({"type": "agent_text", "text": text})
            case Interrupted():
                self.cur.interrupted = True
                self.playout_end = now
                self._flush()
                await self.send_json({"type": "interrupted"})
            case AgentAudio(pcm=pcm):
                await self.on_agent_audio(pcm, now)
            case TurnComplete():
                await self.on_turn_complete()

    async def on_agent_audio(self, pcm: bytes, now: float) -> None:
        turn = self.cur
        if turn.first_audio is None:
            turn.first_audio = now
            if self.first_audio_s is None:
                self.first_audio_s = now - self.started
                FIRST_AUDIO_SECONDS.observe(self.first_audio_s)
            if self.last_heard is not None:
                REPLY_SECONDS.observe(now - self.last_heard)
                self.last_heard = None
            if (
                self.turns
                and self.classifier is not None
                and not self.wrapping
                and self.user_buf.strip()
            ):
                self.spawn(
                    self.classify_soon(len(self.turns), self.turns[-1].text),
                    "scope_trigger",
                )
        self.generating = True
        samples = len(pcm) // 2
        self.agent_out += samples
        seconds = samples / OUT_RATE
        turn.audio_s += seconds
        self.playout_end = max(self.playout_end, now) + seconds
        if self.audio is not None:
            await self.audio.play(pcm)

    def _flush(self) -> None:
        """Drop what Otto has not said yet in the room."""
        if self.audio is not None:
            self.audio.flush()

    async def on_tool_call(self, event: ToolCall) -> None:
        results: list[ToolResult] = []
        for call in event.calls:
            if call.name != VOICE_INTERVIEW_END_TOOL:
                results.append(
                    ToolResult(
                        call_id=call.call_id,
                        name=call.name,
                        response={"error": "unknown tool"},
                    )
                )
                continue
            if not self.finish_called:
                self.finish_called = True
                self.finish_at = time.monotonic()
            first_task = " ".join(str(call.args.get(END_SIGNAL_FIRST_TASK, "")).split())
            if first_task:
                self.first_task = first_task[:MAX_FIRST_TASK_CHARS]
            results.append(
                ToolResult(
                    call_id=call.call_id, name=call.name, response={"result": "saved"}
                )
            )
        await self._vendor_call(
            "tool results", lambda live: live.send_tool_results(results)
        )

    async def on_turn_complete(self) -> None:
        turn = self.cur
        turn.index = len(self.turns)
        turn.text = turn.text.strip()
        self.turns.append(turn)
        self.cur = AgentTurn()
        self.generating = False
        await self.send_json(
            {
                "type": "turn_complete",
                "turn": turn.index,
                "turns_left": max(0, self.caps.max_agent_turns - len(self.turns)),
            }
        )
        heard = self.user_buf.strip()
        self.user_buf = ""
        if heard:
            self.user_turns.append(UserTurn(after=turn.index - 1, text=heard))
        self.turn_marks.append(
            TurnMark(index=turn.index, at=self.received, text=turn.text)
        )
        await self.store.record(turns=[mark.model_dump() for mark in self.turn_marks])

        spoke_at = turn.first_audio
        if self.wrapping:
            if spoke_at is not None and spoke_at >= self.wrap_started:
                self.end(self.wrapping)
            return
        if (
            self.finish_called
            and spoke_at is not None
            and spoke_at >= self.finish_at
            and not turn.interrupted
        ):
            self.end("finished")
            return
        if len(self.turns) >= self.caps.max_agent_turns:
            self.end("turn_cap")
            return
        if len(self.turns) >= self.caps.max_agent_turns - self.caps.soft_wrap_turns:
            await self.nudge("turns")

    # --- the controls -----------------------------------------------------

    async def watchdog(self) -> None:
        tick = self.timing.tick_s
        last = time.monotonic()
        next_end_check = last
        next_refresh = last + LIVE_CALL_REFRESH_S
        next_checkpoint = last + self.timing.checkpoint_s
        while not self.ended.is_set():
            await asyncio.sleep(tick)
            now = time.monotonic()
            gap, last = now - last, now
            elapsed = self.elapsed()
            if elapsed >= self.caps.max_call_s:
                self.end("wall_clock")
                return
            # The nudge is sent off the watchdog, which enforces every limit.
            if (
                elapsed >= self.caps.max_call_s - self.caps.soft_wrap_s
                and self._claim_nudge("time")
            ):
                self.spawn(self._send_nudge("time"), "time_nudge")
            # Quiet counts only while the browser is attached, the person's
            # microphone reaches the relay with its audio flowing, and Otto's
            # speech has played out.
            if (
                now >= self.playout_end
                and self.detached_at is None
                and self.away_since is None
                and not self.stalled
            ):
                self.quiet_s += gap
            if self.quiet_s >= self.caps.idle_s:
                self.end("idle")
                return
            if self.wrapping and now > self.wrap_deadline:
                self.end(self.wrapping)
                return
            if self.finish_called and now > self.finish_at + self.timing.finish_grace_s:
                self.end("finished")
                return
            if (
                self.detached_at is not None
                and now - self.detached_at > self.timing.reattach_s
            ):
                self.end("client_gone")
                return
            # The person's microphone out of the room as long as a socket may
            # be away: LiveKit's own reconnect did not bring it back.
            if (
                self.away_since is not None
                and now - self.away_since > self.timing.reattach_s
            ):
                self.end("client_gone")
                return
            if now >= next_end_check:
                next_end_check = now + self.timing.end_flag_poll_s
                if await self.store.end_requested():
                    self.end("superseded")
                    return
            if now >= next_refresh:
                next_refresh = now + LIVE_CALL_REFRESH_S
                await self.store.refresh_live(time.time())
            if self.recording is not None and now >= next_checkpoint:
                next_checkpoint = now + self.timing.checkpoint_s
                self.recording.checkpoint_soon()

    def _claim_nudge(self, kind: str) -> bool:
        """Whether *kind*'s nudge is still to send; it is sent once a call."""
        if kind in self.nudged or self.wrapping:
            return False
        self.nudged.add(kind)
        return True

    async def _send_nudge(self, kind: str) -> None:
        await self._vendor_call(
            f"{kind} nudge",
            lambda live: live.send_note(NUDGES[kind], turn_complete=False),
        )

    async def nudge(self, kind: str) -> None:
        if self._claim_nudge(kind):
            await self._send_nudge(kind)

    async def classify_soon(self, answering: int, before: str) -> None:
        """Label the person's turn as soon as Otto starts answering it.

        By Otto's first audio the vendor has closed the person's turn; the
        short wait lets its last caption land.
        """
        await asyncio.sleep(self.timing.scope_settle_s)
        if len(self.turns) == answering:
            text = self.user_buf.strip()
        else:
            text = next(
                (t.text for t in self.user_turns if t.after == answering - 1), ""
            )
        if text and not self.wrapping and not self.ended.is_set():
            self.scope_queue.put_nowait((before, text))

    async def scope_worker(self, classifier: ScopeClassifier) -> None:
        while not self.ended.is_set():
            before, user = await self.scope_queue.get()
            try:
                judgement = await asyncio.wait_for(
                    classifier(before=before, user=user),
                    timeout=self.timing.scope_timeout_s,
                )
            except Exception:
                # Fail open: the caps still bound a call nobody could judge.
                logger.warning(
                    f"Voice call {self.call_id}: a turn was not judged", exc_info=True
                )
                continue
            await self.meter.classifier_usage(
                tokens_in=judgement.tokens_in, tokens_out=judgement.tokens_out
            )
            if self.meter.over_cap:
                self.end("spend_cap")
                return
            if judgement.label is None:
                continue
            self.labels.append(judgement.label)
            if (
                scope_verdict(self.labels) == "scope"
                and not self.wrapping
                and not self.ended.is_set()
            ):
                await self.wrap_up("scope")

    async def wrap_up(self, reason: EndReason) -> None:
        """Have Otto say goodbye, then end the call with *reason*."""
        self.wrapping = reason
        self.wrap_started = time.monotonic()
        self.wrap_deadline = self.wrap_started + self.timing.wrap_grace_s
        logger.info(f"Voice call {self.call_id}: wrapping up ({reason})")
        await self.send_json({"type": "wrap_up", "reason": reason})
        spoken = await self._vendor_call(
            "goodbye", lambda live: live.end_audio_stream()
        ) and await self._vendor_call(
            "goodbye", lambda live: live.send_note(GOODBYES[reason], turn_complete=True)
        )
        if not spoken:
            self.end(reason)


def _as_heard(text: str) -> str:
    """Words from the call, made unable to pass as a server note."""
    return text.replace("[", "(").replace("]", ")")
