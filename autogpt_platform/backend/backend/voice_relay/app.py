"""The voice relay's web app: the call socket and the probes.

- ``WS /voice/call`` takes ``start {token, leg}`` or ``resume {ticket}`` as
  its first message, within ``START_TIMEOUT_S``, then runs the call or
  re-attaches to it (``session``). The token rides the first message, never
  the URL, so it stays out of load balancer logs. The browser's origin must
  be one the platform allows (``BACKEND_CORS_ALLOW_ORIGINS``). The token
  names the call's leg; the start's ``leg`` may take a LiveKit call onto the
  socket (the browser could not join the room), never the other way.
- ``GET /healthz`` is liveness. ``GET /readyz`` is readiness, which fails
  while the process drains, so the load balancer sends it no new calls, and
  while it could not record a call: its spool directory takes no writes.

The relay trusts the user a start token names, because only the REST API
and the relay hold the key that signs it; it reads no platform session.

Once a call ends, its transcript job runs here too (``postcall``), and the
process sweeps for calls no pod is transcribing when it starts and every
``POSTCALL_SWEEP_S``. The REST API hands each transcript off. With a LiveKit
project, the process also sweeps the call rooms every ``ROOM_SWEEP_S`` and
closes the ones no call needs (``room_reaper``). As it starts it deletes the
spools a call of an earlier process left; as it stops, its post-call jobs
settle before its services close, so each gives up its claim while Redis
still answers.
"""

import asyncio
import json
import logging
import re
import socket
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any

from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from backend.api.utils.cors import CorsParams
from backend.data.redis_client import AsyncRedisClient
from backend.voice_relay import locks, tokens
from backend.voice_relay.config import (
    CAPS,
    DEFAULT_DRAIN_S,
    MAX_TEXT_BYTES,
    START_TIMEOUT_S,
    TIMING,
    Caps,
    Timing,
)
from backend.voice_relay.metering import FALLBACKS, LIVE_CALLS, CallMeter, CostSink
from backend.voice_relay.postcall import PostCall
from backend.voice_relay.record import (
    CallRecording,
    remove_stale_spools,
    spool_writable,
)
from backend.voice_relay.room_reaper import RoomReaper
from backend.voice_relay.scope import ScopeClassifier
from backend.voice_relay.session import RoomEdge, Session
from backend.voice_relay.vendor import VendorConnector

logger = logging.getLogger(__name__)

# Close codes for a refused socket: a bad request, a server that cannot take
# the call now (try again), and a call that is not on this server.
CLOSE_POLICY = 1008
CLOSE_TRY_AGAIN = 1013
CLOSE_NO_CALL = 4404

FIRST_NAME_TIMEOUT_S = 2.0
RECORD_CALL_TIMEOUT_S = 5.0
REFUSE_TIMEOUT_S = 5.0
SPOOL_PROBE_TIMEOUT_S = 2.0
# How long the post-call jobs still running at shutdown get to finish before
# they are cancelled, giving up their claims for another pod's sweep.
SHUTDOWN_JOBS_S = 5.0
MAX_FIRST_NAME_CHARS = 40
_NAME_MARKS = "'-’"


class RelayDeps(BaseModel):
    """What calls run on: the production services, or test doubles."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vendor: VendorConnector
    classifier: ScopeClassifier | None
    redis: Callable[[], Awaitable[AsyncRedisClient]]
    # The display name on a user's account, or None.
    account_name: Callable[[str], Awaitable[str | None]]
    # Records a started call on the user's onboarding row: (user id, call id).
    # Returns the call it replaced there, or None.
    record_call: Callable[[str, str], Awaitable[str | None]]
    live_model: str
    # Live calls the whole platform takes at once.
    max_concurrent: int
    # Each call's transcript of record, and the sweep.
    postcall: PostCall
    # Where a call's recording is spooled before it is uploaded.
    spool_dir: str
    # A spool older than this is no live call's: the call cap and the drain.
    spool_stale_s: float = CAPS.max_call_s + DEFAULT_DRAIN_S
    # Where priced rows go; None logs them to PlatformCostLog.
    cost_sink: CostSink | None = None
    # Joins the rooms of calls on the LiveKit leg, and closes and sweeps
    # them; None where this relay has no LiveKit project, so such calls take
    # the socket.
    edge: RoomEdge | None = None
    reaper: RoomReaper | None = None
    caps: Caps = CAPS
    timing: Timing = TIMING


class CallSlot:
    """One of this process's call slots, held from a start to the end of its
    call, or to its refusal."""

    def __init__(self, relay: "Relay") -> None:
        self._relay = relay
        self._call_id: str | None = None
        self._released = False

    def hold(self, session: Session) -> bool:
        """The slot's call has started: resumable, and counted as live.

        False, holding nothing, when a session for the same call already runs
        on this process: a call has one session.
        """
        if session.call_id in self._relay.sessions:
            return False
        self._call_id = session.call_id
        self._relay.sessions[session.call_id] = session
        LIVE_CALLS.inc()
        return True

    def release(self) -> None:
        """Give the slot back. Idempotent."""
        if self._released:
            return
        self._released = True
        self._relay.held_slots -= 1
        if self._call_id is not None:
            self._relay.sessions.pop(self._call_id, None)
            LIVE_CALLS.dec()


class Relay:
    """This process's live calls, and whether it is draining."""

    def __init__(
        self,
        *,
        max_calls: int,
        origins: CorsParams,
        deps: RelayDeps | None = None,
    ) -> None:
        self.max_calls = max_calls
        self.origins = origins
        self.deps = deps
        self.sessions: dict[str, Session] = {}
        # Slots held by calls starting or live; at most max_calls.
        self.held_slots = 0
        self.draining = False
        self.drain_started: float | None = None
        self.pod = socket.gethostname()
        self.loop: asyncio.AbstractEventLoop | None = None

    def start_drain(self) -> None:
        self.draining = True
        self.drain_started = time.monotonic()
        if self.deps is not None:
            # A draining process takes no other pod's post-call jobs.
            self.deps.postcall.stopped = True

    @property
    def busy(self) -> bool:
        """Whether a call, starting or live, a post-call job, Otto's last
        words in a call's room, or a room's close are still running here."""
        deps = self.deps
        if self.held_slots > 0:
            return True
        if deps is None:
            return False
        return (
            bool(deps.postcall.jobs)
            or (deps.edge is not None and deps.edge.rooms > 0)
            or (deps.reaper is not None and deps.reaper.closing > 0)
        )

    def reserve_slot(self) -> CallSlot | None:
        """Hold one of this process's call slots, or None when all are held.

        Checked and taken in one step, with nothing awaited between, so
        starts that arrive together can never hold more than max_calls.
        """
        if self.held_slots >= self.max_calls:
            return None
        self.held_slots += 1
        return CallSlot(self)

    def origin_allowed(self, origin: str | None) -> bool:
        if not origin:
            return False
        allowed = self.origins["allow_origins"]
        if "*" in allowed or origin in allowed:
            return True
        pattern = self.origins["allow_origin_regex"]
        return pattern is not None and re.fullmatch(pattern, origin) is not None


def create_app(
    relay: Relay,
    *,
    setup: Callable[[], AbstractAsyncContextManager[RelayDeps]] | None = None,
) -> FastAPI:
    """The relay's app. *setup* builds its services at startup and closes
    them at shutdown; without it, *relay* must come with its own."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        relay.loop = asyncio.get_running_loop()
        if setup is None:
            async with _sweeping(relay):
                yield
            return
        async with setup() as deps:
            relay.deps = deps
            async with _sweeping(relay):
                yield

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/healthz")
    async def healthz() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        ready = (
            not relay.draining
            and relay.deps is not None
            and await _spool_ready(relay.deps.spool_dir)
        )
        return JSONResponse(
            {"ready": ready, "live_calls": len(relay.sessions)},
            status_code=200 if ready else 503,
        )

    @app.websocket("/voice/call")
    async def call_socket(ws: WebSocket) -> None:
        await serve_socket(relay, ws)

    return app


@asynccontextmanager
async def _sweeping(relay: Relay) -> AsyncIterator[None]:
    """Sweep for post-call jobs no pod is running, and for call rooms no
    call needs, while the app serves; settle this process's jobs before its
    services close."""
    deps = relay.deps
    if deps is None:
        yield
        return
    await asyncio.to_thread(remove_stale_spools, deps.spool_dir, deps.spool_stale_s)
    sweeps = [
        asyncio.create_task(deps.postcall.sweep_forever(), name="voice-postcall-sweep")
    ]
    if deps.reaper is not None:
        sweeps.append(
            asyncio.create_task(deps.reaper.sweep_forever(), name="voice-room-sweep")
        )
    try:
        yield
    finally:
        for sweep in sweeps:
            sweep.cancel()
        await asyncio.gather(*sweeps, return_exceptions=True)
        await deps.postcall.settle(SHUTDOWN_JOBS_S)


async def _spool_ready(spool_dir: str) -> bool:
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(spool_writable, spool_dir), timeout=SPOOL_PROBE_TIMEOUT_S
        )
    except Exception:
        logger.warning("Voice spool directory unreadable", exc_info=True)
        return False


class _Refused(Exception):
    def __init__(self, error: str, code: int = CLOSE_POLICY) -> None:
        super().__init__(error)
        self.error = error
        self.code = code


async def serve_socket(relay: Relay, ws: WebSocket) -> None:
    if not relay.origin_allowed(ws.headers.get("origin")):
        # Closed before the handshake completes: the browser sees a 403.
        await ws.close(code=CLOSE_POLICY)
        return
    await ws.accept()
    try:
        first = await asyncio.wait_for(_first_message(ws), timeout=START_TIMEOUT_S)
    except TimeoutError:
        await _refuse(ws, "start_timeout", CLOSE_POLICY)
        return
    except _Refused as refused:
        await _refuse(ws, refused.error, refused.code)
        return
    if first["type"] == "resume":
        await _resume(relay, ws, first.get("ticket"))
    else:
        await _start(relay, ws, first)


async def _first_message(ws: WebSocket) -> dict[str, Any]:
    """The socket's first message: ``start`` or ``resume``, or a refusal."""
    message = await ws.receive()
    text = message.get("text")
    if not isinstance(text, str) or len(text.encode()) > MAX_TEXT_BYTES:
        raise _Refused("protocol")
    try:
        first = json.loads(text)
    except ValueError:
        raise _Refused("protocol") from None
    if not isinstance(first, dict) or first.get("type") not in ("start", "resume"):
        raise _Refused("protocol")
    return first


async def _start(relay: Relay, ws: WebSocket, first: Mapping[str, Any]) -> None:
    started = time.monotonic()
    deps = relay.deps
    if relay.draining or deps is None:
        await _refuse(ws, "draining", CLOSE_TRY_AGAIN)
        return
    slot = relay.reserve_slot()
    if slot is None:
        await _refuse(ws, "busy", CLOSE_TRY_AGAIN)
        return
    try:
        await _start_call(relay, deps, slot, ws, first, started)
    finally:
        slot.release()


async def _start_call(
    relay: Relay,
    deps: RelayDeps,
    slot: CallSlot,
    ws: WebSocket,
    first: Mapping[str, Any],
    started: float,
) -> None:
    try:
        claims = tokens.verify_start(first.get("token"))
    except tokens.TokenError as error:
        await _refuse(ws, str(error), CLOSE_POLICY)
        return
    leg = leg_of(claims.leg, first.get("leg"))
    if leg is None:
        await _refuse(ws, "protocol", CLOSE_POLICY)
        return
    try:
        redis = await asyncio.wait_for(deps.redis(), timeout=locks.REDIS_TIMEOUT_S)
        await asyncio.wait_for(
            tokens.spend(redis, claims), timeout=locks.REDIS_TIMEOUT_S
        )
    except tokens.TokenError as error:
        await _refuse(ws, str(error), CLOSE_POLICY)
        return
    except Exception:
        # Whether this attempt spent the token is unknown, so it gives back
        # nothing: the token and the lock may be a live call's.
        logger.warning(f"Voice call {claims.call_id}: state unavailable", exc_info=True)
        await _refuse(ws, "unavailable", CLOSE_TRY_AGAIN)
        return

    store = locks.CallStore(
        redis,
        call_id=claims.call_id,
        user_id=claims.user_id,
        holder=deps.postcall.pod,
    )
    try:
        holder = await asyncio.wait_for(
            locks.take_user_lock(redis, claims.user_id, claims.call_id),
            timeout=locks.REDIS_TIMEOUT_S,
        )
    except Exception:
        logger.warning(f"Voice call {claims.call_id}: lock unavailable", exc_info=True)
        # The token is this attempt's, and the lock may be.
        await _refuse_try_again(ws, store, claims, "unavailable")
        return
    if holder == "":
        # The lock changed hands while it was read, which passes: the start
        # gives back what it took so the same token can try again.
        await _refuse_try_again(ws, store, claims, "unavailable")
        return
    if holder is not None:
        await _refuse(ws, "call_in_progress", CLOSE_POLICY, call_id=holder)
        return
    refusal, replaced = await _admit(deps, redis, claims)
    if refusal is not None:
        await _refuse_try_again(ws, store, claims, refusal)
        return
    # The person's call before this one, on every write of the call's, so
    # their calls in between can be found from the latest without a scan.
    store.prev = replaced

    owns_state = True
    try:
        session = Session(
            call_id=claims.call_id,
            user_id=claims.user_id,
            leg=leg,
            ws=ws,
            vendor=deps.vendor,
            store=store,
            meter=CallMeter(
                user_id=claims.user_id,
                call_id=claims.call_id,
                leg=leg,
                model=deps.live_model,
                redis=redis,
                sink=deps.cost_sink,
            ),
            classifier=deps.classifier,
            first_name=await _first_name(deps, claims.user_id),
            caps=deps.caps,
            timing=deps.timing,
            started=started,
            on_end=lambda: _hand_over(deps, slot, session),
            edge=deps.edge,
            token_leg=claims.leg,
            reaper=deps.reaper,
        )
        if not slot.hold(session):
            # The call already runs here, so the lock and the place this
            # attempt took are that call's.
            owns_state = False
            await _refuse(ws, "call_in_progress", CLOSE_POLICY, call_id=claims.call_id)
            return
        # Made once the slot holds the call: the recording opens the call's
        # spool, which a refused start must never touch.
        session.recording = CallRecording(
            call_id=claims.call_id,
            user_id=claims.user_id,
            spool_dir=deps.spool_dir,
            store=deps.postcall.recordings,
            call_store=store,
        )
        # Pending before its first record write: a pod that dies between the
        # two leaves a pending call with no record, which the relay's sweep
        # drops once the call is no longer counted live.
        await store.pend_postcall()
        await store.record(
            status="connecting",
            pod=relay.pod,
            leg=leg,
            started=round(time.time(), 3),
            ending=claims.ending,
        )
        if leg != claims.leg:
            # The browser could not join the room, and said so in its start.
            FALLBACKS.labels(claims.leg, "browser_join").inc()
            logger.info(
                f"Voice call {claims.call_id}: the browser's audio takes the socket"
            )
        await session.run()
    finally:
        if owns_state:
            await store.release()


def leg_of(token_leg: tokens.Leg, asked: object) -> tokens.Leg | None:
    """The leg a start runs on: its token's, or the socket where the browser
    takes a LiveKit call there. None for any other ask, a LiveKit leg the
    token does not name included: the token, signed under the edge flag, is
    the only way onto a room."""
    if asked is None or asked == token_leg:
        return token_leg
    if asked == "websocket":
        return "websocket"
    return None


async def _admit(
    deps: RelayDeps, redis: AsyncRedisClient, claims: tokens.StartClaims
) -> tuple[str | None, str | None]:
    """Take the call's place under the platform's live-call ceiling, then
    record the call on the person's onboarding row.

    Returns the refusal to send when either fails, or None, and the call the
    row named before this one.
    """
    try:
        admitted = await asyncio.wait_for(
            locks.acquire_live_call(
                redis, claims.call_id, time.time(), deps.max_concurrent
            ),
            timeout=locks.REDIS_TIMEOUT_S,
        )
    except Exception:
        logger.warning(
            f"Voice call {claims.call_id}: live calls unreadable", exc_info=True
        )
        return "unavailable", None
    if not admitted:
        return "busy", None
    # Recorded once the call holds its lock and its place, so the row names a
    # call that ran.
    try:
        replaced = await asyncio.wait_for(
            deps.record_call(claims.user_id, claims.call_id),
            timeout=RECORD_CALL_TIMEOUT_S,
        )
    except Exception:
        logger.warning(
            f"Voice call {claims.call_id}: not recorded on the onboarding row",
            exc_info=True,
        )
        return "unavailable", None
    return None, replaced


def _hand_over(deps: RelayDeps, slot: CallSlot, session: Session) -> None:
    """A call's end: its post-call work starts, then its slot is given back,
    so a drain never finds this process idle between the two."""
    try:
        deps.postcall.after_call(
            call_id=session.call_id,
            recording=session.recording,
            claimed=session.postcall_claimed,
            first_name=session.first_name,
        )
    finally:
        slot.release()


async def _resume(relay: Relay, ws: WebSocket, ticket: object) -> None:
    """Re-attach *ws* to a live call on this process. Allowed while draining."""
    try:
        claims = tokens.verify_resume(ticket)
    except tokens.TokenError as error:
        expired = str(error) == "token_expired"
        await _refuse(ws, str(error), CLOSE_NO_CALL if expired else CLOSE_POLICY)
        return
    session = relay.sessions.get(claims.call_id)
    if session is None or session.user_id != claims.user_id or session.ended.is_set():
        await _refuse(ws, "no_call", CLOSE_NO_CALL)
        return
    done = await session.attach(ws)
    await done.wait()


async def _refuse_try_again(
    ws: WebSocket, store: locks.CallStore, claims: tokens.StartClaims, error: str
) -> None:
    """Refuse a start with "try again" (1013) once this attempt has given back
    its lock, its place in the live-call count and then its token, so a retry
    with the same token finds all three free and nothing of this attempt's
    runs after it."""
    await store.release()
    await _free_token(store.redis, claims)
    await _refuse(ws, error, CLOSE_TRY_AGAIN)


async def _free_token(redis: AsyncRedisClient, claims: tokens.StartClaims) -> None:
    """Make the start token good again, so the browser can retry with it."""
    try:
        await asyncio.wait_for(
            tokens.unspend(redis, claims), timeout=locks.REDIS_TIMEOUT_S
        )
    except Exception:
        logger.info(f"Voice call {claims.call_id}: start token not freed")


async def _refuse(ws: WebSocket, error: str, code: int, **extra: object) -> None:
    """Tell the browser why, then close. Bounded: a browser that stops
    reading holds nothing here."""
    try:
        await asyncio.wait_for(
            ws.send_text(json.dumps({"type": "error", "error": error, **extra})),
            timeout=REFUSE_TIMEOUT_S,
        )
        await asyncio.wait_for(ws.close(code=code), timeout=REFUSE_TIMEOUT_S)
    except Exception:
        logger.debug("A refused socket was already gone")


async def _first_name(deps: RelayDeps, user_id: str) -> str | None:
    try:
        name = await asyncio.wait_for(
            deps.account_name(user_id), timeout=FIRST_NAME_TIMEOUT_S
        )
    except Exception:
        logger.info("Voice call: the account's name could not be read", exc_info=True)
        return None
    return first_name_of(name)


def first_name_of(name: str | None) -> str | None:
    """The first word of an account's name, when it reads as a name.

    It goes into the conversation, so anything but letters, an apostrophe or
    a hyphen leaves the call without one.
    """
    words = (name or "").split()
    if not words:
        return None
    word = words[0].strip(".,")
    if not word or len(word) > MAX_FIRST_NAME_CHARS:
        return None
    if not all(char.isalpha() or char in _NAME_MARKS for char in word):
        return None
    if not any(char.isalpha() for char in word):
        return None
    return word
