"""The voice relay's web app: the call socket and the probes.

- ``WS /voice/call`` takes ``start {token, leg}`` or ``resume {ticket}`` as
  its first message, within ``START_TIMEOUT_S``, then runs the call or
  re-attaches to it (``session``). The token rides the first message, never
  the URL, so it stays out of load balancer logs. The browser's origin must
  be one the platform allows (``BACKEND_CORS_ALLOW_ORIGINS``).
- ``GET /healthz`` is liveness. ``GET /readyz`` is readiness, which fails
  while the process drains so the load balancer sends it no new calls.

The relay trusts the user a start token names, because only the REST API
and the relay hold the key that signs it; it reads no platform session.

Once a call ends, its post-call job runs here too (``postcall``), and the
process sweeps for jobs no pod is running when it starts and every
``POSTCALL_SWEEP_S``.
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
    MAX_TEXT_BYTES,
    START_TIMEOUT_S,
    TIMING,
    Caps,
    Timing,
)
from backend.voice_relay.metering import LIVE_CALLS, CallMeter, CostSink
from backend.voice_relay.postcall import PostCall
from backend.voice_relay.record import CallRecording
from backend.voice_relay.scope import ScopeClassifier
from backend.voice_relay.session import Session
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
    record_call: Callable[[str, str], Awaitable[None]]
    live_model: str
    # Live calls the whole platform takes at once.
    max_concurrent: int
    # Each call's transcript of record and hand-off, and the sweep.
    postcall: PostCall
    # Where a call's recording is spooled before it is uploaded.
    spool_dir: str
    # Where priced rows go; None logs them to PlatformCostLog.
    cost_sink: CostSink | None = None
    caps: Caps = CAPS
    timing: Timing = TIMING


class CallSlot:
    """One of this process's call slots, held from a start to the end of its
    call, or to its refusal."""

    def __init__(self, relay: "Relay") -> None:
        self._relay = relay
        self._call_id: str | None = None
        self._released = False

    def hold(self, session: Session) -> None:
        """The slot's call has started: resumable, and counted as live."""
        self._call_id = session.call_id
        self._relay.sessions[session.call_id] = session
        LIVE_CALLS.inc()

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
        """Whether a call, starting or live, or a post-call job is still
        running here."""
        return self.held_slots > 0 or bool(
            self.deps is not None and self.deps.postcall.jobs
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
        ready = not relay.draining and relay.deps is not None
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
    """Sweep for post-call jobs no pod is running while the app serves."""
    if relay.deps is None:
        yield
        return
    sweep = asyncio.create_task(
        relay.deps.postcall.sweep_forever(), name="voice-postcall-sweep"
    )
    try:
        yield
    finally:
        sweep.cancel()
        await asyncio.gather(sweep, return_exceptions=True)


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
    if first.get("leg") not in (None, *tokens.LEGS):
        await _refuse(ws, "protocol", CLOSE_POLICY)
        return
    redis: AsyncRedisClient | None = None
    try:
        redis = await deps.redis()
        await tokens.spend(redis, claims)
        holder = await locks.take_user_lock(redis, claims.user_id, claims.call_id)
    except tokens.TokenError as error:
        await _refuse(ws, str(error), CLOSE_POLICY)
        return
    except Exception:
        logger.warning(f"Voice call {claims.call_id}: state unavailable", exc_info=True)
        if redis is not None:
            await locks.CallStore(
                redis, call_id=claims.call_id, user_id=claims.user_id
            ).release()
            await _free_token(redis, claims)
        await _refuse(ws, "unavailable", CLOSE_TRY_AGAIN)
        return
    if holder is not None:
        await _refuse(ws, "call_in_progress", CLOSE_POLICY, call_id=holder or None)
        return

    store = locks.CallStore(
        redis,
        call_id=claims.call_id,
        user_id=claims.user_id,
        holder=deps.postcall.pod,
    )
    try:
        # The platform's live-call ceiling, held here rather than only read
        # when the token was minted.
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
            await _free_token(redis, claims)
            await _refuse(ws, "unavailable", CLOSE_TRY_AGAIN)
            return
        if not admitted:
            await _free_token(redis, claims)
            await _refuse(ws, "busy", CLOSE_TRY_AGAIN)
            return
        # Recorded once the call holds its lock and its place, so the row
        # names a call that ran.
        try:
            await asyncio.wait_for(
                deps.record_call(claims.user_id, claims.call_id),
                timeout=RECORD_CALL_TIMEOUT_S,
            )
        except Exception:
            logger.warning(
                f"Voice call {claims.call_id}: not recorded on the onboarding row",
                exc_info=True,
            )
            await _free_token(redis, claims)
            await _refuse(ws, "unavailable", CLOSE_TRY_AGAIN)
            return

        # This relay carries a call's audio on the control socket.
        leg: tokens.Leg = "websocket"
        await store.record(
            status="connecting",
            uid=claims.user_id,
            pod=relay.pod,
            leg=leg,
            started=round(time.time(), 3),
        )
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
            recording=CallRecording(
                call_id=claims.call_id,
                user_id=claims.user_id,
                spool_dir=deps.spool_dir,
                store=deps.postcall.recordings,
                call_store=store,
            ),
        )
        slot.hold(session)
        await session.run()
    finally:
        await store.release()


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


async def _free_token(redis: AsyncRedisClient, claims: tokens.StartClaims) -> None:
    """Make the start token good again before a refusal that asks the browser
    to try again, so it can, with the same token. The live-call lock still
    holds the person to one call."""
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
