"""
ShimDaemon — the main event loop.

Connects to the AutoGPT platform over WebSocket, sends HELLO with
detected capabilities, consumes HELLO_ACK (which may revise concurrency
and timeout limits), then pumps inbound messages through handlers.

Concurrency model: one task per inbound request, capped by a semaphore
sized from HELLO_ACK.max_concurrent (default 4). PING is handled inline
on the main loop (not via the semaphore) so a saturated handler queue
can't starve keepalive.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import contextvars
import errno
import json
import logging
import math
import random
import secrets
import threading
import urllib.parse
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Never, cast

import httpx
import keyring.errors
import websockets
from pydantic import BaseModel, ValidationError
from websockets.asyncio.client import connect

from . import handlers, platform_info
from .audit import (
    AuditKeyStorageError,
    AuditKeyUnavailableError,
    AuditLogCorruptError,
    AuditWriter,
    SessionAuditWriter,
    get_or_create_audit_key,
)
from .auth import CredentialScope, CredentialScopeError, KeychainTokenStore
from .computer_use import DriverUnavailable
from .computer_use.driver import COMPUTER_USE_INSTALL
from .computer_use.mcp_client import McpError
from .config import ShimConfig
from .directory_browser import DirectoryBrowseError, DirectoryBrowser, DirectoryListing
from .handlers import (
    CommandHandler,
    ComputerUseHandler,
    FileHandler,
    LocalLLMHandler,
    RecordingHandler,
)
from .protocol import (
    MAX_WEBSOCKET_MESSAGE_BYTES,
    ActivateSessionMessage,
    ApplyRecordingReviewMessage,
    Arch,
    AttachSessionMessage,
    ComputerToolCallMessage,
    ComputerToolsRequestMessage,
    DetachSessionMessage,
    DirectoryListRequestMessage,
    DirectoryListResponseMessage,
    DirectoryListResponsePayload,
    DirectoryReferencePayload,
    ErrorCode,
    ExecuteCommandMessage,
    FileDeleteMessage,
    FileListMessage,
    FileMoveMessage,
    FileReadMessage,
    FileStatMessage,
    FileWriteMessage,
    HelloAckMessage,
    HelloMessage,
    HelloPayload,
    LocalLLMCompletionMessage,
    PingMessage,
    Platform,
    ProtocolVersionMismatch,
    RecordingFetchMessage,
    RequestRecordingConsentMessage,
    RestoreSessionMessage,
    SessionActivatedMessage,
    SessionActivatedPayload,
    SessionAttachedMessage,
    SessionAttachedPayload,
    SessionRestoredMessage,
    SessionRestoredPayload,
    SessionRevokedMessage,
    StartRecordingMessage,
    StatusMessage,
    StatusPayload,
    StopRecordingMessage,
    describe_parse_failure,
    dump_message,
    make_ack,
    make_error,
    make_pong,
    negotiate_version,
    new_id,
    now_ts,
    parse_message,
    same_major,
)
from .protocol import (
    VERSION as PROTOCOL_VERSION,
)
from .session_grants import RootBinding, RootGrantError, RootGrantSigner

# How often the shim unprompted-emits a STATUS frame (backpressure / health).
# Also emitted on the full → not-full capacity transition edge.
STATUS_INTERVAL_SECONDS: float = 30.0

# WebSocket close codes the platform may use to signal structured failures.
# Mirrors docs/PROTOCOL.md → Close codes. Application-range (4000-4999).
WS_CLOSE_PROTOCOL_VERSION_MISMATCH = 4426
WS_CLOSE_SESSION_TAKEN_OVER = 4427
WS_CLOSE_SESSION_REVOKED = 4428
WS_CLOSE_PLATFORM_SHUTDOWN = 4429

# Close codes the shim treats as fatal: after receiving them on the WS,
# the shim MUST NOT auto-reconnect. Operator must restart. A platform going
# down gracefully (4429) isn't one: it comes back, so the shim reconnects
# with its backoff, as for any other disconnect.
_FATAL_CLOSE_CODES: frozenset[int] = frozenset(
    {
        WS_CLOSE_PROTOCOL_VERSION_MISMATCH,
        WS_CLOSE_SESSION_TAKEN_OVER,
        WS_CLOSE_SESSION_REVOKED,
    }
)

_CLOSE_CODE_LABELS: dict[int, str] = {
    WS_CLOSE_PROTOCOL_VERSION_MISMATCH: "PROTOCOL_VERSION_MISMATCH",
    WS_CLOSE_SESSION_TAKEN_OVER: "SESSION_TAKEN_OVER",
    WS_CLOSE_SESSION_REVOKED: "SESSION_REVOKED",
    WS_CLOSE_PLATFORM_SHUTDOWN: "PLATFORM_SHUTDOWN",
}


# How long cua-driver gets to report its macOS permissions at startup.
PERMISSION_CHECK_SECONDS = 30.0
# How long the machine's start waits for the look for cua-driver before it
# goes on to connect without the rest of the computer-use preflight
# (_preflight_computer_use): a stalled mount on PATH or sys.path holds the
# look as long as it stalls, and file and shell work wait for the connection.
DRIVER_PREFLIGHT_SECONDS = 5.0


class DaemonPreflightError(RuntimeError):
    """Raised when daemon refuses to bind due to a missing OS permission.

    Per docs/COMPUTER_USE.md: when computer_use is requested but its
    Accessibility or Screen Recording grant is missing (macOS), the daemon
    emits a clear audit record and exits with EX_CONFIG (78). launchd's
    KeepAlive: SuccessfulExit=false starts us again after each nonzero exit
    (at most every ten seconds), each start checking again, so the one
    after the grant comes up.

    `permanent` marks a failure no relaunch can fix (a preview that is
    turned on, a missing package, an audit log that fails verification): a
    supervised `start` then stops instead. `transient` marks one that can
    clear on its own (audit storage that is full or not mounted yet, a
    keychain still locked): `start` exits 75 (EX_TEMPFAIL), which systemd and
    launchd both restart.
    """

    def __init__(self, message: str, *, permanent: bool = False, transient: bool = False) -> None:
        if permanent and transient:
            raise ValueError("a preflight failure can't be both permanent and transient")
        super().__init__(message)
        self.permanent = permanent
        self.transient = transient


def _audit_storage_failure(message: str, exc: OSError) -> DaemonPreflightError:
    """A preflight failure for an audit storage error, permanent when no
    restart can fix it and transient otherwise."""
    permanent = exc.errno in _PERMANENT_STORAGE_ERRNOS
    return DaemonPreflightError(
        f"{message} ({exc}); refusing to run without a mandatory audit log",
        permanent=permanent,
        transient=not permanent,
    )


def _allowed_root_failure(root: Path, exc: OSError) -> DaemonPreflightError:
    """A preflight failure for a legacy session's allowed_root that couldn't
    be created: permanent when no restart can fix it (no permission, a
    read-only disk, a file in the way), transient otherwise (a full disk,
    say). On macOS, EPERM is a folder the app may not open yet (Files and
    Folders), which a grant fixes, so it stays a missing permission: launchd
    relaunches the shim until it is granted."""
    message = f"Could not create allowed_root {root}: {exc}"
    if exc.errno == errno.EPERM and platform_info.detect_platform() == "darwin":
        return DaemonPreflightError(message)
    permanent = exc.errno in _PERMANENT_STORAGE_ERRNOS or exc.errno == errno.EEXIST
    return DaemonPreflightError(message, permanent=permanent, transient=not permanent)


# Storage errors a restart can't fix: the path, or what it may do, is wrong,
# or the device under it failed or is gone. Any other error restarts the
# service (exit 75), which systemd does every five seconds without a limit.
_PERMANENT_STORAGE_ERRNOS = frozenset(
    {
        errno.EACCES,
        errno.EPERM,
        errno.EROFS,
        errno.ENOTDIR,
        errno.EISDIR,
        errno.ELOOP,
        errno.ENAMETOOLONG,
        errno.EINVAL,
        errno.EIO,
        errno.ENXIO,
        errno.ENODEV,
    }
)


class TokenRefreshUnavailableError(RuntimeError):
    """The token endpoint couldn't be reached, or answered with a server error,
    408 (it stopped waiting for the request) or 429.

    That says nothing about the credentials, so the reconnect loop retries;
    only a refresh token the platform refuses ends the session for good.
    """


class KeychainUnavailableError(RuntimeError):
    """The OS keychain couldn't be read just now (still locked at login, say).

    Unlike a rejected credential, this can clear on its own, so the reconnect
    loop retries it rather than stopping.
    """


class DaemonAuthenticationError(RuntimeError):
    """Terminal WebSocket authentication or session-authorization failure."""


class KeychainNotUpError(DaemonPreflightError):
    """No keychain to read the sign-in from. The shim couldn't have signed in
    without one, so it is most often one that hasn't started yet: a Secret
    Service the desktop session starts after the autostart service, say.
    The keyring package picks its backend once per process, so this process
    would never find one that starts later: `start` exits 75 (EX_TEMPFAIL)
    for the service manager to start the shim again, as for the audit key."""

    def __init__(self, exc: Exception) -> None:
        super().__init__(
            f"No OS keychain is available yet ({type(exc).__name__}), so the sign-in "
            "can't be read; exiting to be started again once it is. If none is set "
            "up, set one up and run `autogpt-shim auth`.",
            transient=True,
        )


def _keychain_not_up_yet(exc: Exception) -> KeychainNotUpError:
    return KeychainNotUpError(exc)


def _retrieve_outcome(task: asyncio.Future[Any]) -> None:
    """Take a shielded task's exception, so one whose caller stopped waiting
    isn't logged as never retrieved; the caller still waiting gets it too."""
    if not task.cancelled():
        task.exception()


async def _close_quietly(connection: Any) -> None:
    """Close a stopped chat's connection, if it had one; a failure is logged."""
    if connection is None:
        return
    try:
        await connection.close()
    except Exception:
        logger.debug("Closing a stopped chat's connection failed", exc_info=True)


# RFC 6455 §5.5: a control frame carries at most 125 bytes, two of them the
# close code. A longer reason fails to send, and the close code with it.
MAX_CLOSE_REASON_BYTES = 123


def _close_reason_json(fields: dict[str, str]) -> str:
    """`fields` as compact JSON, or just their `error` if that won't fit."""
    reason = json.dumps(fields, separators=(",", ":"))
    if len(reason.encode("utf-8")) <= MAX_CLOSE_REASON_BYTES:
        return reason
    return json.dumps({"error": fields["error"]}, separators=(",", ":"))


class SessionRevokedSendError(RuntimeError):
    """A frame was offered on a connection that SESSION_REVOKED has ended."""


class CredentialBoundConnect(connect):
    def process_redirect(self, exc: Exception) -> Exception:
        if (
            isinstance(exc, websockets.exceptions.InvalidStatus)
            and 300 <= exc.response.status_code < 400
        ):
            return DaemonAuthenticationError(
                "WebSocket redirects are not allowed for saved credentials. "
                "Configure the intended endpoint and run `autogpt-shim auth` again."
            )
        return exc


logger = logging.getLogger(__name__)
# websockets logs every handshake header at DEBUG, Authorization included.
# Connections log through this logger, held at INFO, so a program that
# imports the daemon and turns on DEBUG still never logs the bearer token.
_websocket_logger = logging.getLogger(f"{__name__}.websocket")
_websocket_logger.setLevel(logging.INFO)

MAX_ACTIVE_CHILD_SESSIONS = 8
MAX_SESSION_BINDINGS = 64
CHILD_IDLE_TTL_SECONDS = 10 * 60
CHILD_REAPER_INTERVAL_SECONDS = 30
# How long a session that ends waits for its cancelled requests to stop. A
# side effect already running in a thread can't be stopped, and finishes
# first (handlers._run_to_completion); one that hangs is let go after this,
# so the session still ends and the shim can reconnect.
DISPATCH_DRAIN_SECONDS = 10.0
# How long a chat's request waits for one of the machine's request slots
# before it's refused with SHIM_OVERLOADED. Other chats' requests can hold
# every slot, and one let go after DISPATCH_DRAIN_SECONDS keeps its slot
# until its thread returns, which a hung one never does. Well inside the 30
# seconds the platform waits for most replies.
MACHINE_WORK_WAIT_SECONDS = 10.0
# How long a session has to stay up after its handshake for its drop to
# start the reconnect backoff over. One the platform ends sooner (closing
# with 1011 right after HELLO_ACK while it is overloaded, say) counts as a
# failed connect, so the backoff keeps growing rather than every shim
# reconnecting every few seconds.
STABLE_SESSION_SECONDS = 60.0
# How often a session whose HELLO went without the desktop's display looks
# for it (SessionDisplay reads the systemd manager at most this often too).
DISPLAY_WATCH_SECONDS = 10.0

# When the shim received the request this task serves (loop time), set by
# _dispatch: a computer-use call's deadline counts from it.
_request_received_at: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "request_received_at", default=None
)


class MachineSlots:
    """The machine's request slots, which every chat's session shares.

    A slot stays taken until its work ends, which can be after the session
    that took it has ended (a thread can't be stopped), so the slots outlive
    a control reconnect: the handshake only sets how many there are. Below
    a lowered limit, new work waits for old work to end. Not reentrant."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._taken = 0
        self._waiters: collections.deque[asyncio.Future[None]] = collections.deque()

    def set_limit(self, limit: int) -> None:
        self._limit = limit
        self._wake()

    def locked(self) -> bool:
        return self._taken >= self._limit

    async def acquire(self) -> bool:
        if self._taken < self._limit and not self._waiters:
            self._taken += 1
            return True
        waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._waiters.append(waiter)
        try:
            await waiter
        except BaseException:
            if waiter.done() and not waiter.cancelled():
                # Handed a slot just as the wait ended: pass it on.
                self.release()
            raise
        finally:
            with contextlib.suppress(ValueError):
                self._waiters.remove(waiter)
        return True

    def release(self) -> None:
        self._taken -= 1
        self._wake()

    def _wake(self) -> None:
        while self._waiters and self._taken < self._limit:
            waiter = self._waiters.popleft()
            if not waiter.done():
                self._taken += 1
                waiter.set_result(None)


@dataclass
class _ActiveChild:
    daemon: ShimDaemon
    task: asyncio.Task[None]


@dataclass
class _SessionLockState:
    lock: asyncio.Lock
    users: int = 0


class ShimDaemon:
    def __init__(
        self,
        config: ShimConfig,
        token_store: KeychainTokenStore | Any | None = None,
        audit: AuditWriter | SessionAuditWriter | None = None,
        *,
        manage_audit_lifecycle: bool = True,
        token_refresh_lock: asyncio.Lock | None = None,
        machine_work_semaphore: MachineSlots | asyncio.Semaphore | None = None,
        computer_use_lock: asyncio.Lock | None = None,
    ) -> None:
        self.config = config
        selected_session_id = (config.session_id or "").strip()
        config.session_id = selected_session_id or None
        self.token_store = token_store or KeychainTokenStore(config)
        self._credential_scope = CredentialScope.from_config(config)
        self.audit = audit if audit is not None else self._build_audit_writer(config)
        self._manage_audit_lifecycle = manage_audit_lifecycle
        self._control_mode = not bool(selected_session_id)
        self._token_refresh_lock = token_refresh_lock or asyncio.Lock()
        self._machine_work_semaphore = machine_work_semaphore
        if self._control_mode and self._machine_work_semaphore is None:
            self._machine_work_semaphore = MachineSlots(config.max_concurrent)
        self._computer_use_lock = computer_use_lock or asyncio.Lock()
        if self.audit is not None:
            self.audit.set_machine_id(config.machine_id)
            if selected_session_id:
                self.audit.set_session_id(selected_session_id)
        handler_audit = cast(AuditWriter, self.audit)
        self._file_handler = FileHandler(config, audit=handler_audit)
        self._command_handler = CommandHandler(config, audit=handler_audit)
        self._computer_handler = ComputerUseHandler(
            config, audit=handler_audit, machine_turn=self._computer_use_lock
        )
        self._local_llm_handler = LocalLLMHandler(config, audit=handler_audit)
        self._recording_handler = RecordingHandler(config, audit=handler_audit)
        self._running = False
        self._ws: Any = None
        self._semaphore: asyncio.Semaphore | None = None
        self._local_max_concurrent = config.max_concurrent
        self._local_command_timeout_seconds = config.command_timeout_seconds
        self._local_max_file_size_bytes = config.max_file_size_bytes
        self._granted_capabilities: frozenset[str] = frozenset()
        self._pending_requests = 0
        self._dispatch_tasks: set[asyncio.Task[None]] = set()
        # Each request's handlers.work_refused, set before its task is cancelled.
        self._work_refused: dict[asyncio.Task[None], threading.Event] = {}
        self._shim_start_logged = False
        # Effective negotiated protocol version, populated on HELLO_ACK.
        # `None` before handshake completes.
        self._negotiated_version: str | None = None
        # Set true when a session terminates fatally and the shim MUST NOT
        # auto-reconnect (protocol version mismatch, session revoked, etc.).
        # The reconnect loop checks this and exits cleanly.
        self._disable_reconnect: bool = False
        # The connection a SESSION_REVOKED frame arrived on. Nothing more may
        # be sent on it, and the revocation is audited before the socket is
        # closed, so `_send_frame` checks this instead of waiting for the close.
        self._revoked_ws: Any = None
        # Whether the current connection got through its handshake. A session
        # that was up and then dropped starts the reconnect backoff over.
        # When the current session's handshake completed (loop time), or None
        # while it isn't up.
        self._session_up_at: float | None = None
        # Backpressure / health bookkeeping (#38). _in_flight = tasks
        # currently holding the semaphore; _queue_depth = tasks waiting
        # to acquire it. Maintained explicitly so we don't have to poke
        # at asyncio.Semaphore internals.
        self._in_flight: int = 0
        self._queue_depth: int = 0
        # Bumped each time a session's requests are cancelled. A request let
        # go then, still running, settles none of the counters above when it
        # ends: by then they are another session's.
        self._dispatch_generation = 0
        # Wall-clock when the current session started (set in _run_session).
        # Used for STATUS.uptime_seconds.
        self._session_started_at: float | None = None
        # Background task that emits periodic STATUS frames.
        self._status_task: asyncio.Task[None] | None = None
        self._child_reaper_task: asyncio.Task[None] | None = None
        # Whether the last HELLO left computer use out only for want of the
        # desktop's display, and the task that says HELLO again once it's up.
        self._hello_waits_for_display = False
        self._display_watch_task: asyncio.Task[None] | None = None
        self._connection_id: str | None = None
        self._directory_browser = DirectoryBrowser() if self._control_mode else None
        self._browser_lock = asyncio.Lock()
        self._session_locks: dict[str, _SessionLockState] = {}
        self._session_bindings: OrderedDict[str, RootBinding] = OrderedDict()
        self._child_sessions: OrderedDict[str, _ActiveChild] = OrderedDict()
        self._children_lock = asyncio.Lock()
        self._last_request_at = now_ts()
        # Set on a child session once the reaper has decided to detach it for
        # idleness; from then on it refuses requests (_reap_idle_children).
        self._detaching = False
        # Set once a session has decided to close its connection to say HELLO
        # again: from then on it refuses requests, until its next connection
        # (_hello_again_once_a_display_is_up).
        self._reconnecting = False
        # Set on every chat's session once the machine's connection has so
        # decided: the chat refuses requests until the machine's next session
        # stops it, across a reconnect of the chat's own too.
        self._machine_reconnecting = False
        audit_key = getattr(self.audit, "audit_key", None)
        if not isinstance(audit_key, bytes):
            audit_key = secrets.token_bytes(32)
        self._root_grants = RootGrantSigner(audit_key, config.machine_id)

    @staticmethod
    def _build_audit_writer(config: ShimConfig) -> AuditWriter:
        """Build the mandatory audit writer or fail startup closed, saying
        whether a restart could fix the failure."""
        try:
            key = get_or_create_audit_key(config.audit_log_path)
        except AuditKeyUnavailableError as exc:
            raise DaemonPreflightError(
                f"{exc}; refusing to run without a mandatory audit log, which that key signs",
                transient=True,
            ) from exc
        except AuditKeyStorageError as exc:
            # A full disk clears; a read-only one doesn't: the errno says.
            raise _audit_storage_failure(exc.description, exc.error) from exc
        except Exception as exc:
            # A malformed stored key, or no keychain and no file fallback.
            raise DaemonPreflightError(
                f"The audit key couldn't be loaded ({exc}); refusing to run without a "
                "mandatory audit log",
                permanent=True,
            ) from exc
        try:
            return AuditWriter(
                path=config.audit_log_path,
                audit_key=key,
                machine_id=config.machine_id,
                session_id=config.session_id,
            )
        except AuditLogCorruptError as exc:
            raise DaemonPreflightError(
                f"{exc}: {config.audit_log_path}. Check it with `autogpt-shim audit verify`, "
                "and move it aside to start a new log; refusing to run without a mandatory "
                "audit log",
                permanent=True,
            ) from exc
        except OSError as exc:
            raise _audit_storage_failure(
                f"The audit log at {config.audit_log_path} couldn't be opened", exc
            ) from exc
        except Exception as exc:
            raise DaemonPreflightError(
                "Audit initialization failed; refusing to run without a mandatory audit log"
            ) from exc

    # ── Public API ────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Connect → handshake → pump messages, retrying only transient failures."""
        try:
            # Fail closed on incomplete preview capabilities and missing OS
            # permissions before opening the WebSocket.
            await self._preflight_or_raise()
            if not self._control_mode:
                # A legacy session is jailed to allowed_root, which the shim
                # creates on first run (CROSS_PLATFORM.md), as `doctor` does:
                # in a worker thread, as a folder on a stalled mount would hold
                # the event loop every session shares.
                try:
                    await asyncio.to_thread(
                        self.config.allowed_root.mkdir, parents=True, exist_ok=True
                    )
                except OSError as exc:
                    raise _allowed_root_failure(self.config.allowed_root, exc) from exc
            if self._manage_audit_lifecycle:
                await self._audit_shim_start()
        except BaseException:
            # On macOS the preflight starts cua-driver to check its grants,
            # and the finally below, which stops it, isn't reached from here.
            await self._computer_handler.close()
            raise
        self._running = True
        attempt = 0
        try:
            while self._running:
                self._session_up_at = None
                try:
                    await self._session()
                except asyncio.CancelledError:
                    raise
                except DaemonAuthenticationError as exc:
                    logger.error("Authentication rejected; will not auto-reconnect: %s", exc)
                    self._disable_reconnect = True
                    self._running = False
                    raise
                except KeychainNotUpError:
                    # Retrying here would find the same backend; a new
                    # process looks again.
                    self._running = False
                    raise
                except ProtocolVersionMismatch as exc:
                    # Refuse to retry: a hot-reconnect loop against an
                    # incompatible platform just spams logs and burns rate
                    # limits. Operator must restart the shim after upgrading
                    # one side.
                    logger.error(
                        "Protocol version mismatch (shim=%s platform=%s). "
                        "Will not auto-reconnect; restart the shim after "
                        "upgrading. Hint: %s",
                        exc.shim_max,
                        exc.platform_max,
                        exc.hint,
                    )
                    self._disable_reconnect = True
                    break
                except Exception as exc:
                    if self._disable_reconnect:
                        # run() returns normally next (`start` exits 0, so no
                        # supervisor restarts it), leaving this log as the only
                        # record of why.
                        logger.error("Not reconnecting: %s", exc)
                        break
                    if not self._running:
                        break
                    if self._session_was_stable():
                        # The drop ended a working session, not a run of failed
                        # connects, so reconnect promptly.
                        attempt = 0
                    delay = self._backoff_delay(attempt)
                    logger.warning(
                        "Disconnected (%s). Reconnecting in %.1fs (attempt %d)",
                        exc,
                        delay,
                        attempt + 1,
                    )
                    await asyncio.sleep(delay)
                    attempt += 1
                else:
                    if not self._running or self._disable_reconnect:
                        break
                    # A clean close (1000, 1001) counts as any other: one right
                    # after HELLO_ACK is a failed connect, and the delay grows.
                    if self._session_was_stable():
                        attempt = 0
                    delay = self._backoff_delay(attempt)
                    logger.warning(
                        "Connection closed cleanly. Reconnecting in %.1fs (attempt %d)",
                        delay,
                        attempt + 1,
                    )
                    await asyncio.sleep(delay)
                    attempt += 1
        finally:
            if self._control_mode:
                await self._shutdown_children()
            await self._command_handler.close()
            await self._computer_handler.close()
            if self._manage_audit_lifecycle:
                await self._audit_shim_stop("graceful")

    async def stop(self) -> None:
        self._running = False
        if self._control_mode:
            await self._shutdown_children()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass

    async def _audit_shim_start(self) -> None:
        if self._shim_start_logged:
            return
        try:
            await self.audit.shim_start(self.config.machine_id)
            self._shim_start_logged = True
        except OSError as exc:
            raise _audit_storage_failure(
                "Mandatory audit log rejected its initial SHIM_START record", exc
            ) from exc
        except Exception as exc:
            raise DaemonPreflightError(
                "Mandatory audit log rejected its initial SHIM_START record"
            ) from exc

    async def _audit_shim_stop(self, reason: str) -> None:
        if self.audit is None or not self._shim_start_logged:
            return
        try:
            await self.audit.shim_stop(reason)
        except Exception:
            logger.debug("SHIM_STOP audit emit failed", exc_info=True)

    # ── Connection lifecycle ──────────────────────────────────────────────

    def _session_was_stable(self) -> bool:
        """Whether the session that just ended got through its handshake and
        stayed up STABLE_SESSION_SECONDS: a working session, after which the
        reconnect backoff starts over, rather than a failed connect."""
        return (
            self._session_up_at is not None
            and asyncio.get_running_loop().time() - self._session_up_at >= STABLE_SESSION_SECONDS
        )

    def _backoff_delay(self, attempt: int) -> float:
        """Per docs/PROTOCOL.md Reconnection: 2^attempt * 1s, capped at 60s,
        plus 0-5s jitter."""
        cap = self.config.reconnect_max_delay
        # ldexp scales by 2**attempt with no exponent cap, so even a tiny base
        # reaches the cap (one at 2**32 left a base below cap / 2**32 retrying
        # every few milliseconds). A result past the largest float is past
        # the cap too.
        try:
            base = min(math.ldexp(self.config.reconnect_base_delay, attempt), cap)
        except OverflowError:
            base = cap
        jitter = random.uniform(0, self.config.reconnect_jitter_seconds)
        return base + jitter

    async def _session(self) -> None:
        url = self._connect_url()
        token = await self._get_access_token(refresh_on_fail=False)
        self._ensure_credential_scope()
        if not token:
            raise DaemonAuthenticationError(
                "No credentials for this deployment. Run `autogpt-shim auth` "
                "with the same deployment configuration before starting."
            )
        headers = self._auth_headers(token)
        try:
            async with CredentialBoundConnect(
                url,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=10,
                max_size=MAX_WEBSOCKET_MESSAGE_BYTES,
                logger=_websocket_logger,
            ) as ws:
                await self._run_session(ws)
        except websockets.exceptions.InvalidStatus as exc:
            status_code = getattr(exc.response, "status_code", None)
            if status_code == 403:
                raise DaemonAuthenticationError(
                    "WebSocket session access was denied (HTTP 403); token refresh was "
                    "not attempted. If this computer signed in before the Local PC "
                    "permission existed, run `autogpt-shim auth` again."
                ) from exc
            if status_code != 401:
                raise
            # A 401 may mean the access token expired. Refresh exactly once;
            # authorization denials (403) never rotate credentials.
            logger.info("401 on WebSocket upgrade — refreshing token and retrying once")
            token = await self._get_access_token(
                refresh_on_fail=True,
                rejected_token=token,
            )
            self._ensure_credential_scope()
            if not token:
                raise DaemonAuthenticationError(
                    "WebSocket authentication failed and the single token refresh failed"
                ) from exc
            headers = self._auth_headers(token)
            try:
                async with CredentialBoundConnect(
                    url,
                    additional_headers=headers,
                    ping_interval=20,
                    ping_timeout=10,
                    max_size=MAX_WEBSOCKET_MESSAGE_BYTES,
                    logger=_websocket_logger,
                ) as ws:
                    await self._run_session(ws)
            except websockets.exceptions.InvalidStatus as retry_exc:
                retry_status = getattr(retry_exc.response, "status_code", None)
                if retry_status in (401, 403):
                    raise DaemonAuthenticationError(
                        "WebSocket authentication remained denied after one refresh "
                        f"(HTTP {retry_status})"
                    ) from retry_exc
                raise

    async def _run_session(self, ws) -> None:
        self._ws = ws
        self._session_started_at = now_ts()
        if self._control_mode and self._directory_browser is not None:
            await self._shutdown_children()
            self._directory_browser.reset()
            self._connection_id = None
        # Closed to say HELLO again, the last connection refused requests from
        # then on; on the machine's, the chats it would have cut off have just
        # been stopped.
        self._reconnecting = False
        # Reset per-session counters so reconnects don't carry over.
        self._in_flight = 0
        self._queue_depth = 0
        self._pending_requests = 0
        if self.audit is not None:
            try:
                await self.audit.ws_connected(self._connect_url())
            except Exception:
                logger.debug("WS_CONNECTED audit emit failed", exc_info=True)
        disconnect_reason = "clean_exit"
        try:
            try:
                await self._handshake(ws)
            except Exception as exc:
                disconnect_reason = f"handshake_failed: {exc.__class__.__name__}"
                raise
            self._session_up_at = asyncio.get_running_loop().time()
            # Start the periodic STATUS ticker AFTER handshake — we want
            # max_concurrent from HELLO_ACK before announcing capacity.
            self._status_task = asyncio.create_task(self._status_ticker(ws))
            if self._control_mode:
                self._child_reaper_task = asyncio.create_task(self._child_reaper(ws))
            if self._hello_waits_for_display:
                self._display_watch_task = asyncio.create_task(
                    self._hello_again_once_a_display_is_up(ws)
                )
            try:
                async for raw in ws:
                    await self._on_frame(ws, raw)
                    if self._disable_reconnect:
                        break
            except websockets.exceptions.ConnectionClosed as exc:
                # Translate fatal application close codes into the
                # _disable_reconnect flag so run() doesn't loop on us.
                code = exc.rcvd.code if exc.rcvd is not None else None
                label = _CLOSE_CODE_LABELS.get(code or -1, "unknown")
                disconnect_reason = f"ws_closed_{code}_{label}"
                if code in _FATAL_CLOSE_CODES:
                    logger.warning(
                        "WebSocket closed with fatal code %s (%s); will not auto-reconnect.",
                        code,
                        label,
                    )
                    self._disable_reconnect = True
                    # As for a SESSION_REVOKED frame: the session's requests
                    # and the machine's chats stop now, not once the record
                    # below is written.
                    cancelled = self._cancel_dispatch_requests()
                    chats = self._stop_chats_at_once()
                    await self._audit_session_revoked(
                        reason=label.lower(),
                        source="ws_close_code",
                        close_code=code,
                    )
                    await chats
                    await self._drain_dispatch_tasks(cancelled)
                raise
            except Exception as exc:
                disconnect_reason = f"loop_error: {exc.__class__.__name__}"
                raise
        finally:
            await self._cancel_dispatch_tasks()
            # Cancel the periodic STATUS ticker first so it doesn't fight
            # us for the dying WS.
            await self._stop_status_ticker()
            if self._child_reaper_task is not None:
                self._child_reaper_task.cancel()
                try:
                    await self._child_reaper_task
                except (asyncio.CancelledError, Exception):
                    pass
                self._child_reaper_task = None
            if self._display_watch_task is not None:
                self._display_watch_task.cancel()
                try:
                    await self._display_watch_task
                except (asyncio.CancelledError, Exception):
                    pass
                self._display_watch_task = None
            self._ws = None
            self._session_started_at = None
            if self.audit is not None:
                try:
                    await self.audit.ws_disconnected(disconnect_reason)
                except Exception:
                    logger.debug("WS_DISCONNECTED audit emit failed", exc_info=True)

    async def _stop_status_ticker(self) -> None:
        if self._status_task is not None:
            self._status_task.cancel()
            try:
                await self._status_task
            except (asyncio.CancelledError, Exception):
                pass
            self._status_task = None

    async def _cancel_dispatch_tasks(self) -> None:
        await self._drain_dispatch_tasks(self._cancel_dispatch_requests())

    def _cancel_dispatch_requests(self) -> tuple[asyncio.Task[None], ...]:
        """Cancel the session's requests, without waiting for them. A change
        still waiting for a worker thread is refused first, so it can't start
        before its task takes in the cancel (handlers.work_refused)."""
        tasks = tuple(self._dispatch_tasks)
        for task in tasks:
            refused = self._work_refused.get(task)
            if refused is not None:
                refused.set()
            task.cancel()
        return tasks

    def _forget_work_refused(self, task: asyncio.Task[None]) -> None:
        self._work_refused.pop(task, None)

    async def _drain_dispatch_tasks(self, tasks: tuple[asyncio.Task[None], ...]) -> None:
        """Wait up to DISPATCH_DRAIN_SECONDS for cancelled requests to stop,
        then let go of any still running and clear the session's counters."""
        if tasks:
            _, running = await asyncio.wait(tasks, timeout=DISPATCH_DRAIN_SECONDS)
            if running:
                logger.warning(
                    "%d request(s) still running in a thread that can't be stopped; "
                    "they finish on their own, and their replies are not sent",
                    len(running),
                )
        self._dispatch_generation += 1
        self._dispatch_tasks.clear()
        self._pending_requests = 0
        self._queue_depth = 0
        self._in_flight = 0

    def _connect_url(self) -> str:
        session_id = (self.config.session_id or "").strip()
        base = self.config.derived_ws_url.rstrip("/")
        return f"{base}/{urllib.parse.quote(session_id, safe='')}" if session_id else base

    @staticmethod
    def _auth_headers(token: str | None) -> dict[str, str]:
        if not token:
            return {}
        return {"Authorization": f"Bearer {token}"}

    async def _get_access_token(
        self,
        *,
        refresh_on_fail: bool,
        rejected_token: str | None = None,
    ) -> str | None:
        self._ensure_credential_scope()
        if not refresh_on_fail:
            return await self._read_access_token()
        # In a task of its own, which stopping this session (its task
        # cancelled, _stop_child_session) leaves running: the platform may
        # have rotated the refresh token already, and only the exchange's
        # answer holds the new pair. The task holds the refresh lock until it
        # ends, so no other session sends the used-up token meanwhile.
        refresh = asyncio.ensure_future(self._refresh_access_token(rejected_token))
        refresh.add_done_callback(_retrieve_outcome)
        return await asyncio.shield(refresh)

    async def _refresh_access_token(self, rejected_token: str | None) -> str | None:
        async with self._token_refresh_lock:
            # Another machine/session socket may already have rotated the
            # refresh family while this connection was receiving its 401.
            current_token = await self._read_access_token()
            if (
                rejected_token is not None
                and current_token is not None
                and current_token != rejected_token
            ):
                return current_token
            try:
                from .auth import OAuthFlow

                flow = OAuthFlow(self.config, self.token_store)
                await flow.refresh_token()
                if self.audit is not None:
                    try:
                        await self.audit.token_refreshed()
                    except Exception:
                        logger.debug("TOKEN_REFRESHED audit emit failed", exc_info=True)
                return await self._read_access_token()
            except (DaemonAuthenticationError, KeychainUnavailableError, KeychainNotUpError):
                # The re-read's own errors: a keychain gone since the refresh
                # stored the new pair is waited for, not a sign-in to redo.
                raise
            except CredentialScopeError as exc:
                raise DaemonAuthenticationError(str(exc)) from None
            except (keyring.errors.NoKeyringError, keyring.errors.InitError) as exc:
                raise self._keychain_not_up(exc) from None
            except keyring.errors.KeyringError as exc:
                # Reading the refresh token or storing the new pair failed: a
                # keychain that locked, say, which can clear on its own.
                raise KeychainUnavailableError(
                    f"The OS keychain could not be used during a token refresh "
                    f"({type(exc).__name__}); it may be locked. Retrying."
                ) from None
            except httpx.TransportError as exc:
                raise TokenRefreshUnavailableError(
                    f"The token endpoint could not be reached ({type(exc).__name__}); retrying."
                ) from None
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                # 408: the server gave up waiting for the request, so it never
                # read the refresh token.
                if status in (408, 429) or status >= 500:
                    raise TokenRefreshUnavailableError(
                        f"The token endpoint answered {status}; retrying."
                    ) from None
                logger.warning(
                    "Token refresh was refused (HTTP %s). Run `autogpt-shim auth` again.", status
                )
                return None
            except Exception as exc:
                if self._holding_unsaved_tokens():
                    # Storing a pair the exchange returned failed some other
                    # way (Windows' CredWrite raises its own error).
                    raise KeychainUnavailableError(
                        f"The OS keychain could not store the refreshed sign-in "
                        f"({type(exc).__name__}); it is kept for the next try. Retrying."
                    ) from None
                logger.warning(
                    "Token refresh failed. Check the deployment configuration and OS keychain, "
                    "then run `autogpt-shim auth` again."
                )
                return None

    def _holding_unsaved_tokens(self) -> bool:
        """Whether a refresh exchange went through but its new pair isn't
        stored yet (OAuthFlow.refresh_token). The keychain then holds a
        refresh token the platform has used up, so a restart, or a stop that
        asks for `autogpt-shim auth`, would lose the sign-in: the refresh is
        retried in this process instead."""
        return getattr(self.token_store, "unsaved_refresh", None) is not None

    def _keychain_not_up(self, exc: Exception) -> KeychainNotUpError | KeychainUnavailableError:
        """KeychainNotUpError, for a new process to look for the keychain
        again, unless a refreshed pair waits to be stored, which a new
        process wouldn't have."""
        if self._holding_unsaved_tokens():
            return KeychainUnavailableError(
                f"The OS keychain could not be used ({type(exc).__name__}) while a refreshed "
                "sign-in waits to be stored; it is kept for the next try. Retrying."
            )
        return _keychain_not_up_yet(exc)

    async def _read_access_token(self) -> str | None:
        self._ensure_credential_scope()
        try:
            token = await self.token_store.get_access_token()
        except CredentialScopeError as exc:
            raise DaemonAuthenticationError(str(exc)) from None
        except (keyring.errors.NoKeyringError, keyring.errors.InitError) as exc:
            raise self._keychain_not_up(exc) from None
        except Exception as exc:
            raise KeychainUnavailableError(
                f"The OS keychain could not be read ({type(exc).__name__}); "
                "it may still be locked. Retrying."
            ) from None
        self._ensure_credential_scope()
        return token

    def _ensure_credential_scope(self) -> None:
        try:
            self._credential_scope.ensure_matches(self.config)
            if (
                isinstance(self.token_store, KeychainTokenStore)
                and self.token_store.scope != self._credential_scope
            ):
                raise CredentialScopeError(
                    "Stored credentials belong to different OAuth configuration. "
                    "Run `autogpt-shim auth` with the intended deployment configuration."
                )
        except CredentialScopeError as exc:
            raise DaemonAuthenticationError(str(exc)) from None

    # ── Handshake ─────────────────────────────────────────────────────────

    @staticmethod
    async def _close_failed_handshake(ws, code: int, reason: str) -> None:
        """Close after a failed handshake. The caller raises why it failed
        next, which a close that fails must not replace."""
        try:
            await ws.close(code=code, reason=reason)
        except Exception:
            logger.debug("ws.close after a failed handshake failed", exc_info=True)

    async def _handshake(self, ws) -> HelloAckMessage:
        hello = await self._build_hello()
        raw_hello = dump_message(hello)
        if len(raw_hello.encode("utf-8")) > MAX_WEBSOCKET_MESSAGE_BYTES:
            self._disable_reconnect = True
            await self._close_failed_handshake(ws, 4400, "HELLO payload too large")
            raise RuntimeError("HELLO payload exceeds the WebSocket message limit")
        await ws.send(raw_hello)
        try:
            raw_ack = await asyncio.wait_for(ws.recv(), timeout=10.0)
        except websockets.exceptions.ConnectionClosed as exc:
            received = getattr(exc, "rcvd", None)
            code = getattr(received, "code", None) or getattr(exc, "code", None)
            if code in _FATAL_CLOSE_CODES:
                self._disable_reconnect = True
            raise
        try:
            ack = parse_message(raw_ack)
        except ValueError as exc:
            # pydantic's ValidationError, like a JSON or UTF-8 error, is a
            # ValueError. A control daemon fails terminally on an ACK without
            # its required fields (PROTOCOL.md → HELLO_ACK). pydantic's text
            # quotes the frame's values, so neither error carries it along.
            failure = f"Malformed HELLO_ACK: {describe_parse_failure(exc)}"
            if not self._control_mode:
                raise ValueError(failure) from None
            self._disable_reconnect = True
            await self._close_failed_handshake(ws, 4400, "Malformed HELLO_ACK")
            raise RuntimeError(failure) from None
        if not isinstance(ack, HelloAckMessage):
            raise RuntimeError(f"Expected HELLO_ACK, got {type(ack).__name__}")
        if ack.id != hello.id:
            self._disable_reconnect = True
            await self._close_failed_handshake(ws, 4400, "HELLO_ACK correlation mismatch")
            raise RuntimeError("HELLO_ACK id does not match HELLO id")
        # Honor the server's negotiated limits.
        payload = ack.payload
        expected_session_id = (self.config.session_id or "").strip()
        if expected_session_id and payload.session_id != expected_session_id:
            await self._close_failed_handshake(ws, 4403, "HELLO_ACK session mismatch")
            raise DaemonAuthenticationError(
                "HELLO_ACK session does not match the locally selected session"
            )
        if not expected_session_id and payload.session_id not in (None, ""):
            await self._close_failed_handshake(ws, 4403, "HELLO_ACK control mismatch")
            raise DaemonAuthenticationError(
                "HELLO_ACK unexpectedly bound the machine control connection to a session"
            )
        self._connection_id = payload.connection_id
        advertised_capabilities = set(hello.payload.capabilities)
        unexpected_capabilities = set(payload.granted_capabilities) - advertised_capabilities
        if unexpected_capabilities:
            self._disable_reconnect = True
            await self._close_failed_handshake(
                ws, 4400, "HELLO_ACK granted unadvertised capability"
            )
            raise RuntimeError(
                "HELLO_ACK granted capabilities the shim did not advertise: "
                + ", ".join(sorted(unexpected_capabilities))
            )
        # Negotiate wire-protocol version BEFORE applying any other limits.
        # If majors disagree we tear the WS down with 4426 and surface
        # ProtocolVersionMismatch so the run() loop disables reconnect.
        try:
            self._negotiated_version = negotiate_version(
                shim_max=PROTOCOL_VERSION,
                platform_max=payload.protocol_version,
            )
        except ProtocolVersionMismatch as exc:
            close_reason = _close_reason_json(exc.to_close_reason())
            await self._close_failed_handshake(ws, WS_CLOSE_PROTOCOL_VERSION_MISMATCH, close_reason)
            raise
        except ValueError as exc:
            self._disable_reconnect = True
            await self._close_failed_handshake(
                ws, WS_CLOSE_PROTOCOL_VERSION_MISMATCH, "Invalid platform protocol version"
            )
            raise RuntimeError("HELLO_ACK contained an invalid protocol version") from exc
        # The model fills in an omitted protocol_version as this shim's own
        # version, and an omitted session_id as null, so whether the platform
        # named them is asked separately. Every 2.x platform speaks the
        # machine-control protocol.
        if self._control_mode and (
            "session_id" not in payload.model_fields_set
            or "protocol_version" not in payload.model_fields_set
            or "directory_browse" not in payload.granted_capabilities
            or not payload.connection_id
        ):
            self._disable_reconnect = True
            await self._close_failed_handshake(
                ws,
                WS_CLOSE_PROTOCOL_VERSION_MISMATCH,
                "Machine control requires a protocol version and directory_browse",
            )
            raise RuntimeError("Platform does not support the persistent machine-control protocol")
        self.config.max_concurrent = min(
            self._local_max_concurrent,
            payload.max_concurrent,
        )
        self.config.command_timeout_seconds = min(
            self._local_command_timeout_seconds,
            payload.command_timeout_seconds,
        )
        self.config.max_file_size_bytes = min(
            self._local_max_file_size_bytes,
            payload.max_file_size_bytes,
        )
        self._granted_capabilities = frozenset(payload.granted_capabilities)
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)
        if self._control_mode:
            # The same slots: work the old connection's sessions left running
            # keeps its slots until it ends.
            if isinstance(self._machine_work_semaphore, MachineSlots):
                self._machine_work_semaphore.set_limit(self.config.max_concurrent)
            else:
                self._machine_work_semaphore = MachineSlots(self.config.max_concurrent)
        # Tell the AuditWriter about the now-known session_id so subsequent
        # records get correctly attributed.
        if self.audit is not None:
            if payload.session_id:
                self.audit.set_session_id(payload.session_id)
            try:
                await self.audit.config_reloaded(payload.granted_capabilities)
            except Exception:
                logger.debug("CONFIG_RELOADED audit emit failed", exc_info=True)
        logger.info(
            "Connected. session=%s granted_capabilities=%s max_concurrent=%d "
            "timeout=%ds protocol=%s",
            payload.session_id or "control",
            payload.granted_capabilities,
            self.config.max_concurrent,
            self.config.command_timeout_seconds,
            self._negotiated_version,
        )
        return ack

    async def _build_hello(self) -> HelloMessage:
        cfg = self.config
        # The driver is looked for first, through the bridge's one look at a
        # time and within HELLO's wait (driver_found_for_hello): one looked
        # for in a thread of HELLO's own, waited for to the end, kept a
        # stalled mount on PATH from letting HELLO out at all.
        driver_found = None
        if cfg.enable_computer_use:
            driver_found = await self._computer_handler.driver_found_for_hello()
            # On macOS, not before a session's check of the grants computer
            # use needs has passed: a start that went on while its look for
            # the driver stalled checked none (_preflight_computer_use).
            if (
                driver_found
                and platform_info.detect_platform() == "darwin"
                and not self._computer_handler.driver.permissions_granted
            ):
                driver_found = False
        # Off the event loop, which carries every other session: the X
        # display probe behind both can take seconds.
        caps = await asyncio.to_thread(
            platform_info.detect_capabilities,
            enable_shell=cfg.enable_shell,
            enable_computer_use=cfg.enable_computer_use,
            enable_local_llm=cfg.enable_local_llm,
            enable_hardware=cfg.enable_hardware,
            enable_recording=cfg.enable_recording,
            cua_driver_path=cfg.cua_driver_path,
            driver_found=driver_found,
        )
        # HELLO is the only place computer use is offered, so a session that
        # left it out for want of the desktop's display (the systemd service
        # starts at login, before the desktop exports it) says HELLO again
        # once it's up (_hello_again_once_a_display_is_up).
        self._hello_waits_for_display = bool(
            cfg.enable_computer_use
            and "computer_use" not in caps
            and await asyncio.to_thread(platform_info.computer_use_waits_for_a_display)
        )
        if self._control_mode:
            caps.append("directory_browse")
        computer_use_driver = None
        computer_use_tools: list[str] = []
        screen = None
        if "computer_use" in caps:
            # Advertised only once cua-driver has started and listed its tools.
            try:
                snapshot = await self._computer_handler.hello_snapshot()
                computer_use_driver = snapshot.info
                computer_use_tools = sorted(tool["name"] for tool in snapshot.tools)
            except DriverUnavailable as exc:
                logger.warning("Not advertising computer_use: %s", exc.message)
                caps = [cap for cap in caps if cap != "computer_use"]
            else:
                screen = await self._computer_handler.screen_size()
        # Local-model routing is not wired into the platform service yet.
        local_llm_models: list[str] = []
        caps = [cap for cap in caps if cap != "local_llm"]
        # Recording advertisement: only when the capability is on. Channels are
        # what this shim can offer; routes are what this machine can interpret
        # with right now, so the platform can gate (§6).
        recording_channels: list[str] = []
        recording_routes: list[str] = []
        if "recording" in caps:
            native_channels = set(platform_info.available_recording_channels())
            recording_channels = [
                channel for channel in cfg.recording_channels if channel in native_channels
            ]
            if recording_channels:
                recording_routes = self._available_recording_routes(
                    channels=recording_channels,
                    local_llm_models=local_llm_models,
                )
            else:
                caps = [capability for capability in caps if capability != "recording"]
        self._recording_handler.set_available_channels(recording_channels)
        self._recording_handler.set_local_llm_models(local_llm_models)
        payload = HelloPayload(
            shim_version=__import__("autogpt_local_executor").__version__,
            machine_id=cfg.machine_id,
            display_name=cfg.display_name,
            platform=Platform(platform_info.detect_platform()),
            arch=Arch(platform_info.detect_arch()),
            screen_resolution=screen,
            capabilities=caps,
            allowed_root=None if self._control_mode else str(cfg.allowed_root),
            local_llm_models=local_llm_models,
            hardware_devices=[],
            computer_use_driver=computer_use_driver,
            computer_use_tools=computer_use_tools,
            recording_channels=recording_channels,  # type: ignore[arg-type]
            recording_routes=recording_routes,  # type: ignore[arg-type]
            protocol_version=PROTOCOL_VERSION,
        )
        return HelloMessage(id=new_id(), ts=now_ts(), payload=payload)

    @staticmethod
    def _available_recording_routes(
        *,
        channels: list[str],
        local_llm_models: list[str],
    ) -> list[str]:
        """Interpretation routes this machine can offer (see §3.1 probes).

        screenshots_to_cloud is always offered (the consent-gated fallback);
        extract_then_cloud when structured channels or OCR are present;
        local_vlm when a capable local vision model is listed. The platform
        gates its per-recording choice against this set.
        """
        from .recording.route import (
            _local_vlm_present,
            _ocr_available,
            _structured_channels_present,
        )

        routes: list[str] = ["screenshots_to_cloud"]
        if _structured_channels_present(channels) or _ocr_available():
            routes.insert(0, "extract_then_cloud")
        if _local_vlm_present(local_llm_models):
            routes.append("local_vlm")
        return routes

    # ── Frame dispatch ────────────────────────────────────────────────────

    async def _on_frame(self, ws, raw: str | bytes) -> None:
        # A computer-use call's deadline counts from here, when the frame
        # arrived: its task can start later, after other work that kept the
        # event loop busy.
        received_at = asyncio.get_running_loop().time()
        try:
            msg = parse_message(raw)
        except (ValidationError, ValueError) as exc:
            # Not pydantic's text, which quotes what the frame held: a
            # malformed computer tool call's typed text, say.
            logger.warning("Failed to parse inbound frame: %s", describe_parse_failure(exc))
            return
        if self._negotiated_version is not None and not same_major(
            msg.version, self._negotiated_version
        ):
            # PROTOCOL.md: a frame from another major is dropped and logged loudly.
            logger.error(
                "Dropped a %s frame for protocol %r; this session negotiated %s",
                type(msg).__name__,
                msg.version,
                self._negotiated_version,
            )
            return

        # PING is handled inline so a saturated semaphore can't starve keepalive.
        if isinstance(msg, PingMessage):
            try:
                await self._send_frame(ws, make_pong(msg.id))
            except Exception:
                logger.debug("Failed to send PONG", exc_info=True)
            return
        # Keepalive doesn't count: an idle child session must still be reaped.
        self._last_request_at = now_ts()

        # SESSION_REVOKED is a one-shot lifecycle event. We audit it, send
        # nothing further, gracefully close, and disable auto-reconnect.
        if isinstance(msg, SessionRevokedMessage):
            await self._handle_session_revoked(ws, msg)
            return

        if self._semaphore is None:
            # We somehow got a request before HELLO_ACK; be defensive.
            self._semaphore = asyncio.Semaphore(self.config.max_concurrent)

        if self._detaching or self._reconnecting or self._machine_reconnecting:
            # Being detached for idleness, or reconnecting to say HELLO again:
            # a request started now would be cut off by the stop. Nothing ran,
            # so it can be sent again once the session is back.
            reason, why = (
                ("detaching", "The session is being detached after being idle")
                if self._detaching
                else ("reconnecting", "The local executor is reconnecting to offer computer use")
            )
            response = make_error(
                msg.id,
                ErrorCode.SHIM_OVERLOADED,
                f"{why}; retry in a moment.",
                details={"reason": reason},
            )
            response.pending_capacity = 0
            await self._send_frame(ws, response)
            return

        if self._pending_requests >= self.config.max_concurrent:
            response = make_error(
                msg.id,
                ErrorCode.SHIM_OVERLOADED,
                "The local executor is at its concurrent request limit.",
                details={"max_concurrent": self.config.max_concurrent},
            )
            response.pending_capacity = 0
            await self._send_frame(ws, response)
            return

        self._pending_requests += 1
        self._queue_depth += 1
        refused = threading.Event()
        task = asyncio.create_task(
            self._dispatch(ws, msg, prequeued=True, received_at=received_at, refused=refused)
        )
        self._dispatch_tasks.add(task)
        self._work_refused[task] = refused
        task.add_done_callback(self._dispatch_tasks.discard)
        task.add_done_callback(self._forget_work_refused)

    async def _handle_session_revoked(self, ws, msg: SessionRevokedMessage) -> None:
        """Per PROTOCOL.md → Session ownership: log, stop sending, close,
        and don't auto-reconnect."""
        # First, before anything awaits: requests still in flight can finish
        # while the audit write below runs in a thread, and their responses
        # must not go out.
        self._revoked_ws = ws
        self._disable_reconnect = True
        # Nor may the revoked session's work go on: stop what is running and
        # what is queued behind it, and the machine's chats with it. A side
        # effect already running in a thread can't be stopped and may never
        # finish, so the record and the close come before any wait for it.
        cancelled = self._cancel_dispatch_requests()
        chats = self._stop_chats_at_once()
        reason = msg.payload.reason
        new_machine = msg.payload.new_shim_machine_id
        logger.warning(
            "SESSION_REVOKED received (reason=%s, new_shim_machine_id=%s); "
            "closing connection without retry.",
            reason,
            new_machine,
        )
        # The record and the close don't wait for each other: an audit
        # writer held up (its lock taken by `audit verify` on a large log, a
        # slow disk) left the revoked connection open until it was done.
        record = asyncio.ensure_future(
            self._audit_session_revoked(
                reason=reason,
                source="frame",
                new_shim_machine_id=new_machine,
            )
        )
        try:
            # We send no further protocol frames; just close cleanly.
            await ws.close(code=WS_CLOSE_SESSION_REVOKED, reason="session_revoked")
        except Exception:
            logger.debug("ws.close after SESSION_REVOKED failed", exc_info=True)
        await self._stop_status_ticker()
        await record
        await chats
        await self._drain_dispatch_tasks(cancelled)

    def _stop_chats_at_once(self) -> asyncio.Future[None]:
        """Stop the machine's chats (none for a chat's own session) when its
        session is revoked: cancel them now, and close and wait for them in a
        task. run() stops them only after the revocation's audit record,
        which a held-up audit writer (its lock taken by `audit verify` on a
        large log, say) can keep from being written, and they ran their
        commands meanwhile."""
        stopping = asyncio.ensure_future(self._close_and_drain_children(self._cancel_children()))
        stopping.add_done_callback(_retrieve_outcome)
        return stopping

    async def _audit_session_revoked(
        self,
        *,
        reason: str,
        source: str,
        close_code: int | None = None,
        new_shim_machine_id: str | None = None,
    ) -> None:
        """Emit a structured audit record for session revocation. Best-effort —
        a failure here must not block close."""
        if self.audit is None:
            return
        details: dict[str, Any] = {
            "reason": reason,
            "source": source,
        }
        if close_code is not None:
            details["close_code"] = close_code
        if new_shim_machine_id is not None:
            details["new_shim_machine_id"] = new_shim_machine_id
        try:
            await self.audit.write(
                "SESSION_REVOKED",
                request_id=None,
                details=details,
            )
        except Exception:
            logger.debug("SESSION_REVOKED audit emit failed", exc_info=True)

    # ── Backpressure / health (STATUS frame, #38) ───────────────────────

    def _build_status_message(self) -> StatusMessage:
        """Snapshot the shim's current backpressure / health state."""
        audit_bytes = 0
        try:
            if self.config.audit_log_path.is_file():
                audit_bytes = self.config.audit_log_path.stat().st_size
        except OSError:
            # Best-effort — a missing/unreadable audit log shouldn't break
            # health reporting.
            audit_bytes = 0
        uptime = 0.0
        if self._session_started_at is not None:
            uptime = max(now_ts() - self._session_started_at, 0.0)
        payload = StatusPayload(
            in_flight=self._in_flight,
            max_concurrent=self.config.max_concurrent,
            queue_depth=self._queue_depth,
            audit_log_bytes=audit_bytes,
            uptime_seconds=uptime,
        )
        msg = StatusMessage(id=new_id(), ts=now_ts(), payload=payload)
        # STATUS itself carries pending_capacity too — it IS a backpressure
        # signal, so be explicit.
        msg.pending_capacity = self._available_capacity()
        return msg

    async def _emit_status(self, ws, *, source: str) -> None:
        """Send a STATUS frame on the WS, swallowing transient send errors.

        `source` is a debug label (capacity_edge, periodic) — not on the wire,
        just for logging.
        """
        try:
            msg = self._build_status_message()
            await self._send_frame(ws, msg)
            logger.debug("STATUS emitted (%s): %s", source, msg.payload)
        except Exception:
            logger.debug("STATUS emit failed (%s)", source, exc_info=True)

    async def _status_ticker(self, ws) -> None:
        """Background task: emit STATUS every STATUS_INTERVAL_SECONDS. The
        loop exits when the WS is gone or the daemon is told to stop.
        """
        try:
            while self._running and self._ws is ws:
                await asyncio.sleep(STATUS_INTERVAL_SECONDS)
                if self._ws is not ws:
                    return
                await self._emit_status(ws, source="periodic")
        except asyncio.CancelledError:
            raise

    async def _child_reaper(self, ws: Any) -> None:
        try:
            while self._running and self._ws is ws:
                await asyncio.sleep(CHILD_REAPER_INTERVAL_SECONDS)
                await self._reap_idle_children()
        except asyncio.CancelledError:
            raise

    async def _hello_again_once_a_display_is_up(self, ws: Any) -> None:
        """Close this connection, to say HELLO again, once the desktop has
        exported the display its HELLO went without and nothing is under
        way: the reconnect's HELLO offers computer use (PROTOCOL.md,
        Reconnection). Work in progress is waited for: a request or a
        recording, and for the machine's connection, whose next session
        stops every chat's, a chat's too. Once it is decided, no new work
        starts: the close and the reconnect's backoff take seconds."""
        while self._running and self._ws is ws:
            await asyncio.sleep(DISPLAY_WATCH_SECONDS)
            if not await asyncio.to_thread(
                platform_info.can_do_computer_use, self.config.cua_driver_path
            ):
                continue
            if self._ws is not ws or not self._quiet_enough_for_a_new_hello():
                continue
            # Decided, with nothing awaited since the check: this session and
            # its chats' refuse requests from now on, so none starts only for
            # the next connection to stop it with them.
            self._reconnecting = True
            for active in self._child_sessions.values():
                active.daemon._machine_reconnecting = True
            logger.info("The desktop's display is up; reconnecting so HELLO offers computer use")
            await ws.close(code=1000, reason="Computer use is available now")
            return

    def _quiet_enough_for_a_new_hello(self) -> bool:
        now = now_ts()
        if not self._idle_since(now):
            return False
        return not self._control_mode or all(
            active.task.done() or active.daemon._idle_since(now)
            for active in self._child_sessions.values()
        )

    async def _reap_idle_children(self) -> None:
        """Detach each child session idle for CHILD_IDLE_TTL_SECONDS."""
        cutoff = now_ts() - CHILD_IDLE_TTL_SECONDS
        for session_id, active in tuple(self._child_sessions.items()):
            child = active.daemon
            if active.task.done() or not child._idle_since(cutoff):
                continue
            async with self._session_lock(session_id):
                current = self._child_sessions.get(session_id)
                if current is not active or not child._idle_since(cutoff):
                    continue
                # Decided, with nothing awaited since the check: the child now
                # refuses requests, so none starts while the detach is audited
                # (in a thread) only to be cut off by the stop.
                child._detaching = True
                try:
                    await self._write_control_audit(
                        "IDLE_SESSION_DETACH",
                        request_id=None,
                        session_id=session_id,
                        details={"idle_seconds": now_ts() - child._last_request_at},
                    )
                except asyncio.CancelledError:
                    # The control connection went (the reaper is cancelled
                    # with it) before the detach: the child keeps serving
                    # until the reconnect stops it.
                    child._detaching = False
                    raise
                except Exception:
                    child._detaching = False
                    logger.exception(
                        "Could not audit idle detach for session %s",
                        session_id,
                    )
                    continue
                await self._stop_child_session(session_id)

    def _idle_since(self, cutoff: float) -> bool:
        """Whether this child session has had no request since `cutoff`, has
        none queued or running (a command may outlast the idle limit), and
        isn't recording."""
        return (
            self._last_request_at <= cutoff
            and self._pending_requests == 0
            and not self._dispatch_tasks
            and self._recording_handler._active_recording_id is None
        )

    async def _dispatch(
        self,
        ws,
        msg: Any,
        *,
        prequeued: bool = False,
        received_at: float | None = None,
        refused: threading.Event | None = None,
    ) -> None:
        assert self._semaphore is not None
        if refused is not None:
            handlers.work_refused.set(refused)
        # A computer-use call's deadline counts from when the frame arrived
        # (_on_frame), else from here, through every wait. Each dispatch runs
        # in its own task, so this is the request's own.
        _request_received_at.set(
            asyncio.get_running_loop().time() if received_at is None else received_at
        )
        generation = self._dispatch_generation

        def counted() -> bool:
            """Whether this request still counts in the session's
            bookkeeping. One let go when its session ended doesn't
            (_drain_dispatch_tasks)."""
            return generation == self._dispatch_generation

        # Track queue depth: incremented now, decremented exactly once when
        # we either acquire the semaphore (transition to in_flight) or bail
        # out before acquiring (cancellation / error in waiter).
        if not prequeued:
            self._pending_requests += 1
            self._queue_depth += 1
        queue_owed = True
        in_flight_owed = False
        was_full_after_decrement = False
        response = None

        # Streaming ops (LOCAL_LLM_COMPLETION) emit intermediate frames
        # via this callback before the terminal RESPONSE flows through the
        # standard one-frame-per-request path below. CHUNK frames do NOT
        # carry pending_capacity (we're still mid-request) and aren't
        # treated as responses for the backpressure accounting.
        async def _stream_send(frame: Any) -> None:
            if not await self._send_frame(ws, frame):
                raise ValueError("Streaming response exceeded its size limit")

        try:
            try:
                async with self._semaphore:
                    # We've acquired — move the credit from queue → in_flight.
                    if counted():
                        self._queue_depth -= 1
                        self._in_flight += 1
                    queue_owed = False
                    in_flight_owed = True
                    try:
                        # A computer tool call takes no machine slot: the
                        # computer's calls take turns at it, each waiting no
                        # longer than its own deadline, and calls waiting
                        # their turn while a long one ran held every slot,
                        # so other chats' file and shell requests got
                        # machine_busy.
                        if (
                            self._machine_work_semaphore is not None
                            and not self._control_mode
                            and not isinstance(msg, ComputerToolCallMessage)
                        ):
                            response = await self._handle_in_a_machine_slot(
                                self._machine_work_semaphore, msg, _stream_send
                            )
                        else:
                            response = await self._handle(msg, send=_stream_send)
                    finally:
                        if counted():
                            was_full_after_decrement = (
                                self._pending_requests >= self.config.max_concurrent
                                or self._in_flight == self.config.max_concurrent
                            )
                            self._in_flight -= 1
                        in_flight_owed = False
            except SessionRevokedSendError:
                logger.debug("%s stopped: the session was revoked", type(msg).__name__)
            except Exception as exc:
                logger.exception("Handler crashed for %s", type(msg).__name__)
                response = make_error(
                    msg.id if hasattr(msg, "id") else new_id(),
                    ErrorCode.INTERNAL_ERROR,
                    str(exc),
                )
        finally:
            # Belt-and-braces: if either counter is still owed (e.g. we got
            # cancelled mid-acquire), make it whole. Without this a crashed
            # waiter would skew queue_depth forever.
            if counted():
                if queue_owed:
                    self._queue_depth -= 1
                if in_flight_owed:
                    self._in_flight -= 1
                self._pending_requests = max(self._pending_requests - 1, 0)

        if response is None:
            return

        # Stamp pending_capacity on the response envelope so the platform can
        # throttle issuance before we trip SHIM_OVERLOADED. Compute AFTER
        # decrementing _in_flight so the number reflects post-response slots.
        try:
            setattr(response, "pending_capacity", self._available_capacity())
        except Exception:
            # _Envelope subclasses all carry the field via the base; this
            # is just paranoia for any future hand-rolled response model.
            logger.debug("could not stamp pending_capacity", exc_info=True)

        try:
            await self._send_frame(ws, response)
        except Exception:
            logger.debug("Failed to send response", exc_info=True)

        # If we just freed a slot from a fully-saturated state, push an
        # unsolicited STATUS frame so the platform's throttle releases
        # promptly (don't wait for the 30s tick).
        if was_full_after_decrement:
            await self._emit_status(ws, source="capacity_edge")

    async def _handle_in_a_machine_slot(
        self, slots: MachineSlots | asyncio.Semaphore, msg: Any, send: Any
    ) -> BaseModel | None:
        """Handle `msg` in one of the machine's request slots, shared by every
        chat, or refuse it when none frees up in MACHINE_WORK_WAIT_SECONDS.
        Only the wait is bounded: a slot taken is held until the work ends,
        since a thread already running it can't be stopped."""
        give_up = asyncio.get_running_loop().time() + MACHINE_WORK_WAIT_SECONDS
        try:
            async with asyncio.timeout_at(give_up):
                await slots.acquire()
        except TimeoutError:
            return make_error(
                msg.id,
                ErrorCode.SHIM_OVERLOADED,
                "Every request slot on this computer is busy with other chats' work; "
                "retry in a moment.",
                details={"reason": "machine_busy"},
            )
        try:
            return await self._handle(msg, send=send)
        finally:
            slots.release()

    def _available_capacity(self) -> int:
        """Free request slots right now. Floored at 0."""
        occupied = max(
            self._pending_requests,
            self._in_flight + self._queue_depth,
        )
        return max(self.config.max_concurrent - occupied, 0)

    async def _send_frame(self, ws, frame: BaseModel) -> bool:
        if ws is self._revoked_ws:
            raise SessionRevokedSendError(
                "SESSION_REVOKED: nothing more is sent on this connection"
            )
        raw = dump_message(frame)
        if len(raw.encode("utf-8")) > MAX_WEBSOCKET_MESSAGE_BYTES:
            fallback = make_error(
                getattr(frame, "id", new_id()),
                ErrorCode.FILE_TOO_LARGE,
                "The local executor response exceeded the WebSocket message limit.",
            )
            fallback.pending_capacity = self._available_capacity()
            await ws.send(dump_message(fallback))
            return False
        await ws.send(raw)
        return True

    _COMPUTER_USE_MESSAGE_TYPES = (
        ComputerToolsRequestMessage,
        ComputerToolCallMessage,
    )

    _FILE_MESSAGE_TYPES = (
        FileReadMessage,
        FileWriteMessage,
        FileStatMessage,
        FileListMessage,
        FileDeleteMessage,
        FileMoveMessage,
    )

    _RECORDING_MESSAGE_TYPES = (
        RequestRecordingConsentMessage,
        StartRecordingMessage,
        StopRecordingMessage,
        ApplyRecordingReviewMessage,
        RecordingFetchMessage,
    )

    _CONTROL_MESSAGE_TYPES = (
        DirectoryListRequestMessage,
        AttachSessionMessage,
        ActivateSessionMessage,
        RestoreSessionMessage,
        DetachSessionMessage,
    )

    def _required_capability(self, msg: Any) -> str | None:
        if isinstance(msg, ExecuteCommandMessage):
            return "shell"
        if isinstance(msg, self._FILE_MESSAGE_TYPES):
            return "files"
        if isinstance(msg, self._COMPUTER_USE_MESSAGE_TYPES):
            return "computer_use"
        if isinstance(msg, LocalLLMCompletionMessage):
            return "local_llm"
        if isinstance(msg, self._RECORDING_MESSAGE_TYPES):
            return "recording"
        if isinstance(msg, self._CONTROL_MESSAGE_TYPES):
            return "directory_browse"
        return None

    async def _handle(self, msg: Any, *, send: Any | None = None) -> BaseModel | None:
        is_control_message = isinstance(msg, self._CONTROL_MESSAGE_TYPES)
        if self._control_mode and not is_control_message:
            return make_error(
                msg.id,
                ErrorCode.FEATURE_NOT_SUPPORTED,
                "Execution operations require an activated per-session data connection.",
            )
        required_capability = self._required_capability(msg)
        if (
            required_capability is not None
            and required_capability not in self._granted_capabilities
        ):
            return make_error(
                msg.id,
                ErrorCode.CAPABILITY_NOT_GRANTED,
                f"The platform did not grant the {required_capability!r} capability.",
                details={
                    "required_capability": required_capability,
                    "granted_capabilities": sorted(self._granted_capabilities),
                },
            )
        if isinstance(msg, ExecuteCommandMessage):
            return await self._command_handler.handle(msg)
        if isinstance(msg, DirectoryListRequestMessage):
            return await self._handle_directory_list(msg)
        if isinstance(msg, AttachSessionMessage):
            return await self._handle_attach_session(msg)
        if isinstance(msg, ActivateSessionMessage):
            return await self._handle_activate_session(msg)
        if isinstance(msg, RestoreSessionMessage):
            return await self._handle_restore_session(msg)
        if isinstance(msg, DetachSessionMessage):
            return await self._handle_detach_session(msg)
        if isinstance(msg, FileReadMessage):
            return await self._file_handler.handle_read(msg)
        if isinstance(msg, FileWriteMessage):
            return await self._file_handler.handle_write(msg)
        if isinstance(msg, FileStatMessage):
            return await self._file_handler.handle_stat(msg)
        if isinstance(msg, FileListMessage):
            return await self._file_handler.handle_list(msg)
        if isinstance(msg, FileDeleteMessage):
            return await self._file_handler.handle_delete(msg)
        if isinstance(msg, FileMoveMessage):
            return await self._file_handler.handle_move(msg)
        if isinstance(msg, ComputerToolCallMessage):
            # One GUI: the handler makes calls from every session take turns,
            # within each call's deadline.
            return await self._computer_handler.handle(msg, received_at=_request_received_at.get())
        if isinstance(msg, ComputerToolsRequestMessage):
            return await self._computer_handler.handle(msg)
        if isinstance(msg, LocalLLMCompletionMessage):
            return await self._local_llm_handler.handle(msg, send=send)
        if isinstance(msg, self._RECORDING_MESSAGE_TYPES):
            # START/STOP/FETCH are request/response ops that count against
            # in-flight (§6). The `send` callback lets START stream unsolicited
            # RECORDING_STEP frames in co-pilot mode (exempt from accounting,
            # like STATUS) while the terminal response flows the normal path.
            return await self._recording_handler.handle(msg, send=send)
        # Anything else (e.g., responses we didn't ask for) is silently dropped.
        logger.debug("No handler for %s; dropping", type(msg).__name__)
        return None

    # ── Persistent machine control channel ──────────────────────────────

    def _control_unavailable(self, msg_id: str) -> BaseModel | None:
        if self._control_mode:
            return None
        return make_error(
            msg_id,
            ErrorCode.FEATURE_NOT_SUPPORTED,
            "Machine-control operations are only accepted on the base control connection.",
        )

    async def _handle_directory_list(
        self, msg: DirectoryListRequestMessage
    ) -> DirectoryListResponseMessage | BaseModel:
        unavailable = self._control_unavailable(msg.id)
        if unavailable is not None:
            return unavailable
        assert self._directory_browser is not None
        try:
            async with self._browser_lock:
                listing = await asyncio.to_thread(
                    self._directory_browser.list_directories,
                    msg.payload.browse_id,
                    msg.payload.directory_ref,
                    msg.payload.cursor,
                )
        except DirectoryBrowseError as exc:
            await self._audit_control_failure(
                "DIRECTORY_LIST",
                msg.id,
                exc.code,
                {"browse_id": msg.payload.browse_id},
            )
            return make_error(msg.id, exc.code, exc.message)
        await self._write_control_audit(
            "DIRECTORY_LIST",
            request_id=msg.id,
            details={
                "browse_id": listing.browse_id,
                "path": listing.current.path if listing.current is not None else None,
                "entries_returned": len(listing.entries),
                "truncated": listing.truncated,
            },
        )
        return DirectoryListResponseMessage(
            id=msg.id,
            ts=now_ts(),
            payload=self._directory_listing_payload(listing),
        )

    async def _handle_attach_session(
        self, msg: AttachSessionMessage
    ) -> SessionAttachedMessage | BaseModel:
        async with self._session_lock(msg.payload.session_id):
            return await self._handle_attach_session_locked(msg)

    async def _handle_attach_session_locked(
        self, msg: AttachSessionMessage
    ) -> SessionAttachedMessage | BaseModel:
        unavailable = self._control_unavailable(msg.id)
        if unavailable is not None:
            return unavailable
        assert self._directory_browser is not None
        active = self._child_sessions.get(msg.payload.session_id)
        if active is not None and not active.task.done():
            return make_error(
                msg.id,
                ErrorCode.SESSION_ALREADY_ACTIVE,
                "Detach the active session before changing its allowed root.",
            )
        try:
            async with self._browser_lock:
                root = await asyncio.to_thread(
                    self._directory_browser.resolve_and_consume,
                    msg.payload.browse_id,
                    msg.payload.directory_ref,
                )
            previous = self._session_bindings.get(msg.payload.session_id)
            revision = previous.revision + 1 if previous is not None else 1
            binding = await asyncio.to_thread(
                self._root_grants.issue,
                msg.payload.session_id,
                root,
                revision,
            )
        except (DirectoryBrowseError, RootGrantError) as exc:
            code = getattr(exc, "code", ErrorCode.DIRECTORY_UNAVAILABLE)
            await self._audit_control_failure(
                "ATTACH_SESSION",
                msg.id,
                str(code),
                {"session_id": msg.payload.session_id},
                session_id=msg.payload.session_id,
            )
            return make_error(msg.id, code, str(exc))
        await self._write_control_audit(
            "SESSION_ALLOWED_ROOT_SET",
            request_id=msg.id,
            session_id=msg.payload.session_id,
            details={
                "previous_allowed_root": (
                    str(previous.allowed_root) if previous is not None else None
                ),
                "allowed_root": str(binding.allowed_root),
                "fingerprint": binding.fingerprint,
                "revision": binding.revision,
            },
        )
        self._remember_binding(binding)
        return SessionAttachedMessage(
            id=msg.id,
            ts=now_ts(),
            payload=self._attached_payload(binding),
        )

    async def _handle_activate_session(
        self, msg: ActivateSessionMessage
    ) -> SessionActivatedMessage | BaseModel:
        async with self._session_lock(msg.payload.session_id):
            return await self._handle_activate_session_locked(msg)

    async def _handle_activate_session_locked(
        self, msg: ActivateSessionMessage
    ) -> SessionActivatedMessage | BaseModel:
        unavailable = self._control_unavailable(msg.id)
        if unavailable is not None:
            return unavailable
        binding = self._session_bindings.get(msg.payload.session_id)
        if binding is None:
            return make_error(
                msg.id,
                ErrorCode.SESSION_NOT_ATTACHED,
                "Attach a directory before activating this session.",
            )
        if binding.revision != msg.payload.revision:
            return make_error(
                msg.id,
                ErrorCode.SESSION_REVISION_MISMATCH,
                "The requested root revision is stale.",
                details={"current_revision": binding.revision},
            )
        try:
            binding = await asyncio.to_thread(
                self._root_grants.verify,
                binding.root_grant,
                binding.session_id,
            )
        except RootGrantError as exc:
            await self._audit_control_failure(
                "ACTIVATE_SESSION",
                msg.id,
                ErrorCode.ROOT_GRANT_INVALID.value,
                {"session_id": msg.payload.session_id},
                session_id=msg.payload.session_id,
            )
            return make_error(msg.id, ErrorCode.ROOT_GRANT_INVALID, str(exc))
        previous_binding = self._session_bindings.get(binding.session_id)
        self._remember_binding(binding)
        active = self._child_sessions.get(binding.session_id)
        started_child = active is None or active.task.done()
        try:
            if started_child:
                await self._start_child_session(binding)
            await self._write_control_audit(
                "ACTIVATE_SESSION",
                request_id=msg.id,
                session_id=binding.session_id,
                details={
                    "allowed_root": str(binding.allowed_root),
                    "fingerprint": binding.fingerprint,
                    "revision": binding.revision,
                },
            )
        except Exception:
            if started_child:
                await self._stop_child_session(binding.session_id)
            if previous_binding is None:
                self._session_bindings.pop(binding.session_id, None)
            else:
                self._remember_binding(previous_binding)
            raise
        return SessionActivatedMessage(
            id=msg.id,
            ts=now_ts(),
            payload=SessionActivatedPayload(
                session_id=binding.session_id,
                allowed_root=str(binding.allowed_root),
                fingerprint=binding.fingerprint,
                revision=binding.revision,
            ),
        )

    async def _handle_restore_session(
        self, msg: RestoreSessionMessage
    ) -> SessionRestoredMessage | BaseModel:
        async with self._session_lock(msg.payload.session_id):
            return await self._handle_restore_session_locked(msg)

    async def _handle_restore_session_locked(
        self, msg: RestoreSessionMessage
    ) -> SessionRestoredMessage | BaseModel:
        unavailable = self._control_unavailable(msg.id)
        if unavailable is not None:
            return unavailable
        try:
            binding = await asyncio.to_thread(
                self._root_grants.verify,
                msg.payload.root_grant,
                msg.payload.session_id,
            )
        except RootGrantError as exc:
            await self._audit_control_failure(
                "RESTORE_SESSION",
                msg.id,
                ErrorCode.ROOT_GRANT_INVALID.value,
                {"session_id": msg.payload.session_id},
                session_id=msg.payload.session_id,
            )
            return make_error(
                msg.id,
                ErrorCode.ROOT_GRANT_INVALID,
                str(exc),
            )
        current = self._session_bindings.get(binding.session_id)
        if current is not None and current.revision > binding.revision:
            return make_error(
                msg.id,
                ErrorCode.SESSION_REVISION_MISMATCH,
                "The supplied root grant is older than the current binding.",
                details={"current_revision": current.revision},
            )
        if (
            current is not None
            and current.revision == binding.revision
            and current.fingerprint != binding.fingerprint
        ):
            return make_error(
                msg.id,
                ErrorCode.SESSION_REVISION_MISMATCH,
                "The supplied root grant conflicts with the current binding revision.",
                details={"current_revision": current.revision},
            )
        previous_binding = self._session_bindings.get(binding.session_id)
        active = self._child_sessions.get(binding.session_id)
        active_is_live = active is not None and not active.task.done()
        replaced_child = active_is_live and previous_binding != binding
        if replaced_child:
            await self._stop_child_session(binding.session_id)
        self._remember_binding(binding)
        started_child = not active_is_live or replaced_child
        try:
            if started_child:
                await self._start_child_session(binding)
            await self._write_control_audit(
                "RESTORE_SESSION",
                request_id=msg.id,
                session_id=binding.session_id,
                details={
                    "allowed_root": str(binding.allowed_root),
                    "fingerprint": binding.fingerprint,
                    "revision": binding.revision,
                },
            )
        except Exception:
            if started_child:
                await self._stop_child_session(binding.session_id)
            if previous_binding is None:
                self._session_bindings.pop(binding.session_id, None)
            else:
                self._remember_binding(previous_binding)
                if replaced_child:
                    try:
                        await self._start_child_session(previous_binding)
                    except Exception:
                        logger.exception(
                            "Failed to restore prior child for session %s",
                            binding.session_id,
                        )
            raise
        return SessionRestoredMessage(
            id=msg.id,
            ts=now_ts(),
            payload=SessionRestoredPayload(
                session_id=binding.session_id,
                allowed_root=str(binding.allowed_root),
                fingerprint=binding.fingerprint,
                revision=binding.revision,
                root_grant=binding.root_grant,
            ),
        )

    async def _handle_detach_session(self, msg: DetachSessionMessage) -> BaseModel:
        async with self._session_lock(msg.payload.session_id):
            return await self._handle_detach_session_locked(msg)

    async def _handle_detach_session_locked(self, msg: DetachSessionMessage) -> BaseModel:
        unavailable = self._control_unavailable(msg.id)
        if unavailable is not None:
            return unavailable
        await self._stop_child_session(msg.payload.session_id)
        binding = self._session_bindings.get(msg.payload.session_id)
        await self._write_control_audit(
            "DETACH_SESSION",
            request_id=msg.id,
            session_id=msg.payload.session_id,
            details={
                "allowed_root": str(binding.allowed_root) if binding is not None else None,
                "revision": binding.revision if binding is not None else None,
            },
        )
        self._session_bindings.pop(msg.payload.session_id, None)
        return make_ack(msg.id)

    @asynccontextmanager
    async def _session_lock(self, session_id: str) -> AsyncIterator[None]:
        state = self._session_locks.get(session_id)
        if state is None:
            state = _SessionLockState(lock=asyncio.Lock())
            self._session_locks[session_id] = state
        state.users += 1
        try:
            async with state.lock:
                yield
        finally:
            state.users -= 1
            if state.users == 0 and self._session_locks.get(session_id) is state:
                self._session_locks.pop(session_id, None)

    def _remember_binding(self, binding: RootBinding) -> None:
        self._session_bindings.pop(binding.session_id, None)
        self._session_bindings[binding.session_id] = binding
        while len(self._session_bindings) > MAX_SESSION_BINDINGS:
            for session_id in tuple(self._session_bindings):
                active = self._child_sessions.get(session_id)
                if active is None or active.task.done():
                    self._session_bindings.pop(session_id, None)
                    break
            else:
                raise RuntimeError("Local-executor session binding limit reached")

    async def _start_child_session(self, binding: RootBinding) -> None:
        async with self._children_lock:
            live_count = sum(not child.task.done() for child in self._child_sessions.values())
            if live_count >= MAX_ACTIVE_CHILD_SESSIONS:
                raise RuntimeError("Active local-executor session limit reached")
            child_config = self.config.model_copy(deep=True)
            child_config.session_id = binding.session_id
            child_config.allowed_root = binding.allowed_root
            base_audit = (
                self.audit.writer if isinstance(self.audit, SessionAuditWriter) else self.audit
            )
            if not isinstance(base_audit, AuditWriter):
                raise RuntimeError("Child sessions require the mandatory shared audit writer")
            session_audit = SessionAuditWriter(base_audit, binding.session_id)
            child = ShimDaemon(
                child_config,
                token_store=self.token_store,
                audit=session_audit,
                manage_audit_lifecycle=False,
                token_refresh_lock=self._token_refresh_lock,
                machine_work_semaphore=self._machine_work_semaphore,
                computer_use_lock=self._computer_use_lock,
            )
            self._command_handler.share_with(child._command_handler)
            self._computer_handler.share_with(child._computer_handler)
            child._local_llm_handler._inflight_lock = self._local_llm_handler._inflight_lock
            task = asyncio.create_task(child.run(), name=f"local-executor-{binding.session_id}")
            active = _ActiveChild(daemon=child, task=task)
            self._child_sessions[binding.session_id] = active

            def child_finished(finished: asyncio.Task[None]) -> None:
                self._child_finished(binding.session_id, finished)

            task.add_done_callback(child_finished)

    def _child_finished(self, session_id: str, task: asyncio.Task[None]) -> None:
        active = self._child_sessions.get(session_id)
        if active is not None and active.task is task:
            self._child_sessions.pop(session_id, None)
        if not task.cancelled() and task.exception() is not None:
            logger.error("Child session %s stopped: %s", session_id, task.exception())

    async def _stop_child_session(self, session_id: str) -> None:
        async with self._children_lock:
            active = self._child_sessions.pop(session_id, None)
            if active is None:
                return
            # Cancelled before anything here waits, as _shutdown_children does
            # every chat: a stop cut short (a detach under way when the
            # machine's session is revoked) let go of a chat still running.
            connection = self._cancel_child(active)
            await _close_quietly(connection)
            await asyncio.gather(active.task, return_exceptions=True)

    async def _shutdown_children(self) -> None:
        await self._close_and_drain_children(self._cancel_children())

    def _cancel_children(self) -> tuple[tuple[str, _ActiveChild, Any], ...]:
        """Cancel every chat, with nothing awaited, for
        _close_and_drain_children to finish. Stopped one after another, each
        waited for in full, a chat held up stopping (its WS_DISCONNECTED
        record waiting on an audit writer that `audit verify` holds, say) left
        the chats after it running their commands, after the machine's
        session was revoked too."""
        return tuple(
            (session_id, active, self._cancel_child(active))
            for session_id, active in self._child_sessions.items()
        )

    @staticmethod
    def _cancel_child(active: _ActiveChild) -> Any:
        """The part of a chat's stop that waits for nothing: it takes no more
        work, and its requests and its session are cancelled. Returns its
        connection, to be closed: the session lets go of it before its
        WS_DISCONNECTED record, and closes it only once that is written."""
        child = active.daemon
        connection = child._ws
        child._running = False
        child._cancel_dispatch_requests()
        active.task.cancel()
        return connection

    async def _close_and_drain_children(
        self, cancelled: tuple[tuple[str, _ActiveChild, Any], ...]
    ) -> None:
        """Close the cancelled chats' connections, all at once and whatever
        holds the children's lock, then wait for their sessions to end."""
        await asyncio.gather(*(_close_quietly(connection) for _, _, connection in cancelled))
        async with self._children_lock:
            for session_id, active, _ in cancelled:
                if self._child_sessions.get(session_id) is active:
                    del self._child_sessions[session_id]
            await asyncio.gather(
                *(active.task for _, active, _ in cancelled), return_exceptions=True
            )

    @staticmethod
    def _directory_listing_payload(listing: DirectoryListing) -> DirectoryListResponsePayload:
        def convert(entry: Any) -> DirectoryReferencePayload:
            return DirectoryReferencePayload(
                directory_ref=entry.directory_ref,
                name=entry.name,
                path=entry.path,
            )

        return DirectoryListResponsePayload(
            browse_id=listing.browse_id,
            current=convert(listing.current) if listing.current is not None else None,
            parent_ref=listing.parent_ref,
            entries=[convert(entry) for entry in listing.entries],
            next_cursor=listing.next_cursor,
            truncated=listing.truncated,
            expires_at=listing.expires_at,
        )

    @staticmethod
    def _attached_payload(binding: RootBinding) -> SessionAttachedPayload:
        return SessionAttachedPayload(
            session_id=binding.session_id,
            allowed_root=str(binding.allowed_root),
            fingerprint=binding.fingerprint,
            revision=binding.revision,
            root_grant=binding.root_grant,
        )

    async def _write_control_audit(
        self,
        op: str,
        *,
        request_id: str | None,
        details: dict[str, Any],
        session_id: str | None = None,
        result: dict[str, Any] | None = None,
    ) -> None:
        await self.audit.write(
            op,
            session_id=session_id,
            request_id=request_id,
            details=details,
            result=result,
        )

    async def _audit_control_failure(
        self,
        op: str,
        request_id: str,
        error_code: str,
        details: dict[str, Any],
        *,
        session_id: str | None = None,
    ) -> None:
        await self._write_control_audit(
            op,
            request_id=request_id,
            session_id=session_id,
            details=details,
            result={
                "ok": False,
                "exit_code": None,
                "duration_ms": 0,
                "error_code": error_code,
            },
        )

    # ── Preflight ────────────────────────────────────────────────────

    async def _preflight_or_raise(self) -> None:
        """Reject incomplete previews or unverified required OS permissions."""
        try:
            Platform(platform_info.detect_platform())
            Arch(platform_info.detect_arch())
        except ValueError as exc:
            # HELLO can't name this machine, so every connect would fail the
            # same way; stop here instead of retrying forever.
            await self._fail_preflight(
                f"This computer can't run the shim: {exc}. It supports macOS, Linux, "
                "WSL2 and Windows on x86_64 or arm64.",
                missing_permissions=["supported_platform"],
                error_code=ErrorCode.UNSUPPORTED_ARCH,
                permanent=True,
            )
        if self.config.enable_local_llm:
            await self._fail_preflight(
                "Local model routing remains disabled until platform routing and "
                "end-to-end cancellation are supported.",
                missing_permissions=["local_model_preview_disabled"],
                error_code=ErrorCode.CAPABILITY_NOT_GRANTED,
                permanent=True,
            )
        if self.config.enable_recording:
            await self._fail_preflight(
                "Workflow recording is a design preview and remains disabled: "
                "capture and interpretation are incomplete, so the shim will not advertise it.",
                missing_permissions=["recording_preview_disabled"],
                error_code=ErrorCode.CAPABILITY_NOT_GRANTED,
                permanent=True,
            )
        if not self.config.enable_computer_use:
            return
        await self._preflight_computer_use()

    async def _preflight_computer_use(self) -> None:
        """Computer use needs cua-driver, and on macOS its Accessibility and
        Screen Recording grants: without Screen Recording a capture shows
        only the wallpaper and menu bar."""
        driver = self._computer_handler.driver
        child = self._computer_handler.shares_a_driver
        on_macos = platform_info.detect_platform() == "darwin"
        # Looked for off the event loop, which the machine's other sessions
        # share, and once for all of them (CuaDriverBridge.find_binary). A
        # stalled mount on PATH or sys.path holds the look as long as it
        # stalls, and this comes before the session connects, its file and
        # shell work with it: past the wait (a chat's, or the machine's start)
        # the session goes on without the rest of this check. The look goes
        # on, and HELLO, waiting for it within its own wait, offers computer
        # use only once it finds the driver, and on macOS only once a
        # session's check of the grants has passed (permissions_granted).
        wait = handlers.CHILD_DRIVER_WAIT_SECONDS if child else DRIVER_PREFLIGHT_SECONDS
        try:
            async with asyncio.timeout(wait):
                found = await driver.find_binary()
        except TimeoutError:
            logger.warning(
                "Looking for cua-driver took longer than %g s: a folder on PATH, or one "
                "Python imports from, may be on a mount that isn't answering. Going on "
                "without computer use until the look finds the driver%s.",
                wait,
                " and a session's check of the macOS permissions passes" if on_macos else "",
            )
            return
        if found is None:
            await self._fail_preflight(
                "Computer use is enabled but cua-driver is not installed. Reinstall with "
                f"`{COMPUTER_USE_INSTALL}`, then run `autogpt-shim doctor`.",
                missing_permissions=["cua_driver"],
                error_code=ErrorCode.DEPENDENCY_MISSING,
                permanent=True,
            )
        if not on_macos:
            return
        try:
            if child:
                # Another chat's call can hold the shared driver far longer
                # than the platform waits for this session to connect, and a
                # driver that takes the check can stall: the whole check, its
                # run as well as its start, ends within a brief wait, without
                # waiting for the stop of a driver that overran it (the stop
                # keeps the driver's turn), and then the session goes ahead
                # unchecked (below).
                start_by = asyncio.get_running_loop().time() + handlers.CHILD_DRIVER_WAIT_SECONDS
                await driver.snapshot(deadline=start_by)
                left = max(start_by - asyncio.get_running_loop().time(), 0)
                timeout = min(PERMISSION_CHECK_SECONDS, left)
            else:
                await driver.snapshot()
                timeout = PERMISSION_CHECK_SECONDS
            result = await self._computer_handler.call_driver(
                "check_permissions",
                {"prompt": False, "session": self._computer_handler.session_label},
                timeout=timeout,
                wait_for_stop=not child,
            )
        except TimeoutError:  # CallNotStarted too
            if not child:
                await self._fail_preflight(
                    "cua-driver did not report its permissions in time. Run `autogpt-shim doctor`.",
                    missing_permissions=["accessibility_probe", "screen_recording_probe"],
                    error_code=ErrorCode.PERMISSION_PENDING,
                )
            # A child session's check waited behind another chat's call on the
            # shared driver, which is busy working, or the driver took it and
            # didn't answer in time: the grants are the app's, checked when the
            # parent started, so the session goes ahead rather than end after
            # activation reported it ready.
            logger.info(
                "Skipped the computer-use permission check: the shared driver didn't "
                "answer it in time"
            )
            return
        except DriverUnavailable as exc:
            await self._fail_preflight(
                f"Could not check computer-use permissions: {exc.message} "
                "Run `autogpt-shim doctor`.",
                missing_permissions=["accessibility_probe", "screen_recording_probe"],
                error_code=exc.code,
                permanent=exc.code == ErrorCode.DEPENDENCY_MISSING,
            )
        except McpError as exc:
            await self._fail_preflight(
                f"cua-driver could not check computer-use permissions: {exc.message}. "
                "Run `autogpt-shim doctor`.",
                missing_permissions=["accessibility_probe", "screen_recording_probe"],
                error_code=ErrorCode.INTERNAL_ERROR,
            )
        structured = result.get("structuredContent")
        granted = {
            name: isinstance(structured, dict) and structured.get(name) is True
            for name in ("accessibility", "screen_recording")
        }
        missing = [name for name, ok in granted.items() if not ok]
        if not missing:
            driver.permissions_granted = True
            return
        labels = {"accessibility": "Accessibility", "screen_recording": "Screen Recording"}
        await self._fail_preflight(
            f"{' and '.join(labels[name] for name in missing)} permission not granted to "
            "the app that runs the local executor; computer use requires it. Run "
            "`autogpt-shim doctor` to request it.",
            missing_permissions=missing,
            error_code=ErrorCode.PERMISSION_PENDING,
        )

    async def _fail_preflight(
        self,
        message: str,
        *,
        missing_permissions: list[str],
        error_code: ErrorCode,
        permanent: bool = False,
    ) -> Never:
        try:
            await self.audit.write(
                "DAEMON_PREFLIGHT_FAILED",
                request_id=new_id(),
                details={
                    "missing_permissions": missing_permissions,
                    "platform": platform_info.detect_platform(),
                    "hint": "Run `autogpt-shim doctor` and grant required access.",
                },
                result={
                    "ok": False,
                    "exit_code": 78,
                    "duration_ms": 0,
                    "error_code": error_code.value,
                },
            )
        except Exception as exc:
            raise DaemonPreflightError(
                f"{message} Mandatory preflight audit write also failed.", permanent=permanent
            ) from exc
        raise DaemonPreflightError(message, permanent=permanent)


__all__ = [
    "MAX_WEBSOCKET_MESSAGE_BYTES",
    "DaemonAuthenticationError",
    "DaemonPreflightError",
    "ShimDaemon",
]
