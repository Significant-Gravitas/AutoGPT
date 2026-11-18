"""
OAuth authentication for the shim.

Uses AutoGPT's existing OAuth 2.0 provider with:
- Authorization Code + PKCE flow
- Localhost callback server on port 41899
- Token storage in OS keychain via the `keyring` library
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import contextvars
import errno
import hashlib
import io
import json
import logging
import os
import secrets
import selectors
import socket
import sys
import threading
import time
import urllib.parse
import webbrowser
from collections.abc import AsyncIterator, Callable
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import IO, Any

import httpx
import keyring

from .audit import _fallback_path
from .config import ShimConfig, _canonical_endpoint

logger = logging.getLogger(__name__)

KEYCHAIN_SERVICE = "autogpt-local-executor"
# How long a token exchange, or a sign-out's revocations, may take in all.
# The stored sign-in's turn (KeychainTokenStore.turn) is held across them, so
# an endpoint that took a request and then went quiet would otherwise hold
# every other process's sign-in, sign-out and refresh for as long.
EXCHANGE_SECONDS = 30.0
# How long a sign-in, a sign-out or a refresh waits for the stored sign-in's
# turn: longer than a refresh or a sign-out holds it, so one that waits
# behind another goes on when that one is done.
TURN_WAIT_SECONDS = 60.0
# How often a wait for the turn tries again.
_TURN_POLL_SECONDS = 0.05
# How long `autogpt-shim auth` waits for the browser to come back.
CALLBACK_TIMEOUT_SECONDS = 300.0
# The redirect ports the OAuth app registers (41899-41910).
CALLBACK_FIRST_PORT = 41899
CALLBACK_LAST_PORT = 41910
# How long the callback server waits for one connection's request. It takes
# one connection at a time, so without this a connection that never sends
# (a browser preconnect, say) would hold off the real redirect.
CALLBACK_READ_TIMEOUT_SECONDS = 10.0
# How long one request may take to arrive in full. The timeout above bounds
# each read, so a client sending a byte every few seconds would otherwise
# hold the server, and keep the sign-in waiting past its deadline.
CALLBACK_REQUEST_SECONDS = 20.0


class _DeadlineReader(io.RawIOBase):
    """A socket's incoming bytes, read with a deadline for all of them: each
    read waits at most `per_read` seconds and never past `total` seconds from
    the start, then raises TimeoutError, which http.server answers by
    dropping the connection."""

    def __init__(self, sock: socket.socket, *, per_read: float, total: float) -> None:
        super().__init__()
        self._sock = sock
        self._per_read = per_read
        self._deadline = time.monotonic() + total

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("the request took too long to arrive")
        self._sock.settimeout(min(self._per_read, remaining))
        return self._sock.recv_into(buffer)


class _CallbackServer(HTTPServer):
    """The OAuth callback's server. On Windows it takes its port
    exclusively: there SO_REUSEADDR, which HTTPServer sets, binds a port
    another socket is already listening on, so the fallback to the next
    port never ran and the redirect could reach the other listener.
    Elsewhere SO_REUSEADDR stays; it only lets a new sign-in bind over the
    last one's closed connections.

    The redirect URI says `localhost`, which a browser may try as ::1 before
    127.0.0.1, so where the machine has IPv6 the server has a companion on
    ::1 and the same port, closed with it."""

    companion: _CallbackServer | None = None
    # Never SO_REUSEPORT, which Python 3.14's first betas had HTTPServer set:
    # another program of the same user could listen on the port too and take
    # the redirect.
    allow_reuse_port = False

    def server_bind(self) -> None:
        if sys.platform == "win32":
            self.allow_reuse_address = False
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

    def server_close(self) -> None:
        super().server_close()
        if self.companion is not None:
            self.companion.server_close()


class _CallbackServer6(_CallbackServer):
    address_family = socket.AF_INET6


# What binding ::1 fails with on a machine without IPv6 loopback.
_NO_IPV6_LOOPBACK = frozenset(
    {errno.EADDRNOTAVAIL, errno.EAFNOSUPPORT, getattr(errno, "EPROTONOSUPPORT", errno.EAFNOSUPPORT)}
)


class CredentialScopeError(ValueError):
    """Stored credentials cannot be used with the configured destinations."""


class OAuthDenied(ValueError):
    """The sign-in redirect came back with an error instead of a code. Its
    text is the provider's error and reason, kept to printable text: safe to
    show, unlike other failures, whose text can hold a token."""


@dataclass(frozen=True)
class CredentialScope:
    backend_url: str
    authorize_url: str
    token_url: str
    revoke_url: str
    websocket_url: str
    client_id: str

    @classmethod
    def from_config(cls, config: ShimConfig) -> CredentialScope:
        return cls(
            backend_url=_canonical_endpoint(config._platform_base()),
            authorize_url=_canonical_endpoint(config.derived_oauth_url.rstrip("/") + "/authorize"),
            token_url=_canonical_endpoint(config.derived_oauth_token_url),
            revoke_url=_canonical_endpoint(config.derived_oauth_revoke_url),
            websocket_url=_canonical_endpoint(config.derived_ws_url.rstrip("/")),
            client_id=config.oauth_client_id,
        )

    def ensure_matches(self, config: ShimConfig) -> None:
        if self != self.from_config(config):
            raise CredentialScopeError(
                "OAuth configuration has changed. Restart and run `autogpt-shim auth` "
                "with the intended deployment configuration before connecting."
            )


@dataclass(frozen=True)
class RefreshedTokens:
    """The pair a refresh exchange returned, and the refresh token it used up."""

    used_refresh_token: str
    access_token: str
    refresh_token: str


class SignInBusyError(RuntimeError):
    """Another use of this deployment's stored sign-in, most likely another
    process's, kept its turn (KeychainTokenStore.turn) for as long as this
    one could wait. Its text is safe to show."""


class ExchangeTimeoutError(TimeoutError):
    """A token exchange, or a sign-out's revocations, took longer than
    EXCHANGE_SECONDS in all."""


_BUSY = (
    "Another AutoGPT Local PC process on this computer is still changing or refreshing "
    "this deployment's stored sign-in. Try again shortly; if this goes on, stop that "
    "process first."
)

# This process's own lock for each record's turn, by its lock file's name.
# The lock file alone doesn't serve inside one process everywhere: where
# flock is emulated with POSIX record locks (Linux on NFS), a process's second
# handle takes a lock its first one holds, and closing either lets go of both.
_process_turns: dict[str, threading.Lock] = {}
# The lock files this process has said it can't use, to say each once.
_unusable_lock_files: set[str] = set()
_process_turns_guard = threading.Lock()
# The records whose turn the running task holds. What it does inside one
# (revoke_tokens' clear_tokens, the refresh the daemon wraps in the turn) is
# part of that turn and doesn't wait for it again. A thread or a task it
# starts there gets a copy, as part of the same work.
_turns_held: contextvars.ContextVar[frozenset[str]] = contextvars.ContextVar(
    "stored_sign_in_turns_held", default=frozenset()
)


def _lock_file(record_key: str) -> Path:
    """Where a record's lock file is: `.<record key>.lock` beside the audit
    key's lock, in AUTOGPT_SHIM_FALLBACK_DIR (the person's own
    ~/.autogpt-local-executor unless it is set).

    Every process of the account that uses a record has to take the same
    file, so it can't sit by the audit log or the config file, which two
    processes for one deployment can each have their own of. It is named for
    the record, so deployments don't wait for one another. And it is the
    person's own, in their folder and created 0600: another account that
    could open it could hold it, and every sign-in and refresh would wait.
    AUTOGPT_SHIM_FALLBACK_DIR moves it with the audit key's lock, for a home
    without file locks."""
    return _fallback_path().with_name(f".{record_key}.lock")


def _lock_without_waiting(handle: IO[bytes]) -> bool:
    """Take an exclusive lock on an open lock file, the lock audit._file_lock
    takes, without waiting for it: False while another handle holds it, this
    process's or another's. Any other failure is raised: the file can't be
    locked at all."""
    if sys.platform == "win32":
        import msvcrt

        try:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            # EACCES while another handle holds the lock, as in _file_lock.
            if exc.errno in (errno.EACCES, errno.EDEADLK):
                return False
            raise
    else:
        import fcntl

        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
    return True


def _unlock(handle: IO[bytes]) -> None:
    if sys.platform == "win32":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle, fcntl.LOCK_UN)


class _Turn:
    """One hold of a stored sign-in's turn: this process's lock for the
    record, and an exclusive lock on the record's lock file (_lock_file),
    which every process takes.

    take() tries for both, again and again until its deadline, and waits in
    no blocking call: a worker thread left in one would keep `asyncio.run`
    from returning, and with it the exit the shim's supervisor restarts it
    after. release() lets go, from any thread, and ends a take() under way.

    A lock file that can't be made or locked at all (a read-only folder, a
    filesystem without file locks) is said once and done without: the turn
    is then this process's alone, and between processes what is left is the
    refresh's own look at the keychain before it writes
    (KeychainTokenStore.store_refreshed)."""

    def __init__(self, record_key: str) -> None:
        self._path: Path | None
        try:
            self._path = _lock_file(record_key)
        except RuntimeError:
            # No home folder to keep it in (Path.home).
            self._path = None
        self._name = str(self._path) if self._path is not None else f".{record_key}.lock"
        with _process_turns_guard:
            self._in_process = _process_turns.setdefault(self._name, threading.Lock())
        # Guards _held against a release() during a take().
        self._state = threading.Lock()
        self._held = False
        self._over = threading.Event()
        self._handle: IO[bytes] | None = None
        self._file_locked = False
        self._without_file = False

    def take(self, wait: float) -> None:
        """Take the turn within `wait` seconds, else raise SignInBusyError."""
        deadline = time.monotonic() + wait
        taken = said = False
        try:
            while not self._try():
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self._over.is_set():
                    raise SignInBusyError(_BUSY)
                if not said:
                    said = True
                    logger.info(
                        "Waiting for another use of this deployment's stored sign-in to "
                        "finish: a sign-in, a sign-out or a token refresh, most likely in "
                        "another AutoGPT Local PC process."
                    )
                self._over.wait(min(_TURN_POLL_SECONDS, remaining))
            taken = True
        finally:
            if not taken:
                self._close_file()

    def release(self) -> None:
        """Let go of the turn, or give up on a take() still under way."""
        with self._state:
            self._over.set()
            held, self._held = self._held, False
        if held:
            self._close_file()
            self._in_process.release()

    def _try(self) -> bool:
        if not self._in_process.acquire(blocking=False):
            return False
        taken = False
        try:
            if self._lock_file_now():
                with self._state:
                    # Given up on meanwhile (release): not taken after all.
                    taken = self._held = not self._over.is_set()
        finally:
            if not taken:
                self._unlock_file()
                self._in_process.release()
        return taken

    def _lock_file_now(self) -> bool:
        """Lock the record's lock file without waiting: False while another
        handle holds it, True once it is locked or is done without."""
        if self._without_file:
            return True
        try:
            if self._handle is None:
                if self._path is None:
                    raise FileNotFoundError("no home folder to keep it in")
                self._path.parent.mkdir(parents=True, exist_ok=True)
                flags = os.O_RDWR | os.O_CREAT | os.O_APPEND | getattr(os, "O_BINARY", 0)
                self._handle = os.fdopen(os.open(self._path, flags, 0o600), "a+b")
            self._file_locked = _lock_without_waiting(self._handle)
        except OSError as exc:
            self._do_without_the_file(exc)
            return True
        return self._file_locked

    def _do_without_the_file(self, exc: OSError) -> None:
        self._close_file()
        self._without_file = True
        with _process_turns_guard:
            said = self._name in _unusable_lock_files
            _unusable_lock_files.add(self._name)
        if not said:
            logger.warning(
                "The stored sign-in's lock file %s can't be used (%s). Going on without it: "
                "two AutoGPT Local PC processes for this deployment that refresh the sign-in "
                "at the same moment can end it. Run one process for the deployment, or set "
                "AUTOGPT_SHIM_FALLBACK_DIR to a folder that takes file locks.",
                self._name,
                exc,
            )

    def _unlock_file(self) -> None:
        if self._file_locked and self._handle is not None:
            self._file_locked = False
            with contextlib.suppress(OSError):
                _unlock(self._handle)

    def _close_file(self) -> None:
        self._unlock_file()
        handle, self._handle = self._handle, None
        if handle is not None:
            with contextlib.suppress(OSError):
                handle.close()


async def _let_go(turn: _Turn) -> None:
    """Release `turn` in a worker thread, as it was taken (unlocking and
    closing its file is file work too), and to the end even if the caller is
    cancelled meanwhile."""
    await asyncio.shield(asyncio.to_thread(turn.release))


async def _to_its_end(change: Callable[[], None], then: Callable[[], None]) -> None:
    """Make a change to the keychain in a worker thread, call `then` once it
    has gone through, and return only when it is over, even if the caller is
    cancelled meanwhile: the cancellation is raised then, and a change that
    failed raises its own failure otherwise.

    For a caller that holds the record's turn (KeychainTokenStore.turn).
    Cancelling the wait for a worker thread doesn't stop the thread: a
    sign-in or a sign-out cancelled during its keychain call (Ctrl+C while
    the keychain asks the person) let go of the turn with the call still
    under way. Another process's sign-in then went ahead, and the call,
    landing after it, replaced or removed it. A refresh cancelled while its
    new pair was being written kept the turn for the store instead, whose
    next sign-out then went ahead of the write. The process couldn't have
    exited sooner in any case: asyncio.run waits for the thread."""
    loop = asyncio.get_running_loop()
    # Held only here, and waited for only through a shield, so nothing
    # cancels it: it is done when the thread is.
    doing = loop.run_in_executor(None, contextvars.copy_context().run, change)
    cancelled: asyncio.CancelledError | None = None
    while not doing.done():
        try:
            await asyncio.shield(doing)
        except asyncio.CancelledError as exc:
            cancelled = exc
        except Exception:
            # The change's own failure, read below.
            break
    failure = doing.exception()
    if failure is None:
        then()
    if cancelled is not None:
        raise cancelled
    if failure is not None:
        raise failure


@contextlib.asynccontextmanager
async def _within_exchange_time() -> AsyncIterator[None]:
    """Give the block EXCHANGE_SECONDS in all, then ExchangeTimeoutError: the
    stored sign-in's turn is held while it runs."""
    limit = asyncio.timeout(EXCHANGE_SECONDS)
    try:
        async with limit:
            yield
    except TimeoutError:
        if limit.expired():
            raise ExchangeTimeoutError(
                f"The platform took longer than {EXCHANGE_SECONDS:g} s to answer."
            ) from None
        raise


class KeychainTokenStore:
    """
    Stores OAuth tokens in the OS keychain using the `keyring` library.

    Backends by platform:
        macOS   → Keychain Services
        Linux   → Secret Service (GNOME Keyring / KWallet)
        Windows → Windows Credential Manager

    Every shim process of the account shares a deployment's record: a second
    daemon, `autogpt-shim auth`, `autogpt-shim revoke`. What changes it takes
    the record's turn (turn), across processes.
    """

    def __init__(self, config: ShimConfig) -> None:
        self.scope = CredentialScope.from_config(config)
        identity = json.dumps(asdict(self.scope), sort_keys=True, separators=(",", ":"))
        self._record_key = "oauth_v1_" + hashlib.sha256(identity.encode()).hexdigest()
        # A refreshed pair not stored yet (OAuthFlow.refresh_token), kept for
        # every session of this process: they share the store.
        self.unsaved_refresh: RefreshedTokens | None = None
        # The turn kept with it (turn), so that no other process sends the
        # refresh token its exchange used up.
        self._kept_turn: _Turn | None = None

    @contextlib.asynccontextmanager
    async def turn(self) -> AsyncIterator[None]:
        """Hold this record's turn while the block runs. One sign-in,
        sign-out or refresh at a time has the stored sign-in, in whatever
        process, from its first read to its last write: two refreshes at
        once would each send the same refresh token, which the platform
        takes for reuse, and a sign-in or sign-out could land between a
        refresh's look at the keychain and its write.

        The turn is this process's lock for the record and an exclusive lock
        on the record's lock file (_Turn, _lock_file). It is waited for at
        most TURN_WAIT_SECONDS, in a worker thread that tries again and
        again rather than block; past that, SignInBusyError, and the caller
        has changed nothing. What the running task does inside its own turn
        doesn't wait for it again.

        While a refreshed pair isn't stored yet (unsaved_refresh), the store
        keeps the turn when the block ends, and the next block goes on with
        it: the keychain still holds the refresh token that exchange used
        up, which another process would otherwise read and send. So until
        the pair is stored, or this process ends, another process's refresh
        fails as busy, and so do its sign-in and sign-out."""
        if self._record_key in _turns_held.get():
            yield
            return
        turn, self._kept_turn = self._kept_turn, None
        if turn is None:
            turn = _Turn(self._record_key)
            try:
                await asyncio.to_thread(turn.take, TURN_WAIT_SECONDS)
            except BaseException:
                # Cancelled while it waited, say: the wait ends, and a turn
                # it took just then is let go.
                await _let_go(turn)
                raise
        held = _turns_held.set(_turns_held.get() | {self._record_key})
        try:
            yield
        finally:
            _turns_held.reset(held)
            if self.unsaved_refresh is not None:
                self._kept_turn = turn
            else:
                await _let_go(turn)

    async def get_access_token(self) -> str | None:
        record = await asyncio.to_thread(self._read_record)
        return record["access_token"] if record is not None else None

    async def store_tokens(self, access_token: str, refresh_token: str) -> None:
        """Persist one versioned, deployment-bound token pair atomically, in
        the record's turn (turn): a sign-in waits for a refresh or a
        sign-out under way in any process, and replaces a refreshed pair
        this store still keeps unsaved. The turn is held until the keychain
        call is over, even if the caller is cancelled meanwhile
        (_to_its_end)."""
        record = self._record(access_token, refresh_token)
        async with self.turn():
            # To its end, cancelled or not: the turn isn't let go under it.
            await _to_its_end(
                lambda: keyring.set_password(KEYCHAIN_SERVICE, self._record_key, record),
                self._forget_unsaved_refresh,
            )

    def _forget_unsaved_refresh(self) -> None:
        self.unsaved_refresh = None

    async def store_refreshed(self, tokens: RefreshedTokens) -> bool:
        """Store a refreshed pair while the keychain still holds the refresh
        token its exchange used up, and say whether it did. Otherwise the
        keychain is left as it is: a sign-in stored since (`autogpt-shim
        auth`) is newer, and a record removed since (`autogpt-shim revoke`)
        signed this computer out.

        For a caller that holds the record's turn (turn), as
        OAuthFlow.refresh_token does: no other process's sign-in, sign-out
        or refresh then lands between this look at the keychain and the
        write. The look stays for where the lock file can't be used. The
        keychain can't compare and replace in one step, so there the read
        and the write run back to back in one thread, and a sign-in stored
        between those two calls is still replaced.

        Like a sign-in and a sign-out, it returns only when its keychain
        calls are over, even if the caller is cancelled meanwhile
        (_to_its_end): the caller's turn isn't left with the write still
        under way, for this store's next sign-out to go ahead of it. Once
        they are over the pair is no longer unsaved, stored or set aside,
        so a cancelled caller doesn't keep the turn for it."""
        record = self._record(tokens.access_token, tokens.refresh_token)
        stored: list[bool] = []

        def store_if_unchanged() -> None:
            current = self._read_record()
            if current is None or current["refresh_token"] != tokens.used_refresh_token:
                stored.append(False)
                return
            keyring.set_password(KEYCHAIN_SERVICE, self._record_key, record)
            stored.append(True)

        await _to_its_end(store_if_unchanged, self._forget_unsaved_refresh)
        return stored[0]

    def _record(self, access_token: str, refresh_token: str) -> str:
        if not isinstance(access_token, str) or not access_token:
            raise CredentialScopeError("OAuth response did not contain a valid access token.")
        if not isinstance(refresh_token, str) or not refresh_token:
            raise CredentialScopeError("OAuth response did not contain a valid refresh token.")
        return json.dumps(
            {
                "version": 1,
                "scope": asdict(self.scope),
                "access_token": access_token,
                "refresh_token": refresh_token,
            },
            separators=(",", ":"),
        )

    async def clear_tokens(self) -> None:
        """Remove only this deployment's record, in the record's turn (turn),
        and with it a refreshed pair this store still keeps unsaved. The
        turn is held until the keychain call is over, even if the caller is
        cancelled meanwhile (_to_its_end)."""
        async with self.turn():
            await _to_its_end(self._delete_record, self._forget_unsaved_refresh)

    def _delete_record(self) -> None:
        try:
            keyring.delete_password(KEYCHAIN_SERVICE, self._record_key)
        except keyring.errors.PasswordDeleteError:
            pass

    async def get_refresh_token(self) -> str | None:
        record = await asyncio.to_thread(self._read_record)
        return record["refresh_token"] if record is not None else None

    def _read_record(self) -> dict[str, str] | None:
        stored = keyring.get_password(KEYCHAIN_SERVICE, self._record_key)
        if stored is None:
            return None
        try:
            record = json.loads(stored)
            if (
                not isinstance(record, dict)
                or record.get("version") != 1
                or record.get("scope") != asdict(self.scope)
                or not isinstance(record.get("access_token"), str)
                or not record["access_token"]
                or not isinstance(record.get("refresh_token"), str)
                or not record["refresh_token"]
            ):
                raise ValueError
        except (ValueError, TypeError):
            raise CredentialScopeError(
                "Stored credentials could not be verified for this deployment. "
                "Run `autogpt-shim auth` with the intended deployment configuration."
            ) from None
        return {"access_token": record["access_token"], "refresh_token": record["refresh_token"]}


# The longest reason from a denied sign-in the terminal shows.
_OAUTH_ERROR_CHARS = 300


def _oauth_error_text(params: dict[str, list[str]]) -> str:
    """A denied sign-in's error code and, when the redirect gives one, its
    reason ("Local PC isn't available for this account yet."), kept to
    printable text, since the terminal shows it."""
    error = params["error"][0]
    reason = (params.get("error_description") or [""])[0]
    text = f"{error}: {reason}" if reason else error
    return "".join(char for char in text if char.isprintable())[:_OAUTH_ERROR_CHARS]


class OAuthFlow:
    """
    Runs the Authorization Code + PKCE flow against the AutoGPT OAuth provider.
    """

    # Only what a Local PC connection needs; USE_TOOLS also covers web search
    # and fetch through the External API.
    SCOPES = ["CONNECT_LOCAL_EXECUTOR"]

    def __init__(self, config: ShimConfig, token_store: KeychainTokenStore) -> None:
        self.config = config
        self.token_store = token_store
        self._scope = CredentialScope.from_config(config)
        self._ensure_scope()

    def _ensure_scope(self) -> None:
        self._scope.ensure_matches(self.config)
        if (
            isinstance(self.token_store, KeychainTokenStore)
            and self.token_store.scope != self._scope
        ):
            raise CredentialScopeError(
                "Stored credentials belong to different OAuth configuration. "
                "Run `autogpt-shim auth` with the intended deployment configuration."
            )

    def run(self) -> None:
        """Execute the full auth flow interactively."""
        code_verifier, code_challenge = self._generate_pkce_pair()
        state = secrets.token_urlsafe(32)
        server, auth_code, oauth_error, callback_port = self._bind_callback_server(
            expected_state=state
        )
        auth_url = self._build_auth_url(
            code_challenge,
            state=state,
            redirect_port=callback_port,
        )

        print("\nOpening browser for AutoGPT authentication...")
        print(f"If it doesn't open automatically, visit:\n  {auth_url}\n")
        webbrowser.open(auth_url)

        code = self._wait_for_callback(server, auth_code, oauth_error)
        tokens = asyncio.run(self._exchange_code(code, code_verifier, redirect_port=callback_port))
        self._ensure_scope()
        asyncio.run(self.token_store.store_tokens(tokens["access_token"], tokens["refresh_token"]))
        print("Authentication successful. Tokens stored in OS keychain.")

    async def refresh_token(self) -> None:
        """Use the stored refresh token to obtain a new access token.

        The platform rotates the refresh token on every exchange, so once one
        goes through the token the keychain holds is used up, and sending it
        again ends the sign-in. So the whole refresh, from reading the
        stored token to storing the new pair, runs in the record's turn
        (KeychainTokenStore.turn): another process's refresh, sign-in or
        sign-out waits for it, and one that can't have its turn in time
        raises SignInBusyError without exchanging. The exchange itself ends
        within EXCHANGE_SECONDS (ExchangeTimeoutError past that).

        A new pair the keychain then fails to store (locked meanwhile, say)
        is kept on the token store, the turn with it, until a later refresh
        stores it without exchanging again: no other process can read the
        used-up token and send it meanwhile. A restart loses the pair, so
        the daemon retries in the same process. The pair is stored only
        while the keychain still holds the token its exchange used up
        (store_refreshed): the turn keeps other processes' changes out, and
        where the lock file can't be used that look keeps a sign-in stored,
        or a sign-out made, while the exchange ran."""
        self._ensure_scope()
        async with self.token_store.turn():
            refresh = await self.token_store.get_refresh_token()
            tokens = self.token_store.unsaved_refresh
            if tokens is not None and tokens.used_refresh_token != refresh:
                tokens = self.token_store.unsaved_refresh = None
            if tokens is None:
                if not refresh:
                    raise ValueError("No refresh token stored. Run `autogpt-shim auth` first.")
                tokens = await self._exchange_refresh_token(refresh)
                self.token_store.unsaved_refresh = tokens
            self._ensure_scope()
            if not await self.token_store.store_refreshed(tokens):
                logger.info(
                    "The stored sign-in changed during the token refresh; keeping it, "
                    "not the refreshed pair."
                )
            self.token_store.unsaved_refresh = None

    async def _exchange_refresh_token(self, refresh: str) -> RefreshedTokens:
        async with _within_exchange_time(), httpx.AsyncClient() as client:
            self._ensure_scope()
            resp = await client.post(
                self._scope.token_url,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh,
                    "client_id": self._scope.client_id,
                },
            )
            resp.raise_for_status()
            tokens = resp.json()
        return RefreshedTokens(
            used_refresh_token=refresh,
            access_token=tokens["access_token"],
            refresh_token=tokens.get("refresh_token", refresh),
        )

    async def revoke_tokens(self) -> None:
        """Revoke both token types remotely before deleting local credentials.

        The desktop shim is a public client, so it sends the well-known client
        id with an empty secret. Any HTTP or keychain read error aborts before
        local deletion, leaving the credentials available for a retry.

        From reading the tokens to deleting them, it holds the record's turn
        (KeychainTokenStore.turn): a refresh under way in another process
        ends first, so the pair revoked is the one it stored, and none
        starts until the record is gone. The revocations end within
        EXCHANGE_SECONDS in all (ExchangeTimeoutError past that), as the turn
        is held meanwhile.
        """
        self._ensure_scope()
        async with self.token_store.turn():
            access = await self.token_store.get_access_token()
            refresh = await self.token_store.get_refresh_token()
            tokens = (
                (access, "access_token"),
                (refresh, "refresh_token"),
            )
            async with _within_exchange_time(), httpx.AsyncClient() as client:
                for token, token_type_hint in tokens:
                    if not token:
                        continue
                    self._ensure_scope()
                    resp = await client.post(
                        self._scope.revoke_url,
                        json={
                            "token": token,
                            "token_type_hint": token_type_hint,
                            "client_id": self._scope.client_id,
                            "client_secret": "",
                        },
                    )
                    resp.raise_for_status()
            self._ensure_scope()
            await self.token_store.clear_tokens()

    @staticmethod
    def _generate_pkce_pair() -> tuple[str, str]:
        code_verifier = secrets.token_urlsafe(64)
        digest = hashlib.sha256(code_verifier.encode()).digest()
        code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
        return code_verifier, code_challenge

    def _build_auth_url(
        self,
        code_challenge: str,
        *,
        state: str | None = None,
        redirect_port: int | None = None,
    ) -> str:
        self._ensure_scope()
        port = redirect_port or self.config.oauth_redirect_port
        params = {
            "response_type": "code",
            "client_id": self._scope.client_id,
            "redirect_uri": f"http://localhost:{port}/callback",
            "scope": " ".join(self.SCOPES),
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "state": state or secrets.token_urlsafe(32),
        }
        return self._scope.authorize_url + "?" + urllib.parse.urlencode(params)

    def _bind_callback_server(
        self, *, expected_state: str
    ) -> tuple[_CallbackServer, list[str], list[str], int]:
        auth_code: list[str] = []
        oauth_error: list[str] = []

        class CallbackHandler(BaseHTTPRequestHandler):
            timeout = CALLBACK_READ_TIMEOUT_SECONDS

            def setup(self) -> None:
                super().setup()
                self.rfile.close()
                self.rfile = io.BufferedReader(
                    _DeadlineReader(
                        self.connection,
                        per_read=CALLBACK_READ_TIMEOUT_SECONDS,
                        total=CALLBACK_REQUEST_SECONDS,
                    )
                )

            def do_GET(self):
                parsed = urllib.parse.urlparse(self.path)
                params = urllib.parse.parse_qs(parsed.query)
                state = params.get("state", [""])[0]
                if parsed.path != "/callback":
                    self.send_response(404)
                    message = b"Not found"
                elif not secrets.compare_digest(state.encode(), expected_state.encode()):
                    # Not this sign-in's redirect: a tab left from an earlier
                    # attempt, or any page the browser loaded requesting this
                    # URL. Refused, and the wait goes on, so neither can end
                    # the sign-in.
                    self.send_response(400)
                    message = (
                        b"This sign-in link is not from the current attempt. "
                        b"Finish signing in from the newest browser tab."
                    )
                elif "error" in params:
                    oauth_error.append(_oauth_error_text(params))
                    self.send_response(400)
                    message = b"Authentication was denied or failed."
                elif "code" in params and params["code"][0]:
                    auth_code.append(params["code"][0])
                    self.send_response(200)
                    message = b"<h1>Authenticated! You can close this tab.</h1>"
                else:
                    oauth_error.append("No authorization code received")
                    self.send_response(400)
                    message = b"Authentication failed: no authorization code."
                self.end_headers()
                self.wfile.write(message)

            def log_message(self, *args):
                pass

        first_port = self.config.oauth_redirect_port
        # The OAuth app lists 41899-41910 as redirect URIs; a port past that is
        # rejected on the redirect. A configured port outside the range is the
        # only one known to be registered.
        in_range = CALLBACK_FIRST_PORT <= first_port <= CALLBACK_LAST_PORT
        last_port = CALLBACK_LAST_PORT if in_range else first_port
        for port in range(first_port, last_port + 1):
            try:
                server = _CallbackServer(("127.0.0.1", port), CallbackHandler)
            except OSError:
                continue
            try:
                server.companion = _CallbackServer6(("::1", port), CallbackHandler)
            except OSError as exc:
                if exc.errno not in _NO_IPV6_LOOPBACK:
                    # Another program listens on ::1 there (or on every
                    # address, dual-stack): a browser trying ::1 first
                    # would take it the code.
                    server.server_close()
                    continue
            return server, auth_code, oauth_error, port
        raise RuntimeError(f"Could not bind OAuth callback on ports {first_port}-{last_port}")

    @staticmethod
    def _wait_for_callback(
        server: _CallbackServer,
        auth_code: list[str],
        oauth_error: list[str],
        *,
        timeout: float = CALLBACK_TIMEOUT_SECONDS,
    ) -> str:
        """Serve requests until the redirect to /callback arrives or `timeout`
        passes. A request for another path, such as a browser fetching
        /favicon.ico, does not end the wait."""
        deadline = time.monotonic() + timeout
        servers = [server] if server.companion is None else [server, server.companion]
        try:
            with selectors.DefaultSelector() as selector:
                for each in servers:
                    selector.register(each, selectors.EVENT_READ)
                while not auth_code and not oauth_error:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    for key, _events in selector.select(remaining):
                        ready = key.fileobj
                        assert isinstance(ready, _CallbackServer)
                        # It has a connection waiting: take it without waiting.
                        ready.timeout = 0
                        ready.handle_request()
                        if auth_code or oauth_error:
                            break
        finally:
            server.server_close()

        if not auth_code:
            if oauth_error:
                raise OAuthDenied(oauth_error[0])
            raise ValueError("OAuth callback timed out")
        return auth_code[0]

    async def _exchange_code(
        self,
        auth_code: str,
        code_verifier: str,
        *,
        redirect_port: int | None = None,
    ) -> dict:
        self._ensure_scope()
        port = redirect_port or self.config.oauth_redirect_port
        async with httpx.AsyncClient() as client:
            self._ensure_scope()
            resp = await client.post(
                self._scope.token_url,
                data={
                    "grant_type": "authorization_code",
                    "code": auth_code,
                    "redirect_uri": f"http://localhost:{port}/callback",
                    "client_id": self._scope.client_id,
                    "code_verifier": code_verifier,
                },
            )
            resp.raise_for_status()
            return resp.json()
