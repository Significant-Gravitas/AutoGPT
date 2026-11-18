"""The stored sign-in is one keychain record per deployment, which every shim
process of the account shares: the daemon, a second daemon, `autogpt-shim
auth` and `autogpt-shim revoke`. What changes it takes turns, across
processes."""

from __future__ import annotations

import asyncio
import errno
import json
import logging
import os
import stat
import subprocess
import sys
import threading
import time
import urllib.parse
from collections.abc import Awaitable, Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
import keyring.errors
import pytest

import autogpt_local_executor.auth as auth_module
from autogpt_local_executor.audit import AuditWriter, _file_lock
from autogpt_local_executor.auth import KeychainTokenStore, OAuthFlow
from autogpt_local_executor.config import ShimConfig
from autogpt_local_executor.daemon import (
    KeychainUnavailableError,
    ShimDaemon,
    TokenRefreshUnavailableError,
)

ROOT = Path(__file__).resolve().parent.parent


def _config(tmp_path: Path) -> ShimConfig:
    return ShimConfig(
        session_id="session-1",
        allowed_root=tmp_path,
        audit_log_path=tmp_path / "audit.log",
    )


def _daemon(config: ShimConfig, store: KeychainTokenStore | None = None) -> ShimDaemon:
    return ShimDaemon(
        config.model_copy(deep=True),
        token_store=store or KeychainTokenStore(config),
        audit=MagicMock(spec=AuditWriter),
    )


async def _refresh(daemon: ShimDaemon, rejected: str = "access-0") -> str | None:
    """The daemon's refresh after a 401 on `rejected`."""
    return await asyncio.wait_for(
        daemon._get_access_token(refresh_on_fail=True, rejected_token=rejected), timeout=120
    )


def _lock_file(store: KeychainTokenStore) -> Path:
    """Where docs/OAUTH_FLOW.md says a record's lock file is."""
    return Path(os.environ["AUTOGPT_SHIM_FALLBACK_DIR"]) / f".{store._record_key}.lock"


class _Keychain:
    """An in-memory keychain every store in the test shares. `after_read`
    runs once, when a read has its value and before it returns it: where
    another process's change lands unseen. `failures` holds what the next
    writes raise, one each."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.passwords: dict[tuple[str, str], str] = {}
        self.after_read: Callable[[], None] | None = None
        self.failures: list[Exception] = []
        monkeypatch.setattr("autogpt_local_executor.auth.keyring.get_password", self._get)
        monkeypatch.setattr("autogpt_local_executor.auth.keyring.set_password", self._set)
        monkeypatch.setattr("autogpt_local_executor.auth.keyring.delete_password", self._delete)

    def _get(self, service: str, user: str) -> str | None:
        value = self.passwords.get((service, user))
        after_read, self.after_read = self.after_read, None
        if after_read is not None:
            after_read()
        return value

    def _set(self, service: str, user: str, value: str) -> None:
        if self.failures:
            raise self.failures.pop(0)
        self.passwords[(service, user)] = value

    def _delete(self, service: str, user: str) -> None:
        if self.passwords.pop((service, user), None) is None:
            raise keyring.errors.PasswordDeleteError("not found")


class _Platform:
    """The platform's token and revocation endpoints. A refresh token works
    once, as the platform rotates it on every exchange, and one sent again
    is refused. `sent` lists the refresh tokens exchanged and `revoked` the
    tokens revoked, in order. `before_answering`, a coroutine function, runs
    once the first exchange has gone through, before its answer comes back."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.sent: list[str] = []
        self.revoked: list[str] = []
        self.before_answering: Callable[[], Awaitable[None]] | None = None
        real_client = httpx.AsyncClient
        monkeypatch.setattr(
            "autogpt_local_executor.auth.httpx.AsyncClient",
            lambda: real_client(transport=httpx.MockTransport(self._answer)),
        )

    async def _answer(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/revoke"):
            self.revoked.append(json.loads(request.content)["token"])
            return httpx.Response(200, json={})
        token = urllib.parse.parse_qs(request.content.decode())["refresh_token"][0]
        used = token in self.sent
        self.sent.append(token)
        number = len(self.sent)
        if used:
            return httpx.Response(400, json={"detail": "refresh token has been revoked"})
        if number == 1 and self.before_answering is not None:
            await self.before_answering()
        tokens = {"access_token": f"access-{number}", "refresh_token": f"refresh-{number}"}
        return httpx.Response(200, json=tokens)


class _AnotherProcess:
    """What another shim process does with the stored sign-in, run in a
    thread with an event loop of its own. It shares nothing with the test's
    task, so it waits for a turn the task holds as a process would."""

    def __init__(self, work: Callable[[], Awaitable[Any]]) -> None:
        self._work = work
        self.done = threading.Event()
        self._outcome: Any = None
        self._error: BaseException | None = None
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        async def work() -> Any:
            return await self._work()

        try:
            self._outcome = asyncio.run(work())
        except BaseException as exc:
            self._error = exc
        finally:
            self.done.set()

    def result(self) -> Any:
        assert self.done.wait(120), "the other process never finished"
        if self._error is not None:
            raise self._error
        return self._outcome


@pytest.fixture
def waiting() -> Iterator[threading.Event]:
    """Set once a process says, in its log, that it is waiting for another
    to finish with the stored sign-in."""
    event = threading.Event()

    class Watch(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if "Waiting for another" in record.getMessage():
                event.set()

    logger = logging.getLogger("autogpt_local_executor.auth")
    handler, level = Watch(), logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield event
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)


def _settled(other: _AnotherProcess, waiting: threading.Event) -> None:
    """Wait until the other process has finished, or waits for its turn."""
    give_up = time.monotonic() + 120
    while not (other.done.is_set() or waiting.is_set()):
        assert time.monotonic() < give_up, "the other process neither finished nor waited"
        time.sleep(0.01)


def _threads_in_the_sign_in_code() -> list[str]:
    """The function each thread now running the sign-in's code is in."""
    running = []
    for frame in sys._current_frames().values():
        current: Any = frame
        while current is not None:
            if current.f_code.co_filename == auth_module.__file__:
                running.append(current.f_code.co_name)
                break
            current = current.f_back
    return running


async def test_two_processes_that_refresh_at_once_send_one_exchange(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, waiting: threading.Event
) -> None:
    """Each daemon process had a refresh lock of its own, and two for one
    deployment share one keychain record: both read the same refresh token
    and sent it. The platform rotates the token on the first exchange and
    takes the second for reuse, which ends the sign-in for every session of
    the deployment. The second process now waits its turn, finds the pair
    the first stored, and sends nothing."""
    _Keychain(monkeypatch)
    platform = _Platform(monkeypatch)
    config = _config(tmp_path)
    store = KeychainTokenStore(config)
    await store.store_tokens("access-0", "refresh-0")
    others: list[_AnotherProcess] = []

    async def second_process_refreshes() -> None:
        other = _AnotherProcess(lambda: _refresh(_daemon(config)))
        others.append(other)
        # Until it has sent the same refresh token too, or waits its turn.
        await asyncio.to_thread(_settled, other, waiting)

    platform.before_answering = second_process_refreshes

    token = await _refresh(_daemon(config, store))

    [other] = others
    assert other.result() == "access-1"
    assert token == "access-1"
    assert platform.sent == ["refresh-0"]
    assert await store.get_access_token() == "access-1"
    assert await store.get_refresh_token() == "refresh-1"


async def test_a_sign_in_stored_as_a_refresh_writes_its_pair_is_the_one_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, waiting: threading.Event
) -> None:
    """A refresh stores its new pair only while the keychain still holds the
    refresh token it used up, which it reads just before it writes. A
    sign-in `autogpt-shim auth` stored between that read and that write was
    replaced by the older sign-in's refreshed pair. The sign-in now waits
    for the refresh's turn to end, and is the one the keychain holds."""
    keychain = _Keychain(monkeypatch)
    platform = _Platform(monkeypatch)
    config = _config(tmp_path)
    store = KeychainTokenStore(config)
    await store.store_tokens("access-0", "refresh-0")
    others: list[_AnotherProcess] = []

    def sign_in_meanwhile() -> None:
        other = _AnotherProcess(
            lambda: KeychainTokenStore(config).store_tokens("access-auth", "refresh-auth")
        )
        others.append(other)
        _settled(other, waiting)

    async def once_the_exchange_is_through() -> None:
        # The keychain's next read is the one just before the refresh's write.
        keychain.after_read = sign_in_meanwhile

    platform.before_answering = once_the_exchange_is_through

    await _refresh(_daemon(config, store))

    [other] = others
    other.result()
    assert platform.sent == ["refresh-0"]
    assert await store.get_access_token() == "access-auth"
    assert await store.get_refresh_token() == "refresh-auth"


async def test_a_sign_out_as_a_refresh_writes_its_pair_leaves_the_computer_signed_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, waiting: threading.Event
) -> None:
    """`autogpt-shim revoke` run between a refresh's read of the keychain and
    its write revoked the pair the refresh had just used up and removed the
    record, and the refresh then wrote its new pair back: the computer was
    signed in again, with tokens nobody had revoked. The sign-out now waits
    for the refresh's turn to end, so it revokes the pair the refresh
    stored, and the record stays removed."""
    keychain = _Keychain(monkeypatch)
    platform = _Platform(monkeypatch)
    config = _config(tmp_path)
    store = KeychainTokenStore(config)
    await store.store_tokens("access-0", "refresh-0")
    others: list[_AnotherProcess] = []

    def sign_out_meanwhile() -> None:
        other = _AnotherProcess(
            lambda: OAuthFlow(config, KeychainTokenStore(config)).revoke_tokens()
        )
        others.append(other)
        _settled(other, waiting)

    async def once_the_exchange_is_through() -> None:
        keychain.after_read = sign_out_meanwhile

    platform.before_answering = once_the_exchange_is_through

    await _refresh(_daemon(config, store))

    [other] = others
    other.result()
    assert platform.sent == ["refresh-0"]
    assert platform.revoked == ["access-1", "refresh-1"]
    assert await store.get_access_token() is None
    assert await store.get_refresh_token() is None


async def test_a_refresh_that_cannot_have_its_turn_in_time_sends_no_exchange(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The turn is held across an exchange with the platform, so a process
    can hold it for a while. One that can't have it by its deadline fails
    the refresh as something to try again, as a locked keychain does, and
    sends no exchange of its own: the refresh token it would send may be the
    one the holder is using up. Nothing is left waiting for the lock in a
    worker thread, which would hold up the shim's exit."""
    _Keychain(monkeypatch)
    platform = _Platform(monkeypatch)
    monkeypatch.setattr(auth_module, "TURN_WAIT_SECONDS", 0.5, raising=False)
    config = _config(tmp_path)
    store = KeychainTokenStore(config)
    await store.store_tokens("access-0", "refresh-0")
    daemon = _daemon(config, store)

    # Held as another process holds it: through a handle of its own.
    with _file_lock(_lock_file(store)):
        with pytest.raises(KeychainUnavailableError):
            await _refresh(daemon)

        assert platform.sent == []
        assert _threads_in_the_sign_in_code() == []

    assert await _refresh(daemon) == "access-1"
    assert platform.sent == ["refresh-0"]


async def test_a_refreshed_pair_the_keychain_would_not_store_keeps_its_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refresh whose new pair the keychain failed to store keeps the pair
    in memory to store on its next try, and the keychain keeps the refresh
    token the exchange used up. Another process that refreshed meanwhile
    read that token and sent it, which the platform takes for reuse. The
    process holding the pair now keeps the turn with it, so the other one
    waits, and uses the pair once it is stored."""
    keychain = _Keychain(monkeypatch)
    platform = _Platform(monkeypatch)
    config = _config(tmp_path)
    store = KeychainTokenStore(config)
    await store.store_tokens("access-0", "refresh-0")
    daemon = _daemon(config, store)
    keychain.failures.append(keyring.errors.PasswordSetError("the keychain is locked"))
    with pytest.raises(KeychainUnavailableError):
        await _refresh(daemon)

    with monkeypatch.context() as short_wait:
        short_wait.setattr(auth_module, "TURN_WAIT_SECONDS", 0.5, raising=False)
        with pytest.raises(KeychainUnavailableError):
            _AnotherProcess(lambda: _refresh(_daemon(config))).result()
    assert platform.sent == ["refresh-0"]

    assert await _refresh(daemon) == "access-1"
    assert _AnotherProcess(lambda: _refresh(_daemon(config))).result() == "access-1"
    assert platform.sent == ["refresh-0"]
    assert await store.get_refresh_token() == "refresh-1"


@pytest.mark.parametrize("unusable", ["folder", "locking"])
async def test_a_home_whose_lock_file_cannot_be_used_still_signs_in_and_refreshes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    unusable: str,
) -> None:
    """A folder the shim can't write to, or a filesystem with no file locks,
    must not stop a sign-in or a refresh. They go ahead without the lock
    file, the shim says so once, and the refresh's own check of the
    keychain before it writes is what is left."""
    _Keychain(monkeypatch)
    platform = _Platform(monkeypatch)
    if unusable == "folder":
        # A file where the folder's parent belongs: the folder can't be made.
        (tmp_path / "in-the-way").write_text("", encoding="utf-8")
        monkeypatch.setenv("AUTOGPT_SHIM_FALLBACK_DIR", str(tmp_path / "in-the-way" / "state"))
    else:

        def no_locks(_handle: Any) -> bool:
            raise OSError(errno.ENOLCK, os.strerror(errno.ENOLCK))

        monkeypatch.setattr(auth_module, "_lock_without_waiting", no_locks, raising=False)
    config = _config(tmp_path)
    store = KeychainTokenStore(config)

    with caplog.at_level(logging.WARNING, logger="autogpt_local_executor.auth"):
        await store.store_tokens("access-0", "refresh-0")
        token = await _refresh(_daemon(config, store))

    assert token == "access-1"
    assert platform.sent == ["refresh-0"]
    assert await store.get_refresh_token() == "refresh-1"
    said = [record for record in caplog.records if record.name == "autogpt_local_executor.auth"]
    assert len(said) == 1
    assert str(_lock_file(store)) in said[0].getMessage()


async def test_a_records_lock_file_is_the_persons_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lock file sits in the person's own folder, named for the record,
    so every process of the account that reads the record takes the same
    one, and only the owner can open it: another account that could, could
    hold it, and every refresh would wait."""
    _Keychain(monkeypatch)
    config = _config(tmp_path)
    store = KeychainTokenStore(config)
    other = KeychainTokenStore(
        config.model_copy(update={"platform_url": "https://other.example.com"})
    )

    await store.store_tokens("access-0", "refresh-0")
    await other.store_tokens("other-access", "other-refresh")

    lock = _lock_file(store)
    assert lock.is_file()
    assert _lock_file(other).is_file()
    assert _lock_file(other) != lock
    if os.name != "nt":
        assert stat.S_IMODE(lock.stat().st_mode) == 0o600


async def test_an_exchange_that_never_answers_gives_the_turn_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The turn is held across the exchange, so a token endpoint that took
    the request and then went quiet would hold it, and every other process's
    sign-in, sign-out and refresh with it, for as long as it stayed quiet.
    The exchange gets a time limit, and the refresh fails as something to
    try again."""
    _Keychain(monkeypatch)
    platform = _Platform(monkeypatch)
    monkeypatch.setattr(auth_module, "EXCHANGE_SECONDS", 0.5, raising=False)
    config = _config(tmp_path)
    store = KeychainTokenStore(config)
    await store.store_tokens("access-0", "refresh-0")

    async def never() -> None:
        await asyncio.Event().wait()

    platform.before_answering = never

    with pytest.raises(TokenRefreshUnavailableError):
        await asyncio.wait_for(
            _daemon(config, store)._get_access_token(
                refresh_on_fail=True, rejected_token="access-0"
            ),
            timeout=30,
        )

    _AnotherProcess(
        lambda: KeychainTokenStore(config).store_tokens("access-auth", "refresh-auth")
    ).result()
    assert await store.get_refresh_token() == "refresh-auth"


async def test_a_refresh_cancelled_during_its_exchange_gives_the_turn_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _Keychain(monkeypatch)
    platform = _Platform(monkeypatch)
    config = _config(tmp_path)
    store = KeychainTokenStore(config)
    await store.store_tokens("access-0", "refresh-0")
    exchanging = asyncio.Event()

    async def hold_the_answer() -> None:
        exchanging.set()
        await asyncio.Event().wait()

    platform.before_answering = hold_the_answer
    refresh = asyncio.create_task(OAuthFlow(config, store).refresh_token())
    await asyncio.wait_for(exchanging.wait(), timeout=60)

    refresh.cancel()
    with pytest.raises(asyncio.CancelledError):
        await refresh

    _AnotherProcess(
        lambda: KeychainTokenStore(config).store_tokens("access-auth", "refresh-auth")
    ).result()
    assert await store.get_refresh_token() == "refresh-auth"


@pytest.mark.parametrize("command", ["auth", "revoke"])
def test_a_sign_in_or_sign_out_that_cannot_have_its_turn_says_why(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
) -> None:
    """`auth` and `revoke` wait for the turn too. One that can't have it,
    since another process keeps a refreshed pair it hasn't been able to
    store, said only to check the configuration, the network and the
    keychain."""
    from autogpt_local_executor import cli

    busy = getattr(auth_module, "SignInBusyError", RuntimeError)("another process holds it")

    async def refuse(*_args: Any) -> None:
        raise busy

    monkeypatch.setattr(KeychainTokenStore, "store_tokens", refuse)
    monkeypatch.setattr(KeychainTokenStore, "clear_tokens", refuse)
    monkeypatch.setattr(KeychainTokenStore, "get_access_token", refuse)
    config = _config(tmp_path)
    if command == "auth":
        # The browser's part of the sign-in, which this test isn't about.
        monkeypatch.setattr(
            OAuthFlow, "_bind_callback_server", lambda *_args, **_kwargs: (None, [], [], 41899)
        )
        monkeypatch.setattr(OAuthFlow, "_wait_for_callback", lambda *_args: "the-code")
        monkeypatch.setattr(auth_module.webbrowser, "open", lambda _url: True)

        async def exchanged(*_args: Any, **_kwargs: Any) -> dict[str, str]:
            return {"access_token": "access-0", "refresh_token": "refresh-0"}

        monkeypatch.setattr(OAuthFlow, "_exchange_code", exchanged)
        status = cli._cmd_auth(config)
    else:
        status = asyncio.run(cli._cmd_revoke(config))

    assert status == 1
    assert "another process holds it" in capsys.readouterr().out


# ── Real processes ───────────────────────────────────────────────────────


class _TokenEndpoint(ThreadingHTTPServer):
    """A token endpoint on this computer for shim processes to reach, which
    rotates the refresh token as the platform does. It holds the first
    exchange's answer until `release` is set."""

    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _TokenRequest)
        self.sent: list[str] = []
        self.first_received = threading.Event()
        self.second_received = threading.Event()
        self.release = threading.Event()
        self.released_in_time = True


class _TokenRequest(BaseHTTPRequestHandler):
    server: _TokenEndpoint

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers["Content-Length"])).decode()
        token = urllib.parse.parse_qs(body)["refresh_token"][0]
        endpoint = self.server
        used = token in endpoint.sent
        endpoint.sent.append(token)
        number = len(endpoint.sent)
        if number == 1:
            endpoint.first_received.set()
            endpoint.released_in_time = endpoint.release.wait(120)
        else:
            endpoint.second_received.set()
        if used:
            self._answer(400, {"detail": "refresh token has been revoked"})
        else:
            self._answer(
                200, {"access_token": f"access-{number}", "refresh_token": f"refresh-{number}"}
            )

    def _answer(self, status: int, payload: dict[str, str]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: Any) -> None:
        pass


class _ShimProcess:
    """One run of tests/stored_sign_in_process.py: a shim process with the
    test's keychain. `waiting` is set once it says it waits for another
    process to finish with the stored sign-in."""

    def __init__(self, environment: dict[str, str], *arguments: str) -> None:
        self.waiting = threading.Event()
        self.logged: list[str] = []
        self._process = subprocess.Popen(
            [sys.executable, "-m", "tests.stored_sign_in_process", *arguments],
            cwd=ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        self._reader = threading.Thread(target=self._read_log, daemon=True)
        self._reader.start()

    def _read_log(self) -> None:
        assert self._process.stderr is not None
        for line in self._process.stderr:
            self.logged.append(line)
            if "Waiting for another" in line:
                self.waiting.set()

    def answer(self) -> dict[str, Any]:
        """What the process printed, once it has ended."""
        try:
            self._process.wait(timeout=120)
        finally:
            self.stop()
        self._reader.join(timeout=60)
        assert self._process.stdout is not None
        printed = self._process.stdout.read()
        assert self._process.returncode == 0, "".join(self.logged)
        return json.loads(printed.splitlines()[-1])

    def stop(self) -> None:
        if self._process.poll() is None:
            self._process.kill()
            self._process.wait(timeout=60)


def test_two_real_processes_that_refresh_at_once_send_one_exchange(tmp_path: Path) -> None:
    """The same as for two daemons in one process, with the lock file
    between real processes: a file lock that excludes two handles of one
    process doesn't have to exclude two processes, nor the other way about.
    Each is a shim process of its own, with a keychain the test gives it."""
    environment = {
        name: value
        for name, value in os.environ.items()
        if name.lower() not in ("http_proxy", "https_proxy", "all_proxy")
    }
    environment["PYTHON_KEYRING_BACKEND"] = "tests.stored_sign_in_process.FileKeyring"
    environment["SIGN_IN_TEST_KEYCHAIN"] = str(tmp_path / "keychain.sqlite")
    endpoint = _TokenEndpoint()
    serving = threading.Thread(target=endpoint.serve_forever, daemon=True)
    serving.start()
    url = f"http://127.0.0.1:{endpoint.server_address[1]}"

    def shim(*arguments: str) -> _ShimProcess:
        return _ShimProcess(environment, url, str(tmp_path), *arguments)

    processes: list[_ShimProcess] = []
    try:
        assert shim("store", "access-0", "refresh-0").answer() == {"stored": True}
        first = shim("refresh", "access-0")
        processes.append(first)
        assert endpoint.first_received.wait(120), "".join(first.logged)
        second = shim("refresh", "access-0")
        processes.append(second)
        # Until the second has sent the same refresh token too, or waits.
        give_up = time.monotonic() + 120
        while not (endpoint.second_received.is_set() or second.waiting.is_set()):
            assert time.monotonic() < give_up, "".join(second.logged)
            time.sleep(0.01)
        endpoint.release.set()

        answers = [first.answer(), second.answer()]
        stored = shim("read").answer()
    finally:
        endpoint.release.set()
        for process in processes:
            process.stop()
        endpoint.shutdown()
        endpoint.server_close()

    assert endpoint.released_in_time
    assert endpoint.sent == ["refresh-0"]
    assert answers == [{"token": "access-1"}, {"token": "access-1"}]
    assert stored == {"access_token": "access-1", "refresh_token": "refresh-1"}
    # The lock file itself was between them, not a way round one that
    # couldn't be used.
    logged = "".join(first.logged + second.logged)
    assert "lock file" not in logged, logged
    assert list(Path(os.environ["AUTOGPT_SHIM_FALLBACK_DIR"]).glob(".oauth_v1_*.lock"))
