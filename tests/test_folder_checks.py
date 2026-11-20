"""A look at a folder that never returns doesn't take the machine down.

The control connection verifies and issues a chat's folder grant, and lists
folders for the picker, by looking at the folder. On a network share that
stopped answering (a laptop that left its network) each look never returned,
and it was waited for without a limit, holding the chat's lock, a request
slot and, for the picker, its lock. After a few requests every control
request on the computer got SHIM_OVERLOADED, and only a restart cleared it.

A look is now waited for a bounded time, in a thread of its own, and one
still running when its caller stopped waiting holds one thread for its
folder, however many requests come in.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import textwrap
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

import autogpt_local_executor.daemon as daemon_module
from autogpt_local_executor.audit import AuditWriter
from autogpt_local_executor.config import ShimConfig
from autogpt_local_executor.daemon import DaemonPreflightError, ShimDaemon
from autogpt_local_executor.directory_browser import DirectoryBrowser
from autogpt_local_executor.protocol import (
    ActivateSessionMessage,
    ActivateSessionPayload,
    AttachSessionMessage,
    AttachSessionPayload,
    DetachSessionMessage,
    DetachSessionPayload,
    DirectoryListRequestMessage,
    DirectoryListRequestPayload,
    ErrorCode,
    ErrorMessage,
    MessageType,
    RestoreSessionMessage,
    RestoreSessionPayload,
    SessionAttachedMessage,
    now_ts,
)

# The bound the tests give a look, and how long they wait for an answer
# that has to come by then: a hundred times as long.
BOUND = 0.5
WITHIN = 50.0
ROOT = Path(__file__).resolve().parent.parent


def _checks(**kwargs: Any) -> Any:
    from autogpt_local_executor.folder_checks import FolderChecks

    return FolderChecks(**kwargs)


def _timeout() -> type[Exception]:
    from autogpt_local_executor.folder_checks import FolderCheckTimeout

    return FolderCheckTimeout


class _Away:
    """A look at a folder on a share that went away: it returns only once
    the test lets it go, at its end. `looks` counts the threads that came
    in, and `threads` keeps them."""

    def __init__(self) -> None:
        self.looks = 0
        self.threads: list[threading.Thread] = []
        self.back = threading.Event()
        self._lock = threading.Lock()

    def __call__(self, *_args: Any, **_kwargs: Any) -> Any:
        with self._lock:
            self.looks += 1
            self.threads.append(threading.current_thread())
        self.back.wait(WITHIN * 4)
        return "back"


@pytest.fixture
def away() -> Any:
    share = _Away()
    yield share
    share.back.set()
    for thread in share.threads:
        thread.join(WITHIN)


async def _by(deadline_in: float) -> float:
    return asyncio.get_running_loop().time() + deadline_in


async def _until(condition: Callable[[], bool]) -> None:
    deadline = asyncio.get_running_loop().time() + WITHIN
    while not condition():
        assert asyncio.get_running_loop().time() < deadline, "timed out waiting"
        await asyncio.sleep(0.01)


# ── The looks themselves ─────────────────────────────────────────────────────


async def test_a_look_that_returns_gives_its_answer_or_its_failure() -> None:
    checks = _checks()

    def fails() -> None:
        raise PermissionError("not yours")

    assert await checks.run("/a", lambda: "there", deadline=await _by(WITHIN)) == "there"
    with pytest.raises(PermissionError, match="not yours"):
        await checks.run("/a", fails, deadline=await _by(WITHIN))
    assert checks.stuck() == 0


async def test_a_look_that_never_returns_is_given_up_on_at_its_deadline(away: _Away) -> None:
    checks = _checks()
    abandoned: list[bool] = []

    with pytest.raises(_timeout(), match="didn't answer in time"):
        await asyncio.wait_for(
            checks.run(
                "/share/gone",
                away,
                deadline=await _by(BOUND),
                on_abandoned=lambda: abandoned.append(True),
            ),
            timeout=WITHIN,
        )

    assert away.looks == 1
    assert checks.stuck() == 1
    assert abandoned == [True]


async def test_the_look_runs_in_a_thread_that_does_not_hold_the_shim_from_stopping(
    away: _Away,
) -> None:
    """asyncio.run waits for the event loop's worker threads, and so does
    the interpreter as it exits; a daemon thread of its own is waited for
    by neither."""
    checks = _checks()

    with pytest.raises(_timeout()):
        await checks.run("/share/gone", away, deadline=await _by(BOUND))

    [thread] = away.threads
    assert thread.daemon is True
    assert thread.name == "autogpt-shim-folder-check"
    assert getattr(asyncio.get_running_loop(), "_default_executor", None) is None


async def test_more_requests_for_a_folder_that_is_stuck_start_no_more_looks(away: _Away) -> None:
    checks = _checks()
    with pytest.raises(_timeout()):
        await checks.run("/share/gone", away, deadline=await _by(BOUND))
    started = asyncio.get_running_loop().time()

    for _ in range(20):
        with pytest.raises(_timeout(), match="still hasn't answered an earlier request"):
            await checks.run("/share/gone", away, deadline=await _by(WITHIN))

    # At once, with no thread of their own: none waited out its deadline.
    assert asyncio.get_running_loop().time() - started < WITHIN / 2
    assert away.looks == 1
    assert checks.stuck() == 1


async def test_another_folder_is_looked_at_while_one_is_stuck(away: _Away) -> None:
    checks = _checks()
    with pytest.raises(_timeout()):
        await checks.run("/share/gone", away, deadline=await _by(BOUND))

    assert await checks.run("/home/me", lambda: "there", deadline=await _by(WITHIN)) == "there"


async def test_a_folder_is_looked_at_again_once_its_stuck_look_returns(away: _Away) -> None:
    checks = _checks()
    with pytest.raises(_timeout()):
        await checks.run("/share/gone", away, deadline=await _by(BOUND))

    away.back.set()
    await _until(lambda: checks.stuck() == 0)

    assert await checks.run("/share/gone", lambda: "there", deadline=await _by(WITHIN)) == "there"


async def test_a_second_request_waits_for_a_look_whose_caller_is_still_waiting() -> None:
    """Two requests about one folder at once, on a share that answers: the
    second waits its turn and then looks for itself."""
    checks = _checks()
    first_may_end = threading.Event()
    order: list[str] = []

    def first() -> str:
        first_may_end.wait(WITHIN)
        order.append("first")
        return "one"

    def second() -> str:
        order.append("second")
        return "two"

    one = asyncio.ensure_future(checks.run("/a", first, deadline=await _by(WITHIN)))
    await asyncio.sleep(0)
    two = asyncio.ensure_future(checks.run("/a", second, deadline=await _by(WITHIN)))
    await asyncio.sleep(0.05)
    assert not two.done()
    first_may_end.set()

    assert await asyncio.wait_for(asyncio.gather(one, two), timeout=WITHIN) == ["one", "two"]
    assert order == ["first", "second"]


async def test_a_request_whose_turn_does_not_come_in_time_starts_no_look(away: _Away) -> None:
    checks = _checks()
    in_progress = asyncio.ensure_future(checks.run("/a", away, deadline=await _by(WITHIN)))
    await _until(lambda: away.looks == 1)

    with pytest.raises(_timeout(), match="didn't answer in time"):
        await checks.run("/a", away, deadline=await _by(BOUND))

    assert away.looks == 1
    away.back.set()
    assert await asyncio.wait_for(in_progress, timeout=WITHIN) == "back"


async def test_a_look_whose_caller_is_cancelled_counts_as_given_up_on(away: _Away) -> None:
    checks = _checks()
    abandoned: list[bool] = []
    request = asyncio.ensure_future(
        checks.run(
            "/share/gone",
            away,
            deadline=await _by(WITHIN),
            on_abandoned=lambda: abandoned.append(True),
        )
    )
    await _until(lambda: away.looks == 1)

    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request

    assert abandoned == [True]
    with pytest.raises(_timeout(), match="still hasn't answered"):
        await checks.run("/share/gone", away, deadline=await _by(WITHIN))
    assert away.looks == 1


@pytest.mark.parametrize("given_up_by", ["its-caller-cancelled", "its-deadline"])
async def test_a_request_waiting_its_turn_is_refused_once_the_look_ahead_is_given_up_on(
    away: _Away, given_up_by: str
) -> None:
    """A second request for a folder waits its turn behind the first's
    look. When that look was given up on, a request that came a moment
    later was refused at once, but the one already waiting went on waiting
    for the look, to its own deadline."""
    checks = _checks()
    first = asyncio.ensure_future(
        checks.run(
            "/share/gone",
            away,
            deadline=await _by(BOUND if given_up_by == "its-deadline" else WITHIN),
        )
    )
    await _until(lambda: away.looks == 1)
    second = asyncio.ensure_future(checks.run("/share/gone", away, deadline=await _by(WITHIN)))
    await asyncio.sleep(0.05)
    assert not second.done()

    if given_up_by == "its-caller-cancelled":
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
    else:
        with pytest.raises(_timeout(), match="didn't answer in time"):
            await asyncio.wait_for(first, timeout=WITHIN)

    # Then, not at its own deadline: that is further off than this waits.
    with pytest.raises(_timeout(), match="still hasn't answered"):
        await asyncio.wait_for(second, timeout=WITHIN / 5)
    assert away.looks == 1


async def test_a_request_waiting_for_a_thread_is_refused_once_every_look_is_given_up_on(
    away: _Away,
) -> None:
    """With every thread taken and the last look still waited for given up
    on, nothing frees a thread soon: the request that waited for one is
    refused then, as one that came a moment later is."""
    checks = _checks(max_stuck=2)
    with pytest.raises(_timeout()):
        await checks.run("/share/a", away, deadline=await _by(BOUND))
    last = asyncio.ensure_future(checks.run("/share/b", away, deadline=await _by(WITHIN)))
    await _until(lambda: away.looks == 2)
    waiting = asyncio.ensure_future(checks.run("/share/c", away, deadline=await _by(WITHIN)))
    await asyncio.sleep(0.05)
    assert not waiting.done()

    last.cancel()
    with pytest.raises(asyncio.CancelledError):
        await last

    # Then, not at its own deadline: that is further off than this waits.
    with pytest.raises(_timeout(), match="Too many folders"):
        await asyncio.wait_for(waiting, timeout=WITHIN / 5)
    assert away.looks == 2


async def test_only_so_many_looks_are_left_running_whatever_their_folders(away: _Away) -> None:
    checks = _checks(max_stuck=3)
    for folder in ("/share/a", "/share/b", "/share/c"):
        with pytest.raises(_timeout()):
            await checks.run(folder, away, deadline=await _by(BOUND))

    with pytest.raises(_timeout(), match="Too many folders"):
        await checks.run("/share/d", away, deadline=await _by(WITHIN))

    assert away.looks == 3
    assert checks.stuck() == 3


async def test_a_request_waiting_for_a_thread_goes_on_when_a_look_given_up_on_returns() -> None:
    """Every thread is taken: one look was given up on, and another's caller
    still waits. A third request waits for a thread, and it watched only the
    looks still waited for. So when the one given up on returned, and its
    thread was free, the request went on waiting for the other, to its own
    deadline, and was refused."""
    checks = _checks(max_stuck=2)
    given_up_on, waited_for = threading.Event(), threading.Event()

    def left_running() -> str:
        given_up_on.wait(WITHIN * 4)
        return "late"

    def still_waited_for() -> str:
        waited_for.wait(WITHIN * 4)
        return "held"

    with pytest.raises(_timeout()):
        await checks.run("/share/a", left_running, deadline=await _by(BOUND))
    holding = asyncio.ensure_future(
        checks.run("/share/b", still_waited_for, deadline=await _by(WITHIN * 2))
    )
    try:
        await _until(lambda: len(checks._looks) == 2)
        third = asyncio.ensure_future(
            checks.run("/share/c", lambda: "third", deadline=await _by(WITHIN / 5))
        )
        await asyncio.sleep(0.05)
        assert not third.done()

        given_up_on.set()

        assert await asyncio.wait_for(third, timeout=WITHIN) == "third"
    finally:
        given_up_on.set()
        waited_for.set()
        assert await asyncio.wait_for(holding, timeout=WITHIN) == "held"


async def test_an_answer_that_comes_after_the_deadline_is_dropped(away: _Away) -> None:
    checks = _checks()
    with pytest.raises(_timeout()):
        await checks.run("/share/gone", away, deadline=await _by(BOUND))

    away.back.set()
    await _until(lambda: checks.stuck() == 0)
    # Its failure, had it failed, would be taken too: nothing is reported
    # as never retrieved once the loop collects the look.
    assert checks.stuck() == 0


# ── The control connection ───────────────────────────────────────────────────


@pytest.fixture
def quick(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(daemon_module, "FOLDER_CHECK_SECONDS", BOUND, raising=False)


def _control_daemon(tmp_path: Path) -> ShimDaemon:
    config = ShimConfig(
        session_id=None,
        machine_id="machine-1",
        allowed_root=tmp_path / "unused-global-root",
        audit_log_path=tmp_path / "audit.log",
    )
    audit = AuditWriter(config.audit_log_path, b"a" * 32, machine_id=config.machine_id)
    daemon = ShimDaemon(config, token_store=object(), audit=audit)
    daemon._directory_browser = DirectoryBrowser(platform_name="linux", home=tmp_path)
    daemon._granted_capabilities = frozenset({"directory_browse"})
    daemon._start_child_session = AsyncMock()  # type: ignore[method-assign]
    return daemon


async def _answer(daemon: ShimDaemon, message: Any) -> Any:
    """The daemon's answer to a control request, which has to come within
    WITHIN whatever the folder does."""
    return await asyncio.wait_for(daemon._handle(message), timeout=WITHIN)


def _list(browse_id: str | None = None, directory_ref: str | None = None) -> Any:
    return DirectoryListRequestMessage(
        id="list",
        ts=now_ts(),
        payload=DirectoryListRequestPayload(browse_id=browse_id, directory_ref=directory_ref),
    )


async def _browse_to(daemon: ShimDaemon, name: str) -> tuple[str, str]:
    roots = await _answer(daemon, _list())
    assert roots.type == MessageType.DIRECTORY_LIST_RESPONSE, roots
    home = roots.payload.entries[0]
    listing = await _answer(daemon, _list(roots.payload.browse_id, home.directory_ref))
    assert listing.type == MessageType.DIRECTORY_LIST_RESPONSE, listing
    child = next(entry for entry in listing.payload.entries if entry.name == name)
    return listing.payload.browse_id, child.directory_ref


def _attach(session_id: str, browse_id: str, directory_ref: str) -> Any:
    return AttachSessionMessage(
        id="attach",
        ts=now_ts(),
        payload=AttachSessionPayload(
            session_id=session_id, browse_id=browse_id, directory_ref=directory_ref
        ),
    )


def _restore(session_id: str, root_grant: str) -> Any:
    return RestoreSessionMessage(
        id="restore",
        ts=now_ts(),
        payload=RestoreSessionPayload(session_id=session_id, root_grant=root_grant),
    )


async def _attached(daemon: ShimDaemon, session_id: str, name: str) -> SessionAttachedMessage:
    """Attach `session_id` to the folder `name` in the home folder."""
    browse_id, directory_ref = await _browse_to(daemon, name)
    attached = await _answer(daemon, _attach(session_id, browse_id, directory_ref))
    assert isinstance(attached, SessionAttachedMessage), attached
    return attached


def _assert_not_answering(response: Any) -> None:
    assert isinstance(response, ErrorMessage), response
    assert response.payload.code == ErrorCode.DIRECTORY_UNAVAILABLE
    assert response.payload.details == {"reason": "not_answering"}


def _folders(tmp_path: Path, *names: str) -> None:
    for name in names:
        (tmp_path / name).mkdir()


def _away_for(away: _Away, folder: Path, real: Callable[..., Any]) -> Callable[..., Any]:
    """`real`, except that a call naming `folder` never returns."""

    def look(*args: Any, **kwargs: Any) -> Any:
        if any(str(folder) in str(argument) for argument in args):
            return away(*args, **kwargs)
        return real(*args, **kwargs)

    return look


def _verify_away_for(daemon: ShimDaemon, away: _Away, gone: SessionAttachedMessage) -> None:
    real = daemon._root_grants.verify

    def verify(root_grant: str, session_id: str) -> Any:
        if root_grant == gone.payload.root_grant:
            away()
        return real(root_grant, session_id)

    daemon._root_grants.verify = verify  # type: ignore[method-assign]


async def test_a_restore_whose_folder_never_answers_is_answered_and_lets_go_of_the_chat(
    tmp_path: Path, quick: None, away: _Away
) -> None:
    _folders(tmp_path, "on-the-share", "at-home")
    daemon = _control_daemon(tmp_path)
    gone = await _attached(daemon, "chat-1", "on-the-share")
    here = await _attached(daemon, "chat-2", "at-home")
    _verify_away_for(daemon, away, gone)

    response = await _answer(daemon, _restore("chat-1", gone.payload.root_grant))

    _assert_not_answering(response)
    # The chat's lock and the picker's are free, and so is every other chat.
    assert daemon._session_locks == {}
    assert not daemon._browser_lock.locked()
    restored = await _answer(daemon, _restore("chat-2", here.payload.root_grant))
    assert restored.type == MessageType.SESSION_RESTORED, restored
    detached = await _answer(
        daemon,
        DetachSessionMessage(
            id="detach", ts=now_ts(), payload=DetachSessionPayload(session_id="chat-1")
        ),
    )
    assert detached.type == MessageType.ACK, detached
    await asyncio.wait_for(daemon._reap_idle_children(), timeout=WITHIN)


async def test_more_restores_for_a_stuck_folder_are_answered_at_once_and_park_no_thread(
    tmp_path: Path, quick: None, away: _Away, monkeypatch: pytest.MonkeyPatch
) -> None:
    _folders(tmp_path, "on-the-share")
    daemon = _control_daemon(tmp_path)
    gone = await _attached(daemon, "chat-1", "on-the-share")
    _verify_away_for(daemon, away, gone)
    _assert_not_answering(await _answer(daemon, _restore("chat-1", gone.payload.root_grant)))
    # From here a look would be waited for far longer than the test allows:
    # an answer that comes at all didn't wait for one.
    monkeypatch.setattr(daemon_module, "FOLDER_CHECK_SECONDS", WITHIN * 10)

    for _ in range(daemon.config.max_concurrent * 3):
        _assert_not_answering(await _answer(daemon, _restore("chat-1", gone.payload.root_grant)))

    assert away.looks == 1
    assert daemon._folder_checks.stuck() == 1


async def test_a_restore_that_answers_after_its_request_was_answered_starts_no_chat(
    tmp_path: Path, quick: None, away: _Away
) -> None:
    _folders(tmp_path, "on-the-share")
    daemon = _control_daemon(tmp_path)
    gone = await _attached(daemon, "chat-1", "on-the-share")
    bindings = dict(daemon._session_bindings)
    _verify_away_for(daemon, away, gone)
    _assert_not_answering(await _answer(daemon, _restore("chat-1", gone.payload.root_grant)))

    away.back.set()
    await _until(lambda: daemon._folder_checks.stuck() == 0)
    await asyncio.sleep(0.05)

    daemon._start_child_session.assert_not_awaited()  # type: ignore[attr-defined]
    assert dict(daemon._session_bindings) == bindings
    assert daemon._child_sessions == {}


async def test_an_activation_whose_folder_never_answers_is_answered(
    tmp_path: Path, quick: None, away: _Away
) -> None:
    _folders(tmp_path, "on-the-share")
    daemon = _control_daemon(tmp_path)
    gone = await _attached(daemon, "chat-1", "on-the-share")
    _verify_away_for(daemon, away, gone)

    response = await _answer(
        daemon,
        ActivateSessionMessage(
            id="activate",
            ts=now_ts(),
            payload=ActivateSessionPayload(session_id="chat-1", revision=1),
        ),
    )

    _assert_not_answering(response)
    assert daemon._session_locks == {}
    daemon._start_child_session.assert_not_awaited()  # type: ignore[attr-defined]


async def test_an_attach_whose_grant_is_never_issued_is_answered_and_attaches_nothing(
    tmp_path: Path, quick: None, away: _Away
) -> None:
    _folders(tmp_path, "on-the-share", "at-home")
    daemon = _control_daemon(tmp_path)
    browse_id, directory_ref = await _browse_to(daemon, "on-the-share")
    daemon._root_grants.issue = _away_for(  # type: ignore[method-assign]
        away, tmp_path / "on-the-share", daemon._root_grants.issue
    )

    response = await _answer(daemon, _attach("chat-1", browse_id, directory_ref))

    _assert_not_answering(response)
    assert daemon._session_locks == {}
    assert not daemon._browser_lock.locked()
    assert "chat-1" not in daemon._session_bindings
    # Another folder attaches, and the grant issued late attaches nothing.
    assert (await _attached(daemon, "chat-2", "at-home")).payload.revision == 1
    away.back.set()
    await _until(lambda: daemon._folder_checks.stuck() == 0)
    await asyncio.sleep(0.05)
    assert "chat-1" not in daemon._session_bindings


async def test_an_attach_whose_folder_never_resolves_is_answered_and_frees_the_picker(
    tmp_path: Path, quick: None, away: _Away, monkeypatch: pytest.MonkeyPatch
) -> None:
    _folders(tmp_path, "on-the-share", "at-home")
    daemon = _control_daemon(tmp_path)
    browse_id, directory_ref = await _browse_to(daemon, "on-the-share")
    browser = daemon._directory_browser
    monkeypatch.setattr(
        DirectoryBrowser,
        "_canonical_accessible_directory",
        _away_for_method(away, tmp_path / "on-the-share"),
    )

    response = await _answer(daemon, _attach("chat-1", browse_id, directory_ref))

    _assert_not_answering(response)
    assert not daemon._browser_lock.locked()
    # The look may still be in the old picker, so there is a new one.
    assert daemon._directory_browser is not browser
    assert (await _attached(daemon, "chat-2", "at-home")).payload.revision == 1


def _away_for_method(away: _Away, folder: Path) -> Callable[..., Any]:
    """DirectoryBrowser._canonical_accessible_directory, whose next look at
    `folder` never returns: the share goes away as the folder is picked."""
    real = DirectoryBrowser._canonical_accessible_directory

    def look(self: DirectoryBrowser, path: Path, **how: Any) -> Path:
        if Path(path) == folder and away.looks == 0:
            away()
        return real(self, path, **how)

    return look


async def test_a_listing_that_never_returns_is_answered_and_other_folders_still_list(
    tmp_path: Path, quick: None, away: _Away, monkeypatch: pytest.MonkeyPatch
) -> None:
    _folders(tmp_path, "on-the-share", "at-home")
    (tmp_path / "at-home" / "inside").mkdir()
    daemon = _control_daemon(tmp_path)
    browse_id, gone_ref = await _browse_to(daemon, "on-the-share")
    browser = daemon._directory_browser
    real = DirectoryBrowser._list_children

    def list_children(self: DirectoryBrowser, session: Any, current: Path, *rest: Any) -> Any:
        if current == tmp_path / "on-the-share":
            away()
        return real(self, session, current, *rest)

    monkeypatch.setattr(DirectoryBrowser, "_list_children", list_children)

    response = await _answer(daemon, _list(browse_id, gone_ref))

    _assert_not_answering(response)
    assert not daemon._browser_lock.locked()
    assert daemon._directory_browser is not browser
    # The old picker's references went with it, as after a reconnect.
    stale = await _answer(daemon, _list(browse_id, gone_ref))
    assert stale.payload.code == ErrorCode.DIRECTORY_REFERENCE_INVALID
    # Another folder lists, and the stuck one is refused at once, with no
    # second look: its answer comes though a look would wait far longer.
    browse_id, home_ref = await _browse_to(daemon, "at-home")
    listed = await _answer(daemon, _list(browse_id, home_ref))
    assert [entry.name for entry in listed.payload.entries] == ["inside"]
    browse_id, gone_ref = await _browse_to(daemon, "on-the-share")
    monkeypatch.setattr(daemon_module, "FOLDER_CHECK_SECONDS", WITHIN * 10)
    _assert_not_answering(await _answer(daemon, _list(browse_id, gone_ref)))
    assert away.looks == 1


class _FakeWS:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))


async def test_requests_for_a_stuck_folder_leave_every_request_slot_free(
    tmp_path: Path, quick: None, away: _Away
) -> None:
    """Each RESTORE for the chat parked a slot; after max_concurrent of them
    every control request got SHIM_OVERLOADED."""
    _folders(tmp_path, "on-the-share", "at-home")
    daemon = _control_daemon(tmp_path)
    gone = await _attached(daemon, "chat-1", "on-the-share")
    _verify_away_for(daemon, away, gone)
    daemon._semaphore = asyncio.Semaphore(daemon.config.max_concurrent)
    ws = _FakeWS()

    await asyncio.wait_for(
        asyncio.gather(
            *(
                daemon._dispatch(ws, _restore("chat-1", gone.payload.root_grant))
                for _ in range(daemon.config.max_concurrent * 2)
            )
        ),
        timeout=WITHIN,
    )

    assert [frame["payload"]["code"] for frame in ws.sent if frame["type"] == "ERROR"] == [
        "DIRECTORY_UNAVAILABLE"
    ] * (daemon.config.max_concurrent * 2)
    assert (daemon._pending_requests, daemon._in_flight, daemon._queue_depth) == (0, 0, 0)
    assert away.looks == 1
    await asyncio.wait_for(daemon._dispatch(ws, _list()), timeout=WITHIN)
    assert ws.sent[-1]["type"] == "DIRECTORY_LIST_RESPONSE"


async def test_a_bad_grant_is_refused_without_a_look_at_any_folder(
    tmp_path: Path, quick: None, away: _Away
) -> None:
    _folders(tmp_path, "on-the-share")
    daemon = _control_daemon(tmp_path)
    gone = await _attached(daemon, "chat-1", "on-the-share")
    daemon._root_grants.verify = away  # type: ignore[method-assign]

    forged = gone.payload.root_grant[:-4] + "AAAA"
    response = await _answer(daemon, _restore("chat-1", forged))
    another = await _answer(daemon, _restore("chat-2", gone.payload.root_grant))

    assert response.payload.code == ErrorCode.ROOT_GRANT_INVALID
    assert another.payload.code == ErrorCode.ROOT_GRANT_INVALID
    assert away.looks == 0


async def test_a_sessions_folder_that_never_answers_ends_its_start_with_a_failure_to_retry(
    tmp_path: Path, quick: None, away: _Away, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A chat's session, and a legacy one, make their folder as they start."""
    root = tmp_path / "on-the-share"
    config = ShimConfig(
        session_id="chat-1",
        machine_id="machine-1",
        allowed_root=root,
        audit_log_path=tmp_path / "audit.log",
    )
    audit = AuditWriter(config.audit_log_path, b"a" * 32, machine_id=config.machine_id)
    daemon = ShimDaemon(config, token_store=object(), audit=audit)
    monkeypatch.setattr(daemon, "_preflight_or_raise", AsyncMock())
    real = Path.mkdir

    def mkdir(self: Path, *args: Any, **kwargs: Any) -> None:
        if self == root:
            away()
        real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", mkdir)

    with pytest.raises(DaemonPreflightError, match="did not answer") as failure:
        await asyncio.wait_for(daemon.run(), timeout=WITHIN)

    assert failure.value.transient is True
    assert away.threads[0].daemon is True


def test_the_shim_can_still_stop_while_a_folder_never_answers(tmp_path: Path) -> None:
    """In a process of its own, which has to end: a look left in one of the
    event loop's worker threads kept asyncio.run from returning, and with it
    the exit the shim's supervisor restarts it after."""
    script = textwrap.dedent(
        """
        import asyncio, sys, threading
        from pathlib import Path

        import autogpt_local_executor.daemon as daemon_module
        from autogpt_local_executor.audit import AuditWriter
        from autogpt_local_executor.config import ShimConfig
        from autogpt_local_executor.daemon import ShimDaemon
        from autogpt_local_executor.protocol import (
            RestoreSessionMessage, RestoreSessionPayload, now_ts,
        )

        folder = Path(sys.argv[1])
        daemon_module.FOLDER_CHECK_SECONDS = 0.5


        async def main() -> None:
            config = ShimConfig(
                session_id=None,
                machine_id="machine-1",
                allowed_root=folder / "unused",
                audit_log_path=folder / "audit.log",
            )
            audit = AuditWriter(config.audit_log_path, b"a" * 32, machine_id="machine-1")
            daemon = ShimDaemon(config, token_store=object(), audit=audit)
            daemon._granted_capabilities = frozenset({"directory_browse"})
            grant = daemon._root_grants.issue("chat-1", folder, 1).root_grant
            # The folder's share goes away: a look at it never returns.
            daemon._root_grants.verify = lambda *_: threading.Event().wait()
            response = await daemon._handle(
                RestoreSessionMessage(
                    id="restore",
                    ts=now_ts(),
                    payload=RestoreSessionPayload(session_id="chat-1", root_grant=grant),
                )
            )
            print(response.payload.code.value, flush=True)


        asyncio.run(main())
        print("stopped", flush=True)
        """
    )

    ran = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=ROOT,
        timeout=WITHIN * 2,
        check=False,
    )

    assert ran.stdout.split() == ["DIRECTORY_UNAVAILABLE", "stopped"], ran.stderr
    assert ran.returncode == 0, ran.stderr


# ── Review of the looks' own limits ──────────────────────────────────────────


async def test_two_folders_whose_names_differ_only_in_case_are_two_folders(
    away: _Away, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows can have two folders side by side whose names differ only in
    case (in a folder marked case-sensitive, as WSL makes them). The looks
    were kept by each path with its case folded the way Windows folds one,
    so while `Foo` wasn't answering, `foo` was refused with no look at it.
    Run here with Windows' folding, whatever the OS."""
    import ntpath

    from autogpt_local_executor import folder_checks

    monkeypatch.setattr(folder_checks.os.path, "normcase", ntpath.normcase)
    checks = _checks()
    with pytest.raises(_timeout()):
        await checks.run(r"C:\Work\Foo", away, deadline=await _by(BOUND))

    assert await checks.run(r"C:\Work\foo", lambda: "there", deadline=await _by(WITHIN)) == "there"
    # The folder that is away, asked about again by its name, has its one look.
    with pytest.raises(_timeout(), match="still hasn't answered an earlier request"):
        await checks.run(r"C:\Work\Foo", away, deadline=await _by(WITHIN))
    assert away.looks == 1


async def test_looks_under_way_count_toward_the_most_left_running(away: _Away) -> None:
    """The limit was checked against the looks already given up on. With one
    short of it given up on, two requests about two other folders both
    passed the check and both started a thread; when neither folder
    answered, one more thread was left running than the limit allows."""
    checks = _checks(max_stuck=2)
    with pytest.raises(_timeout()):
        await checks.run("/share/one", away, deadline=await _by(BOUND))
    assert checks.stuck() == 1

    answers = await asyncio.wait_for(
        asyncio.gather(
            checks.run("/share/two", away, deadline=await _by(BOUND)),
            checks.run("/share/three", away, deadline=await _by(BOUND)),
            return_exceptions=True,
        ),
        timeout=WITHIN,
    )

    assert [type(answer) for answer in answers] == [_timeout(), _timeout()]
    # One of the two had the last thread; the other never started one.
    assert away.looks == 2
    assert checks.stuck() == 2


async def test_a_burst_of_looks_at_folders_that_answer_all_get_their_answers() -> None:
    """More requests at once than threads may run. The later ones wait for
    one to finish, within their own time, and then look for themselves:
    none is refused because others are under way."""
    import functools
    import time

    checks = _checks(max_stuck=2)
    lock = threading.Lock()
    under_way = 0
    most = 0

    def look(folder: int) -> int:
        nonlocal under_way, most
        with lock:
            under_way += 1
            most = max(most, under_way)
        time.sleep(0.05)
        with lock:
            under_way -= 1
        return folder

    deadline = await _by(WITHIN)
    answers = await asyncio.gather(
        *(
            checks.run(f"/home/me/{folder}", functools.partial(look, folder), deadline=deadline)
            for folder in range(6)
        )
    )

    assert answers == list(range(6))
    assert most <= 2


async def test_with_every_thread_held_by_a_folder_given_up_on_a_look_is_refused_at_once(
    away: _Away,
) -> None:
    checks = _checks(max_stuck=2)
    for folder in ("/share/one", "/share/two"):
        with pytest.raises(_timeout()):
            await checks.run(folder, away, deadline=await _by(BOUND))
    started = asyncio.get_running_loop().time()

    with pytest.raises(_timeout(), match="Too many folders"):
        await checks.run("/home/me", lambda: "there", deadline=await _by(WITHIN))

    assert asyncio.get_running_loop().time() - started < WITHIN / 2
    assert away.looks == 2
