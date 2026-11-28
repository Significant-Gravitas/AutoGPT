"""An edit lock stays put while an edit holds it or waits for it, however
many other files are edited meanwhile."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock

import pytest

from backend.copilot.sdk.e2b_file_tools import (
    _edit_lock,
    _edit_locks,
    _handle_edit_file,
)
from backend.copilot.tools.local_pc_shim import LocalPCShim, ShimHello, _FilesProxy


def _shim(
    allowed_root: str = "/Users/test/workspace", platform: str = "darwin"
) -> LocalPCShim:
    """Build a LocalPCShim instance with _rpc stubbed."""
    shim = LocalPCShim.__new__(LocalPCShim)
    hello = ShimHello(
        machine_id="m", platform=platform, arch="arm64", allowed_root=allowed_root
    )
    shim.sandbox_id = "s1"
    shim.machine_id = hello.machine_id
    shim.platform = hello.platform
    shim.arch = hello.arch
    shim.allowed_root = hello.allowed_root
    shim.capabilities = hello.capabilities
    shim._rpc = AsyncMock(return_value={"payload": {"exists": True}})
    shim.files = _FilesProxy(shim)
    return shim


def _contents(data: bytes) -> dict:
    return {
        "type": "FILE_CONTENTS",
        "payload": {"content": base64.b64encode(data).decode("ascii")},
    }


# Edits of more files than this must not cost a file its lock while an edit
# holds it or waits for it, as a map that keeps only its newest thousand would.
_LOCKS_THE_MAP_KEPT = 1_000


async def _until(condition: Callable[[], bool]) -> None:
    """Wait for what another task brings about. The limit only fails a test
    that would otherwise hang."""
    async with asyncio.timeout(30):
        while not condition():
            await asyncio.sleep(0.001)


async def _end(*tasks: asyncio.Task[Any] | None) -> None:
    """End the edits a test started, so one left waiting by a failed check
    doesn't keep its claim on a lock into the next test."""
    started = [task for task in tasks if task is not None]
    for task in started:
        task.cancel()
    await asyncio.gather(*started, return_exceptions=True)


class TestAnEditLockInUseIsKept:
    """An edit of a file whose lock was dropped took a fresh lock and ran
    beside the edit still holding the old one, so one of the two writes was
    lost."""

    @pytest.mark.asyncio
    async def test_an_edit_waits_for_one_that_began_a_thousand_files_ago(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        shim = _shim(allowed_root="/Users/test/ws")
        held = "/Users/test/ws/notes.txt"
        stored = {held: b"one two"}
        first_reading = asyncio.Event()
        second_read = asyncio.Event()
        first_may_finish = asyncio.Event()

        async def rpc(op: str, payload: dict, **_kwargs) -> dict:
            path = payload["path"]
            if op != "FILE_READ":
                stored[path] = payload["content"].encode()
                return {"type": "ACK", "payload": {"ok": True}}
            data = stored.setdefault(path, b"old")
            if path == held and first_reading.is_set():
                second_read.set()
            elif path == held:
                first_reading.set()
                await first_may_finish.wait()
            return _contents(data)

        shim._rpc = rpc
        monkeypatch.setattr(
            "backend.copilot.sdk.e2b_file_tools._get_sandbox", lambda: shim
        )

        first = asyncio.create_task(
            _handle_edit_file(
                {"file_path": "notes.txt", "old_string": "one", "new_string": "1"}
            )
        )
        second: asyncio.Task[dict[str, Any]] | None = None
        try:
            await asyncio.wait_for(first_reading.wait(), 30)
            # Only the first edit is under way, so the one lock is the file's.
            ((key, held_lock),) = _edit_locks.items()
            for n in range(_LOCKS_THE_MAP_KEPT + 1):
                other = await _handle_edit_file(
                    {
                        "file_path": f"other-{n}.txt",
                        "old_string": "old",
                        "new_string": "x",
                    }
                )
                assert other["isError"] is False
            second = asyncio.create_task(
                _handle_edit_file(
                    {"file_path": "notes.txt", "old_string": "two", "new_string": "2"}
                )
            )
            # The second edit either queues on the lock the first holds, or
            # reads the file beside it. Whichever it does shows, so nothing
            # here depends on how long it takes to get there.
            await _until(lambda: held_lock.users == 2 or second_read.is_set())
            assert not second_read.is_set()
            assert _edit_locks.get(key) is held_lock
            first_may_finish.set()
            results = await asyncio.wait_for(asyncio.gather(first, second), 30)
        finally:
            await _end(first, second)

        assert [result["isError"] for result in results] == [False, False]
        assert stored[held] == b"1 2"

    @pytest.mark.asyncio
    async def test_a_lock_an_edit_waits_for_outlives_its_holder(self):
        """Free for a moment between its holder and the edit waiting for it, a
        lock is still in use: a third edit must queue on it, not on a new one."""
        key = "local_pc:machine-1:/Users/test/ws/queued.txt"
        inside: list[str] = []
        most_inside = 0

        async def edit(name: str, leave: asyncio.Event) -> None:
            nonlocal most_inside
            async with _edit_lock(key):
                inside.append(name)
                most_inside = max(most_inside, len(inside))
                await leave.wait()
                inside.remove(name)

        first_leaves, others_leave = asyncio.Event(), asyncio.Event()
        first = asyncio.create_task(edit("first", first_leaves))
        waiting: asyncio.Task[None] | None = None
        third: asyncio.Task[None] | None = None
        try:
            await _until(lambda: inside == ["first"])
            file_lock = _edit_locks[key]
            waiting = asyncio.create_task(edit("waiting", others_leave))
            await _until(lambda: file_lock.users == 2)
            first_leaves.set()
            await asyncio.wait_for(first, 30)
            for n in range(_LOCKS_THE_MAP_KEPT + 1):
                async with _edit_lock(
                    f"local_pc:machine-1:/Users/test/ws/other-{n}.txt"
                ):
                    pass
            third = asyncio.create_task(edit("third", others_leave))
            # The third edit either queues on the lock the waiting one claimed
            # (one of the two is in the file, and both hold a claim on that
            # lock), or it took a second lock and the two are in the file
            # together. Whichever it does shows, with no wait to outrun.
            await _until(
                lambda: len(inside) == 2 or (len(inside) == 1 and file_lock.users == 2)
            )
            assert _edit_locks.get(key) is file_lock
            others_leave.set()
            await asyncio.wait_for(asyncio.gather(waiting, third), 30)
        finally:
            await _end(first, waiting, third)

        assert most_inside == 1

    @pytest.mark.asyncio
    async def test_only_files_being_edited_keep_a_lock(self):
        """The map can't grow without bound: a file's lock goes with its last
        edit."""
        held = "local_pc:machine-1:/Users/test/ws/held.txt"

        async with _edit_lock(held):
            for n in range(50):
                async with _edit_lock(f"local_pc:machine-1:/Users/test/ws/{n}.txt"):
                    pass
            assert list(_edit_locks) == [held]

        assert not _edit_locks
