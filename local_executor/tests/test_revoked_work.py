"""Work that is already running when its task is cancelled (SESSION_REVOKED,
session teardown) still finishes, so it must still be audited."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

import pytest

import autogpt_local_executor.handlers as handlers_module
from autogpt_local_executor.config import ShimConfig
from autogpt_local_executor.handlers import FileHandler
from autogpt_local_executor.protocol import (
    AckMessage,
    FileWriteMessage,
    FileWritePayload,
    new_id,
    now_ts,
)


class _RecordingAudit:
    def __init__(self) -> None:
        self.ops: list[tuple[str, dict[str, Any]]] = []

    async def write(self, op: str, *, request_id: Any = None, details: Any, result: Any) -> None:
        self.ops.append((op, result))


@pytest.mark.asyncio
async def test_a_write_cancelled_mid_flight_finishes_and_is_audited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancelling the task that awaits the worker thread doesn't stop the
    thread: the write landed, and its audit record was never written."""
    audit = _RecordingAudit()
    handler = FileHandler(
        ShimConfig(allowed_root=tmp_path, audit_log_path=tmp_path / "audit.log"),
        audit=audit,  # type: ignore[arg-type]
    )
    started = threading.Event()
    release = threading.Event()
    real_write = handlers_module.write_jail_file

    def slow_write(path, root, data) -> None:
        started.set()
        release.wait(30)
        real_write(path, root, data)

    monkeypatch.setattr(handlers_module, "write_jail_file", slow_write)
    target = tmp_path / "notes.txt"
    msg = FileWriteMessage(
        id=new_id(), ts=now_ts(), payload=FileWritePayload(path=str(target), content="written")
    )
    task = asyncio.create_task(handler.handle_write(msg))
    assert await asyncio.to_thread(started.wait, 30)

    task.cancel()
    await asyncio.sleep(0)
    release.set()
    response = await asyncio.wait_for(task, timeout=30)

    assert isinstance(response, AckMessage)
    assert target.read_text(encoding="utf-8") == "written"
    assert [op for op, result in audit.ops if result["ok"]] == ["FILE_WRITE"]


@pytest.mark.asyncio
async def test_a_change_that_finishes_as_the_cancel_lands_is_still_returned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the thread finished just as the cancel landed, the task re-raised
    the cancel, so its handler never audited a change that had happened."""
    loop = asyncio.get_running_loop()
    finished: asyncio.Future[str] = loop.create_future()
    thread: dict[str, asyncio.Task[Any] | None] = {}

    async def fake_thread(func, /, *args, **kwargs):
        thread["task"] = asyncio.current_task()
        return await finished

    monkeypatch.setattr(handlers_module.asyncio, "to_thread", fake_thread)
    task = asyncio.create_task(handlers_module._run_to_completion(lambda: None))
    for _ in range(2):
        await asyncio.sleep(0)
    finished.set_result("written")
    await asyncio.sleep(0)
    # The work is done, but the task hasn't resumed yet.
    assert thread["task"] is not None and thread["task"].done() and not task.done()

    task.cancel()

    assert await task == "written"


@pytest.mark.asyncio
async def test_a_change_still_waiting_for_a_thread_when_cancelled_never_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shield kept the cancel from reaching a job still queued for a
    thread, so it ran after SESSION_REVOKED: a write on a revoked session."""
    loop = asyncio.get_running_loop()
    ran: list[str] = []
    a_thread_frees_up: asyncio.Future[None] = loop.create_future()
    queued: list[Any] = []

    async def busy_pool(func, /, *args, **kwargs):
        queued.append(func)
        await a_thread_frees_up
        return func(*args, **kwargs)

    monkeypatch.setattr(handlers_module.asyncio, "to_thread", busy_pool)
    task = asyncio.create_task(handlers_module._run_to_completion(lambda: ran.append("write")))
    for _ in range(2):
        await asyncio.sleep(0)
    assert queued

    task.cancel()
    await asyncio.sleep(0)
    a_thread_frees_up.set_result(None)

    with pytest.raises(asyncio.CancelledError):
        await task
    assert ran == []


@pytest.mark.asyncio
async def test_a_cancel_during_the_audit_write_still_records_the_change(
    tmp_path: Path,
) -> None:
    """After the change, the handler's audit write waited on the writer's
    lock, and a cancel landing there stopped it: a change that had happened
    had no record."""
    entered = asyncio.Event()
    release = asyncio.Event()

    class SlowAudit(_RecordingAudit):
        async def write(
            self, op: str, *, request_id: Any = None, details: Any, result: Any
        ) -> None:
            entered.set()
            await release.wait()  # the writer's lock
            self.ops.append((op, result))

    audit = SlowAudit()
    handler = FileHandler(
        ShimConfig(allowed_root=tmp_path, audit_log_path=tmp_path / "audit.log"),
        audit=audit,  # type: ignore[arg-type]
    )
    target = tmp_path / "notes.txt"
    msg = FileWriteMessage(
        id=new_id(), ts=now_ts(), payload=FileWritePayload(path=str(target), content="written")
    )
    task = asyncio.create_task(handler.handle_write(msg))
    await asyncio.wait_for(entered.wait(), timeout=30)

    task.cancel()
    await asyncio.sleep(0)
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=30)

    assert target.read_text(encoding="utf-8") == "written"
    assert [op for op, result in audit.ops if result["ok"]] == ["FILE_WRITE"]


@pytest.mark.asyncio
async def test_a_change_that_fails_after_a_cancel_lets_the_held_cancel_go() -> None:
    """The cancellations held while the thread ran were released only when
    it returned. One that raised (a full disk, say) left the task counting
    a cancel it had absorbed, which asyncio.timeout and TaskGroup read to
    tell their own cancels from others."""
    started = threading.Event()
    release = threading.Event()

    def write() -> None:
        started.set()
        release.wait(5)
        raise OSError("No space left on device")

    async def handler() -> int:
        try:
            await handlers_module._run_to_completion(write)
        except OSError:
            pass
        task = asyncio.current_task()
        assert task is not None
        return task.cancelling()

    task = asyncio.create_task(handler())
    await asyncio.to_thread(started.wait, 5)
    task.cancel()
    for _ in range(5):
        await asyncio.sleep(0)
    release.set()

    assert await asyncio.wait_for(task, timeout=5) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("op", ["list", "read"])
async def test_a_cancelled_read_keeps_its_task_until_its_thread_ends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, op: str
) -> None:
    """A cancelled FILE_LIST or FILE_READ ended its task at once while its
    thread, blocked on a network filesystem say, went on, so its request let
    go of its machine slot: repeated disconnects piled up reads past the
    limit, holding the worker threads every chat's requests share."""
    from autogpt_local_executor import platform_info
    from autogpt_local_executor.protocol import (
        FileListMessage,
        FileListPayload,
        FileReadMessage,
        FileReadPayload,
    )

    (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
    handler = FileHandler(
        ShimConfig(allowed_root=tmp_path, audit_log_path=tmp_path / "audit.log"),
        audit=_RecordingAudit(),  # type: ignore[arg-type]
    )
    entered = threading.Event()
    release = threading.Event()

    def blocking(real: Any) -> Any:
        def blocked(*args: Any, **kwargs: Any) -> Any:
            entered.set()
            release.wait(30)
            return real(*args, **kwargs)

        return blocked

    if op == "list":
        monkeypatch.setattr(
            platform_info,
            "is_case_insensitive_fs",
            blocking(platform_info.is_case_insensitive_fs),
        )
        work = handler.handle_list(
            FileListMessage(id=new_id(), ts=now_ts(), payload=FileListPayload(path=str(tmp_path)))
        )
    else:
        monkeypatch.setattr(
            handlers_module, "open_jail_file", blocking(handlers_module.open_jail_file)
        )
        work = handler.handle_read(
            FileReadMessage(
                id=new_id(), ts=now_ts(), payload=FileReadPayload(path=str(tmp_path / "notes.txt"))
            )
        )
    task = asyncio.ensure_future(work)
    try:
        assert await asyncio.to_thread(entered.wait, 30)
        task.cancel()
        for _ in range(20):
            await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
    await asyncio.wait({task}, timeout=30)

    assert task.cancelled()
