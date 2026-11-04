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
    assert target.read_text() == "written"
    assert [op for op, result in audit.ops if result["ok"]] == ["FILE_WRITE"]
