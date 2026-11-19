"""The wire-op latency and error series are recorded once per attempt."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest

from backend.copilot.tools import local_pc_shim
from backend.copilot.tools.local_pc_shim import LocalPCShim, _RpcAttemptFailed


class _AnsweringTransport:
    """Answers each request through the shim's own receive path."""

    def __init__(
        self, shim: LocalPCShim, answer: Callable[[dict], dict[str, Any] | None]
    ) -> None:
        self._shim = shim
        self._answer = answer

    async def send_text(self, data: str) -> None:
        reply = self._answer(json.loads(data))
        if reply is not None:
            asyncio.get_running_loop().call_soon(
                self._shim._process_raw_message, json.dumps(reply)
            )

    async def iter_text(self):
        if False:
            yield ""

    async def close(self) -> None:
        return None


def _shim(answer: Callable[[dict], dict[str, Any] | None]) -> LocalPCShim:
    shim = LocalPCShim.__new__(LocalPCShim)
    shim._pending = {}
    shim._streaming = {}
    shim._recording_steps = {}
    shim._pending_capacity = None
    shim._ws = _AnsweringTransport(shim, answer)
    return shim


@pytest.fixture
def metrics(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, MagicMock]:
    record_rpc, record_rpc_error = MagicMock(), MagicMock()
    monkeypatch.setattr(local_pc_shim, "record_rpc", record_rpc)
    monkeypatch.setattr(local_pc_shim, "record_rpc_error", record_rpc_error)
    return record_rpc, record_rpc_error


@pytest.mark.asyncio
async def test_an_answered_op_records_ok(metrics) -> None:
    record_rpc, record_rpc_error = metrics
    shim = _shim(
        lambda request: {
            "type": "FILE_CONTENTS",
            "id": request["id"],
            "payload": {"content": "hi"},
        }
    )

    await shim._send_and_wait("FILE_READ", {"path": "a.txt"}, timeout=5)

    [call] = record_rpc.call_args_list
    assert call.args == ("FILE_READ",)
    assert call.kwargs["outcome"] == "ok"
    assert call.kwargs["duration_seconds"] >= 0
    record_rpc_error.assert_not_called()


@pytest.mark.asyncio
async def test_a_wire_error_records_its_code(metrics) -> None:
    record_rpc, record_rpc_error = metrics
    shim = _shim(
        lambda request: {
            "type": "ERROR",
            "id": request["id"],
            "payload": {"code": "PATH_NOT_FOUND", "message": "missing"},
        }
    )

    await shim._send_and_wait("FILE_READ", {"path": "a.txt"}, timeout=5)

    assert record_rpc.call_args.kwargs["outcome"] == "error"
    record_rpc_error.assert_called_once_with("FILE_READ", "PATH_NOT_FOUND")


@pytest.mark.asyncio
async def test_an_unanswered_op_records_a_timeout(metrics) -> None:
    record_rpc, record_rpc_error = metrics
    shim = _shim(lambda _request: None)

    with pytest.raises(_RpcAttemptFailed):
        await shim._send_and_wait("FILE_READ", {"path": "a.txt"}, timeout=0.01)

    assert record_rpc.call_args.kwargs["outcome"] == "timeout"
    record_rpc_error.assert_not_called()


@pytest.mark.asyncio
async def test_a_cancelled_op_records_nothing(metrics) -> None:
    """A cancellation says nothing about the machine, so it is no sample."""
    record_rpc, _ = metrics
    sent = asyncio.Event()

    def answer(_request: dict) -> None:
        sent.set()

    shim = _shim(answer)
    request = asyncio.create_task(
        shim._send_and_wait("FILE_READ", {"path": "a.txt"}, timeout=30)
    )
    await sent.wait()
    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request

    record_rpc.assert_not_called()
