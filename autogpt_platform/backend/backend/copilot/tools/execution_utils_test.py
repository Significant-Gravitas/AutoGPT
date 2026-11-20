"""Tests for execution waiting and run-health helpers in execution_utils."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.data.execution import (
    ExecutionStatus,
    GraphExecutionEvent,
    NodeExecutionResult,
)

from . import execution_utils
from .execution_utils import (
    _subscribe_and_wait,
    build_run_health_warning,
    summarize_node_failures,
    wait_for_execution,
)

STORE_VALUE_BLOCK_ID = "1ff065e9-88e8-4358-9d82-8dc91f622ba9"


def make_node_execution(
    status: ExecutionStatus,
    output_data: dict | None = None,
    block_id: str = STORE_VALUE_BLOCK_ID,
) -> NodeExecutionResult:
    now = datetime.now(timezone.utc)
    return NodeExecutionResult(
        user_id="user-1",
        graph_id="graph-1",
        graph_version=1,
        graph_exec_id="exec-1",
        node_exec_id="node-exec-1",
        node_id="node-1",
        block_id=block_id,
        status=status,
        input_data={},
        output_data=output_data or {},
        add_time=now,
        queue_time=None,
        start_time=now,
        end_time=now,
    )


def test_summarize_node_failures_extracts_failed_nodes():
    failures = summarize_node_failures(
        [
            make_node_execution(ExecutionStatus.COMPLETED),
            make_node_execution(
                ExecutionStatus.FAILED,
                output_data={"error": ["NameError: name 'variables' is not defined"]},
            ),
        ]
    )

    assert len(failures) == 1
    assert failures[0]["node_id"] == "node-1"
    assert failures[0]["block_name"] == "StoreValueBlock"
    assert failures[0]["error"] == "NameError: name 'variables' is not defined"


def test_summarize_node_failures_handles_missing_error_and_unknown_block():
    failures = summarize_node_failures(
        [make_node_execution(ExecutionStatus.FAILED, block_id="not-a-block-id")]
    )

    assert failures == [
        {"node_id": "node-1", "block_name": "not-a-block-id", "error": None}
    ]


def test_summarize_node_failures_truncates_long_errors():
    failures = summarize_node_failures(
        [
            make_node_execution(
                ExecutionStatus.FAILED, output_data={"error": ["x" * 900]}
            )
        ]
    )

    assert failures[0]["error"] is not None
    assert len(failures[0]["error"]) == 500


def test_build_run_health_warning_reports_failed_nodes():
    warning = build_run_health_warning(
        outputs={"result": ["ok"]},
        node_failures=[
            {"node_id": "n1", "block_name": "CodeExecutionBlock", "error": "boom"}
        ],
    )

    assert warning is not None
    assert "1 node(s) FAILED" in warning
    assert "CodeExecutionBlock: boom" in warning


def test_build_run_health_warning_flags_empty_outputs():
    warning = build_run_health_warning(outputs={}, node_failures=[])

    assert warning is not None
    assert "no outputs" in warning


def test_build_run_health_warning_none_when_healthy():
    assert build_run_health_warning({"result": ["ok"]}, []) is None


def test_incomplete_nodes_are_not_flagged_as_failures():
    """INCOMPLETE nodes are legal in a COMPLETED run — unexecuted branches
    (e.g. a ConditionBlock's untaken path) finish INCOMPLETE by design, so
    flagging them would false-positive every conditional agent. Only FAILED
    nodes indicate a broken run."""
    from unittest.mock import MagicMock

    from backend.data.execution import ExecutionStatus

    ne = MagicMock()
    ne.status = ExecutionStatus.INCOMPLETE
    ne.node_id = "n1"
    ne.block_id = "block-x"
    ne.output_data = {}

    assert summarize_node_failures([ne]) == []


def test_many_failures_capped_detail_but_full_count():
    """The warning lists at most 5 failure details but counts all of them."""
    from unittest.mock import MagicMock

    from backend.data.execution import ExecutionStatus

    nodes = []
    for i in range(8):
        ne = MagicMock()
        ne.status = ExecutionStatus.FAILED
        ne.node_id = f"n{i}"
        ne.block_id = f"block-{i}"
        ne.output_data = {"error": [f"boom {i}"]}
        nodes.append(ne)

    failures = summarize_node_failures(nodes)
    assert len(failures) == 8

    warning = build_run_health_warning({"out": ["x"]}, failures)
    assert warning is not None
    assert "8 node(s) FAILED" in warning
    assert warning.count("boom") == 5


class _Stream:
    """The execution event bus: ``listen_with_resync`` yields ``items``, then
    stays quiet. ``None`` marks a confirmed (re)subscription."""

    def __init__(self, *items: object) -> None:
        self._items = items

    async def listen_with_resync(self, channel_key: str):
        for item in self._items:
            yield item
        await asyncio.Event().wait()


def _with_status(status: ExecutionStatus) -> MagicMock:
    execution = MagicMock()
    execution.status = status
    return execution


def _wire(monkeypatch: pytest.MonkeyPatch, stream: _Stream, *reads: object):
    """Each DB read returns the next of ``reads`` (raising an exception one)."""
    db = MagicMock()
    db.get_graph_execution = AsyncMock(side_effect=list(reads))
    monkeypatch.setattr(execution_utils, "execution_db", lambda: db)
    monkeypatch.setattr(execution_utils, "AsyncRedisExecutionEventBus", lambda: stream)
    return db


async def test_wait_returns_when_the_end_arrives_live():
    """Called directly: through ``wait_for_execution`` an error here would be
    hidden by its fallback read."""
    running = _with_status(ExecutionStatus.RUNNING)
    done = _with_status(ExecutionStatus.COMPLETED)
    finished = MagicMock(spec=GraphExecutionEvent)
    finished.status = ExecutionStatus.COMPLETED
    db = MagicMock()
    # The recheck once subscribed, then the read after the live end.
    db.get_graph_execution = AsyncMock(side_effect=[running, done])
    stream = _Stream(None, finished)

    async with asyncio.timeout(5):
        result = await _subscribe_and_wait(stream, "chan", "u", "e", db)

    assert result is done


async def test_wait_sees_an_end_reached_while_the_stream_was_down(
    monkeypatch: pytest.MonkeyPatch,
):
    running = _with_status(ExecutionStatus.RUNNING)
    done = _with_status(ExecutionStatus.COMPLETED)
    # First check, the recheck once subscribed, the recheck after a resubscribe.
    _wire(monkeypatch, _Stream(None, None), running, running, done)

    async with asyncio.timeout(5):
        result = await wait_for_execution("u", "g", "e", timeout_seconds=300)

    assert result is done


async def test_a_failed_recheck_returns_the_current_state_without_waiting_out_the_timeout(
    monkeypatch: pytest.MonkeyPatch,
):
    running = _with_status(ExecutionStatus.RUNNING)
    db = _wire(
        monkeypatch,
        _Stream(None),
        running,
        ConnectionError("database manager unreachable"),
        running,
    )

    async with asyncio.timeout(5):
        result = await wait_for_execution("u", "g", "e", timeout_seconds=300)

    assert result is running
    assert db.get_graph_execution.await_count == 3
