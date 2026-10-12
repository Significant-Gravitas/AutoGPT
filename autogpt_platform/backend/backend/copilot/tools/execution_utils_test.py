"""Tests for run-health helpers in execution_utils."""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.data.event_bus import EventStreamLostError
from backend.data.execution import (
    AsyncRedisExecutionEventBus,
    ExecutionStatus,
    NodeExecutionResult,
)

from .execution_utils import (
    _subscribe_and_wait,
    build_run_health_warning,
    summarize_node_failures,
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


class _LostThenSilentBus:
    """listen_events loses its stream once, then waits quietly."""

    def __init__(self) -> None:
        self.listens = 0

    async def listen_events(self, channel_key: str, on_subscribed=None):
        self.listens += 1
        if self.listens == 1:
            raise EventStreamLostError("connection dropped")
        if on_subscribed is not None:
            on_subscribed()
        await asyncio.sleep(3600)
        yield  # pragma: no cover — unreachable


@pytest.mark.asyncio
async def test_wait_checks_the_database_when_the_event_stream_is_lost():
    """The COMPLETED event can be lost with the stream: the wait must find the
    finished run in the database, not report it as missing."""
    bus = _LostThenSilentBus()
    finished = SimpleNamespace(status=ExecutionStatus.COMPLETED)
    exec_db = MagicMock()
    # The first stream never confirmed, so the first read is after the
    # resubscription is live.
    exec_db.get_graph_execution = AsyncMock(return_value=finished)
    tasks: list[asyncio.Task] = []

    with patch("backend.copilot.tools.execution_utils._RESUBSCRIBE_DELAY", 0):
        result = await asyncio.wait_for(
            _subscribe_and_wait(
                cast(AsyncRedisExecutionEventBus, bus),
                "chan",
                "user-1",
                "exec-1",
                exec_db,
                tasks,
            ),
            timeout=5,
        )

    assert result is finished
    for task in tasks:
        task.cancel()


class _SlowToConfirmBus:
    """The first subscription confirms only when the test opens ``gate``."""

    def __init__(self) -> None:
        self.gate = asyncio.Event()
        self.confirmed = False

    async def listen_events(self, channel_key: str, on_subscribed=None):
        await self.gate.wait()
        self.confirmed = True
        if on_subscribed is not None:
            on_subscribed()
        await asyncio.sleep(3600)
        yield  # pragma: no cover — unreachable


@pytest.mark.asyncio
async def test_wait_reads_the_database_only_once_the_first_subscription_is_live():
    """The run finishes, and its COMPLETED is published, before the first
    subscription is live, so it is never delivered. Read before confirmation
    and the wait sees RUNNING and then blocks until its timeout; read after it
    and the wait sees COMPLETED."""
    bus = _SlowToConfirmBus()
    finished = SimpleNamespace(status=ExecutionStatus.COMPLETED)
    running = SimpleNamespace(status=ExecutionStatus.RUNNING)
    exec_db = MagicMock()
    exec_db.get_graph_execution = AsyncMock(
        side_effect=lambda **_: finished if bus.confirmed else running
    )
    tasks: list[asyncio.Task] = []

    wait = asyncio.create_task(
        _subscribe_and_wait(
            cast(AsyncRedisExecutionEventBus, bus),
            "chan",
            "user-1",
            "exec-1",
            exec_db,
            tasks,
        )
    )
    await asyncio.sleep(0.3)
    exec_db.get_graph_execution.assert_not_awaited()

    bus.gate.set()
    assert await asyncio.wait_for(wait, timeout=2) is finished
    for task in tasks:
        task.cancel()
