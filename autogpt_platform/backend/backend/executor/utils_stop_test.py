"""The cancels a stop sends (#17178).

A stop that reads a run as QUEUED, INCOMPLETE or REVIEW ends it itself, without
waiting for an executor. An executor can have started the run between that read
and the write, and it ignores a cancel for a run it has not registered yet, so
the stop sends the cancel once more after the run is written TERMINATED.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest
from pytest_mock import MockerFixture

from backend.data.execution import ExecutionStatus, GraphExecutionMeta
from backend.executor.utils import (
    GRAPH_EXECUTION_CANCEL_EXCHANGE,
    CancelExecutionEvent,
    stop_graph_execution,
)

USER_ID = "stop-user"
RUN_ID = "stop-run"
CANCEL = {
    "routing_key": "",
    "message": CancelExecutionEvent(graph_exec_id=RUN_ID).model_dump_json(),
    "exchange": GRAPH_EXECUTION_CANCEL_EXCHANGE,
}


def _a_stop_that_reads_the_run_as(
    mocker: MockerFixture,
    *reads: ExecutionStatus,
    event_error: Exception | None = None,
    failing_cancel: int | None = None,
    cancel_error: BaseException | None = None,
) -> tuple[list[str], AsyncMock]:
    """Set up a stop of RUN_ID that reads it back as each of ``reads`` in turn.

    Returns what the stop will do, in order ("cancel" for a cancel it
    publishes, "terminated" for its write of the run, "published" for the
    event it publishes, which fails with ``event_error`` if given), and the
    queue its cancels go to.
    """
    events: list[str] = []
    queue = mocker.AsyncMock()

    def publish_cancel(**_: object) -> None:
        events.append("cancel")
        if failing_cancel is not None and events.count("cancel") == failing_cancel:
            raise cancel_error or ConnectionError("the broker refused the cancel")

    queue.publish_message.side_effect = publish_cancel
    mocker.patch("backend.executor.utils.get_async_execution_queue", return_value=queue)
    mocker.patch("backend.executor.utils.prisma").is_connected.return_value = True

    def publish_event(_run: GraphExecutionMeta) -> None:
        events.append("published")
        if event_error:
            raise event_error

    event_bus = mocker.patch(
        "backend.executor.utils.get_async_execution_event_bus"
    ).return_value
    event_bus.publish = mocker.AsyncMock(side_effect=publish_event)
    mocker.patch(
        "backend.executor.utils.human_review_db"
    ).cancel_pending_reviews_for_execution = mocker.AsyncMock(return_value=0)

    runs = []
    for status in reads:
        run = mocker.MagicMock(spec=GraphExecutionMeta)
        run.id = RUN_ID
        run.status = status
        runs.append(run)
    execution_db = mocker.patch("backend.executor.utils.execution_db")
    execution_db.get_graph_execution_meta = mocker.AsyncMock(side_effect=runs)
    execution_db.update_graph_execution_stats = mocker.AsyncMock(
        side_effect=lambda **_: events.append("terminated")
    )
    return events, queue.publish_message


@pytest.mark.parametrize(
    "error",
    [ConnectionError("the broker refused the cancel"), RuntimeError("closed channel")],
    ids=["connection", "other"],
)
async def test_a_second_cancel_the_broker_refuses_still_lets_the_event_out(
    mocker: MockerFixture, caplog: pytest.LogCaptureFixture, error: Exception
):
    """The run is TERMINATED and the first cancel went out, so a failed second
    cancel, whatever it failed with, neither fails the stop nor keeps its event
    from being published; it is logged."""
    events, _ = _a_stop_that_reads_the_run_as(
        mocker, ExecutionStatus.QUEUED, failing_cancel=2, cancel_error=error
    )

    with caplog.at_level("WARNING"):
        await stop_graph_execution(user_id=USER_ID, graph_exec_id=RUN_ID, cascade=False)

    assert events == ["cancel", "terminated", "cancel", "published"]
    assert f"Could not send the second cancel for run #{RUN_ID}" in caplog.text


async def test_a_second_cancel_cancelled_with_the_stop_is_not_swallowed(
    mocker: MockerFixture,
):
    """Best effort covers a failed publish, not the stop itself being
    cancelled: that still propagates."""
    _a_stop_that_reads_the_run_as(
        mocker,
        ExecutionStatus.QUEUED,
        failing_cancel=2,
        cancel_error=asyncio.CancelledError(),
    )

    with pytest.raises(asyncio.CancelledError):
        await stop_graph_execution(user_id=USER_ID, graph_exec_id=RUN_ID, cascade=False)


@pytest.mark.parametrize(
    "status",
    [ExecutionStatus.QUEUED, ExecutionStatus.INCOMPLETE, ExecutionStatus.REVIEW],
)
async def test_a_run_the_stop_ends_itself_is_cancelled_again_once_written(
    mocker: MockerFixture, status: ExecutionStatus
):
    events, publish = _a_stop_that_reads_the_run_as(mocker, status)

    await stop_graph_execution(user_id=USER_ID, graph_exec_id=RUN_ID, cascade=False)

    assert events == ["cancel", "terminated", "cancel", "published"]
    assert [call.kwargs for call in publish.await_args_list] == [CANCEL, CANCEL]


async def test_the_second_cancel_goes_out_when_the_event_cannot_be_published(
    mocker: MockerFixture,
):
    events, publish = _a_stop_that_reads_the_run_as(
        mocker, ExecutionStatus.QUEUED, event_error=RuntimeError("event bus down")
    )

    with pytest.raises(RuntimeError, match="event bus down"):
        await stop_graph_execution(user_id=USER_ID, graph_exec_id=RUN_ID, cascade=False)

    assert events == ["cancel", "terminated", "cancel", "published"]
    assert [call.kwargs for call in publish.await_args_list] == [CANCEL, CANCEL]


@pytest.mark.parametrize(
    "reads",
    [
        pytest.param(
            [ExecutionStatus.RUNNING, ExecutionStatus.TERMINATED],
            id="running-then-stopped",
        ),
        pytest.param(
            [
                ExecutionStatus.RUNNING,
                ExecutionStatus.RUNNING,
                ExecutionStatus.COMPLETED,
            ],
            id="running-then-completed",
        ),
        pytest.param([ExecutionStatus.TERMINATED], id="already-stopped"),
        pytest.param([ExecutionStatus.FAILED], id="already-failed"),
    ],
)
async def test_a_run_an_executor_has_gets_one_cancel_and_no_write_from_the_stop(
    mocker: MockerFixture, reads: list[ExecutionStatus]
):
    events, publish = _a_stop_that_reads_the_run_as(mocker, *reads)

    await stop_graph_execution(user_id=USER_ID, graph_exec_id=RUN_ID, cascade=False)

    assert events == ["cancel", "published"]
    assert [call.kwargs for call in publish.await_args_list] == [CANCEL]
