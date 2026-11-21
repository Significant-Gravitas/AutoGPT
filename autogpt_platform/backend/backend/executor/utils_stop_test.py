"""The cancels a stop sends (#17178).

A stop that reads a run as QUEUED, INCOMPLETE or REVIEW ends it itself, without
waiting for an executor. An executor can have started the run between that read
and the write, and it ignores a cancel for a run it has not registered yet, so
the stop sends the cancel once more after the run is written TERMINATED.
"""

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


async def _stop_run_read_as(
    mocker: MockerFixture, *reads: ExecutionStatus
) -> tuple[list[str], AsyncMock]:
    """Stop RUN_ID, which the stop reads back as each of ``reads`` in turn.

    Returns what the stop did in order ("cancel" for a cancel it published,
    "terminated" for its write of the run) and the queue its cancels went to.
    """
    events: list[str] = []
    queue = mocker.AsyncMock()
    queue.publish_message.side_effect = lambda **_: events.append("cancel")
    mocker.patch("backend.executor.utils.get_async_execution_queue", return_value=queue)
    mocker.patch("backend.executor.utils.prisma").is_connected.return_value = True
    event_bus = mocker.patch(
        "backend.executor.utils.get_async_execution_event_bus"
    ).return_value
    event_bus.publish = mocker.AsyncMock()
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

    await stop_graph_execution(user_id=USER_ID, graph_exec_id=RUN_ID, cascade=False)

    return events, queue.publish_message


@pytest.mark.parametrize(
    "status",
    [ExecutionStatus.QUEUED, ExecutionStatus.INCOMPLETE, ExecutionStatus.REVIEW],
)
async def test_a_run_the_stop_ends_itself_is_cancelled_again_once_written(
    mocker: MockerFixture, status: ExecutionStatus
):
    events, publish = await _stop_run_read_as(mocker, status)

    assert events == ["cancel", "terminated", "cancel"]
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
    events, publish = await _stop_run_read_as(mocker, *reads)

    assert events == ["cancel"]
    assert [call.kwargs for call in publish.await_args_list] == [CANCEL]
