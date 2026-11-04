import pytest

from backend.data.execution import VALID_STATUS_TRANSITIONS, ExecutionStatus
from backend.executor.manager import SKIPPED_NODE_STATUSES


@pytest.mark.parametrize(
    "start",
    [
        # The executor queues every node execution in one of these statuses.
        ExecutionStatus.QUEUED,
        ExecutionStatus.RUNNING,
        ExecutionStatus.TERMINATED,
        ExecutionStatus.REVIEW,
    ],
)
def test_a_skipped_node_reaches_completed(start: ExecutionStatus):
    """The executor completes a node skipped for missing optional credentials
    by writing SKIPPED_NODE_STATUSES in order. Each write must be one that
    VALID_STATUS_TRANSITIONS accepts, or a no-op because the node already has
    that status. Writing COMPLETED straight from QUEUED was rejected, so the
    node stayed QUEUED after its run had finished."""
    status = start
    for next_status in SKIPPED_NODE_STATUSES:
        assert next_status == status or status in VALID_STATUS_TRANSITIONS[next_status]
        status = next_status

    assert status == ExecutionStatus.COMPLETED
