from types import SimpleNamespace

import pytest

from backend.data.execution import ExecutionStatus, GraphExecutionMeta
from backend.util.exceptions import ExecutionFailureReason


def _db_execution(*, status: ExecutionStatus, stats: dict):
    return SimpleNamespace(
        id="exec-1",
        userId="user-1",
        agentGraphId="graph-1",
        agentGraphVersion=1,
        inputs=None,
        credentialInputs=None,
        nodesInputMasks=None,
        agentPresetId=None,
        executionStatus=status,
        startedAt=None,
        endedAt=None,
        stats=stats,
        isShared=False,
        shareToken=None,
        organizationId=None,
        teamId=None,
        expertId=None,
    )


def test_failure_reason_round_trips_through_api_stats():
    stats = GraphExecutionMeta.Stats(
        error="Organization has 0 credits but needs 10",
        failure_reason=ExecutionFailureReason.INSUFFICIENT_BALANCE,
        activity_status="The available credit balance was insufficient.",
        correctness_score=0.0,
    )

    persisted = stats.to_db()

    assert persisted.failure_reason == ExecutionFailureReason.INSUFFICIENT_BALANCE
    assert persisted.model_dump()["failure_reason"] == (
        ExecutionFailureReason.INSUFFICIENT_BALANCE
    )
    assert persisted.correctness_score == 0.0


@pytest.mark.parametrize(
    ("status", "expected_reason"),
    [
        (ExecutionStatus.FAILED, ExecutionFailureReason.INSUFFICIENT_BALANCE),
        (ExecutionStatus.COMPLETED, None),
    ],
)
def test_legacy_failure_reason_is_derived_only_for_failed_executions(
    status,
    expected_reason,
):
    execution = GraphExecutionMeta.from_db(
        _db_execution(
            status=status,
            stats={
                "error": "Organization has 0 credits but needs 10",
                "activity_status": None,
                "correctness_score": None,
            },
        )
    )

    assert execution.stats is not None
    assert execution.stats.failure_reason == expected_reason


def test_failed_execution_without_error_has_no_derived_failure_reason():
    execution = GraphExecutionMeta.from_db(
        _db_execution(
            status=ExecutionStatus.FAILED,
            stats={"error": None},
        )
    )

    assert execution.stats is not None
    assert execution.stats.failure_reason is None


def test_persisted_failure_reason_is_preserved():
    execution = GraphExecutionMeta.from_db(
        _db_execution(
            status=ExecutionStatus.FAILED,
            stats={
                "error": "Producer wording can change",
                "failure_reason": "insufficient_balance",
            },
        )
    )

    assert execution.stats is not None
    assert execution.stats.failure_reason == ExecutionFailureReason.INSUFFICIENT_BALANCE
