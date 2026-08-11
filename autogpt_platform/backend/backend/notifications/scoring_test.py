"""Interestingness ranking: blocked and failed outrank success, and a
first-ever success outranks everything."""

from prisma.enums import AgentExecutionStatus

from backend.notifications.scoring import ANOMALY_FACTOR, compute_score


def _score(status, **over):
    args = dict(
        cost_cents=10.0,
        node_error_count=0,
        has_activity=True,
        first_success=False,
        cost_baseline=None,
    )
    args.update(over)
    return compute_score(status=status, **args)


def test_failed_outranks_success():
    assert _score(AgentExecutionStatus.FAILED) > _score(
        AgentExecutionStatus.COMPLETED
    )


def test_first_ever_success_outranks_a_routine_one():
    assert _score(AgentExecutionStatus.COMPLETED, first_success=True) > _score(
        AgentExecutionStatus.COMPLETED
    )


def test_a_no_op_ranks_below_a_run_that_produced_something():
    assert _score(AgentExecutionStatus.COMPLETED, has_activity=False) < _score(
        AgentExecutionStatus.COMPLETED, has_activity=True
    )


def test_cost_anomaly_is_measured_against_the_agents_own_baseline():
    baseline = 10.0
    normal = _score(AgentExecutionStatus.COMPLETED, cost_cents=baseline * 2,
                    cost_baseline=baseline)
    spike = _score(
        AgentExecutionStatus.COMPLETED,
        cost_cents=baseline * ANOMALY_FACTOR + 1,
        cost_baseline=baseline,
    )
    assert spike > normal


def test_finished_but_not_cleanly_outranks_a_silent_success():
    assert _score(AgentExecutionStatus.COMPLETED, node_error_count=2) > _score(
        AgentExecutionStatus.COMPLETED
    )


def test_score_never_goes_negative():
    assert (
        _score(AgentExecutionStatus.COMPLETED, has_activity=False, cost_cents=0)
        >= 0
    )
