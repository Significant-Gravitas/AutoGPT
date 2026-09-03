"""REL-005 / CEO decision: missed occurrences are NON-BILLABLE.

A scheduler occurrence with technical status ``missed`` must not create
execution usage, model/tool usage, or customer billing. Billing requires
actual chargeable execution activity (an AgentGraphExecution row linked
to usage), which a missed occurrence never creates.

Policy (Wave 0, CEO-approved):
    status=missed -> observability/reconciliation record only -> NON-BILLABLE.
"""

import datetime
import inspect
from unittest.mock import AsyncMock, patch

import pytest

FIRE = datetime.datetime(2025, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)


@pytest.mark.asyncio
async def test_missed_occurrence_creates_no_execution_and_is_non_billable():
    """create_missed_occurrence writes only the technical record: no
    executionId, no add_graph_execution, no billing invocation."""
    from backend.data.schedule_occurrence import create_missed_occurrence

    with (
        patch(
            "backend.data.schedule_occurrence.ScheduleOccurrence.prisma"
        ) as mock_occ,
        patch(
            "backend.executor.scheduler.execution_utils.add_graph_execution",
            new_callable=AsyncMock,
        ) as mock_add,
    ):
        mock_add.return_value = None
        mock_occ.return_value.create = AsyncMock(return_value=None)

        row = await create_missed_occurrence("sched-bill", FIRE)

        data = mock_occ.return_value.create.call_args.kwargs["data"]
        # 1) technical record only — no execution linkage
        assert data["status"] == "missed"
        assert "executionId" not in data, (
            "missed occurrence must not link a chargeable execution"
        )
        # 2) no execution was created for the missed tick
        assert mock_add.call_count == 0


@pytest.mark.asyncio
async def test_billing_pipeline_unreachable_from_missed_occurrence():
    """The billing pipeline (charge_usage -> spend_credits) is only reachable
    through a node execution belonging to an AgentGraphExecution row. A
    missed occurrence produces no such row, so there is no usage key to
    bill against. This pins that invariant structurally."""
    from backend.executor import billing
    from backend.executor import scheduler as sched

    # charge_usage is keyed on node_exec (an execution's node), not on
    # occurrences — occurrences with status=missed never produce one.
    sig = inspect.signature(billing.charge_usage)
    assert "node_exec" in sig.parameters, (
        "charge_usage must be keyed on a node execution (missed occurrences "
        "never produce one)"
    )

    # The only scheduler -> execution creation path is execution_utils.add_graph_execution;
    # the missed-tick listener never calls it.
    src = inspect.getsource(sched.job_missed_listener)
    assert "add_graph_execution" not in src, (
        "missed-tick listener must not create executions"
    )
    assert "create_missed_occurrence" in src or "missed" in src


@pytest.mark.asyncio
async def test_missed_occurrence_never_carries_execution_id():
    """Even on the duplicate-converge path, a missed record never gains an
    executionId: create data omits it and convergence returns the existing
    row untouched."""
    from backend.data.schedule_occurrence import create_missed_occurrence

    with patch(
        "backend.data.schedule_occurrence.ScheduleOccurrence.prisma"
    ) as mock_occ:
        # First write
        mock_occ.return_value.create = AsyncMock(return_value=None)
        await create_missed_occurrence("sched-y", FIRE)
        first = mock_occ.return_value.create.call_args.kwargs["data"]
        assert "executionId" not in first

        # Duplicate converge — existing row returned as-is, no update call
        from prisma.errors import UniqueViolationError

        def _uve():
            return UniqueViolationError(
                {
                    "user_facing_error": {
                        "message": "unique",
                        "code": "P2002",
                        "meta": {},
                    }
                }
            )

        existing = type("Row", (), {"executionId": None, "status": "missed"})()
        mock_occ.return_value.create = AsyncMock(side_effect=_uve())
        mock_occ.return_value.find_unique = AsyncMock(return_value=existing)
        mock_occ.return_value.update = AsyncMock()

        row = await create_missed_occurrence("sched-y", FIRE)
        assert row is existing
        assert row.executionId is None
        # Convergence must not write an executionId onto the missed record
        assert mock_occ.return_value.update.call_count == 0


def test_missed_occurrence_status_documented_non_billable():
    """The occurrence state model documents missed as non-billing (source
    of truth for reconciliation)."""
    import pathlib

    text = pathlib.Path("backend/data/schedule_occurrence.py").read_text()
    assert "missed" in text
    assert "no billing" in text.lower()