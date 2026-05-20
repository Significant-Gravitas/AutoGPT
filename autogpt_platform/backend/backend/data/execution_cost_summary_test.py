"""Integration tests for the SQL-aggregated user cost summary."""

import logging
from datetime import date, datetime, timedelta, timezone
from uuid import uuid4

import pytest
from prisma.enums import AgentExecutionStatus
from prisma.errors import UniqueViolationError
from prisma.models import AgentGraph, AgentGraphExecution, User

from backend.data.execution_cost_summary import get_user_cost_summary
from backend.util.json import SafeJson
from backend.util.test import SpinTestServer

logger = logging.getLogger(__name__)


async def _create_user(user_id: str) -> None:
    try:
        await User.prisma().create(
            data={
                "id": user_id,
                "email": f"cost-summary-{user_id}@example.com",
                "name": "Cost Summary Test",
            }
        )
    except UniqueViolationError:
        pass


async def _create_graph(graph_id: str, user_id: str, name: str = "Test") -> None:
    try:
        await AgentGraph.prisma().create(
            data={
                "id": graph_id,
                "version": 1,
                "name": name,
                "description": "test",
                "userId": user_id,
                "isActive": True,
            }
        )
    except UniqueViolationError:
        pass


async def _create_run(
    *,
    run_id: str,
    user_id: str,
    graph_id: str,
    status: AgentExecutionStatus,
    cost_cents: int,
    started_at: datetime,
    duration: float = 1.0,
    node_error_count: int = 0,
    is_dry_run: bool = False,
    created_at: datetime | None = None,
) -> None:
    stats = {
        "cost": cost_cents,
        "duration": duration,
        "node_error_count": node_error_count,
    }
    if is_dry_run:
        stats["is_dry_run"] = True
    await AgentGraphExecution.prisma().create(
        data={
            "id": run_id,
            "agentGraphId": graph_id,
            "agentGraphVersion": 1,
            "executionStatus": status,
            "userId": user_id,
            "createdAt": created_at if created_at is not None else started_at,
            "startedAt": started_at,
            "endedAt": started_at + timedelta(seconds=duration),
            "stats": SafeJson(stats),
        }
    )


async def _cleanup(user_id: str, graph_ids: list[str]) -> None:
    try:
        await AgentGraphExecution.prisma().delete_many(where={"userId": user_id})
        for gid in graph_ids:
            await AgentGraph.prisma().delete_many(where={"id": gid})
        await User.prisma().delete_many(where={"id": user_id})
    except Exception as exc:
        logger.warning("cleanup for %s failed: %s", user_id, exc)


@pytest.mark.asyncio(loop_scope="session")
async def test_empty_window_returns_zeroed_summary(server: SpinTestServer):
    user_id = f"cs-empty-{uuid4()}"
    await _create_user(user_id)
    try:
        now = datetime.now(timezone.utc)
        summary = await get_user_cost_summary(
            user_id=user_id,
            since=now - timedelta(days=1),
            until=now,
        )
        assert summary.total_cents == 0
        assert summary.run_count == 0
        assert summary.billable_run_count == 0
        assert summary.failed_cost_cents == 0
        assert summary.by_agent == []
        assert summary.top_runs == []
        assert summary.daily == []
    finally:
        await _cleanup(user_id, [])


@pytest.mark.asyncio(loop_scope="session")
async def test_aggregates_by_agent_and_top_runs(server: SpinTestServer):
    user_id = f"cs-agg-{uuid4()}"
    graph_a = f"cs-g-a-{uuid4()}"
    graph_b = f"cs-g-b-{uuid4()}"
    await _create_user(user_id)
    await _create_graph(graph_a, user_id, "Alpha")
    await _create_graph(graph_b, user_id, "Beta")
    try:
        now = datetime.now(timezone.utc)
        since = now - timedelta(days=3)

        await _create_run(
            run_id=f"e1-{uuid4()}",
            user_id=user_id,
            graph_id=graph_a,
            status=AgentExecutionStatus.COMPLETED,
            cost_cents=300,
            started_at=now - timedelta(hours=2),
        )
        await _create_run(
            run_id=f"e2-{uuid4()}",
            user_id=user_id,
            graph_id=graph_a,
            status=AgentExecutionStatus.COMPLETED,
            cost_cents=200,
            started_at=now - timedelta(hours=1),
        )
        big_run_id = f"e3-{uuid4()}"
        await _create_run(
            run_id=big_run_id,
            user_id=user_id,
            graph_id=graph_b,
            status=AgentExecutionStatus.COMPLETED,
            cost_cents=1000,
            started_at=now - timedelta(minutes=30),
        )
        failed_run_id = f"e4-{uuid4()}"
        await _create_run(
            run_id=failed_run_id,
            user_id=user_id,
            graph_id=graph_b,
            status=AgentExecutionStatus.FAILED,
            cost_cents=50,
            started_at=now - timedelta(minutes=15),
            node_error_count=2,
        )
        # TERMINATED is also "wasted" cost — locks the `IN ('FAILED', 'TERMINATED')`
        # branch in the failed_cost_cents aggregator.
        terminated_run_id = f"e5-{uuid4()}"
        await _create_run(
            run_id=terminated_run_id,
            user_id=user_id,
            graph_id=graph_b,
            status=AgentExecutionStatus.TERMINATED,
            cost_cents=20,
            started_at=now - timedelta(minutes=10),
        )
        # Zero-cost run — included in run_count but excluded from billable_run_count,
        # so the frontend Avg / run denominator stays honest.
        await _create_run(
            run_id=f"e6-{uuid4()}",
            user_id=user_id,
            graph_id=graph_a,
            status=AgentExecutionStatus.COMPLETED,
            cost_cents=0,
            started_at=now - timedelta(minutes=5),
        )

        summary = await get_user_cost_summary(
            user_id=user_id,
            since=since,
            until=now + timedelta(minutes=1),
            top_runs_limit=3,
        )

        assert summary.total_cents == 1570
        assert summary.run_count == 6
        assert summary.billable_run_count == 5
        # 50 FAILED + 20 TERMINATED
        assert summary.failed_cost_cents == 70

        by_agent = {row.graph_id: row for row in summary.by_agent}
        assert by_agent[graph_a].cost_cents == 500
        # 2 paid runs + 1 zero-cost run on graph_a
        assert by_agent[graph_a].run_count == 3
        assert by_agent[graph_b].cost_cents == 1070
        assert by_agent[graph_b].run_count == 3
        # graph_b leads on total spend so it sorts first
        assert summary.by_agent[0].graph_id == graph_b

        # Top runs ordered by cost desc; limit honored
        assert summary.top_runs[0].execution_id == big_run_id
        assert summary.top_runs[0].cost_cents == 1000
        assert len(summary.top_runs) == 3
    finally:
        await _cleanup(user_id, [graph_a, graph_b])


@pytest.mark.asyncio(loop_scope="session")
async def test_excludes_dry_runs_and_outside_window(server: SpinTestServer):
    user_id = f"cs-filter-{uuid4()}"
    graph_id = f"cs-g-f-{uuid4()}"
    await _create_user(user_id)
    await _create_graph(graph_id, user_id)
    try:
        now = datetime.now(timezone.utc)

        # In window, real run -> counted
        await _create_run(
            run_id=f"in-{uuid4()}",
            user_id=user_id,
            graph_id=graph_id,
            status=AgentExecutionStatus.COMPLETED,
            cost_cents=400,
            started_at=now - timedelta(hours=1),
        )
        # In window, dry run -> excluded
        await _create_run(
            run_id=f"dry-{uuid4()}",
            user_id=user_id,
            graph_id=graph_id,
            status=AgentExecutionStatus.COMPLETED,
            cost_cents=900,
            started_at=now - timedelta(hours=1),
            is_dry_run=True,
        )
        # Outside window -> excluded
        await _create_run(
            run_id=f"old-{uuid4()}",
            user_id=user_id,
            graph_id=graph_id,
            status=AgentExecutionStatus.COMPLETED,
            cost_cents=700,
            started_at=now - timedelta(days=10),
        )

        summary = await get_user_cost_summary(
            user_id=user_id,
            since=now - timedelta(days=1),
            until=now + timedelta(minutes=1),
        )

        assert summary.total_cents == 400
        assert summary.run_count == 1
    finally:
        await _cleanup(user_id, [graph_id])


@pytest.mark.asyncio(loop_scope="session")
async def test_daily_buckets_group_by_utc_date(server: SpinTestServer):
    user_id = f"cs-daily-{uuid4()}"
    graph_id = f"cs-g-d-{uuid4()}"
    await _create_user(user_id)
    await _create_graph(graph_id, user_id)
    try:
        day1 = datetime(2026, 1, 10, 9, 0, tzinfo=timezone.utc)
        day2 = datetime(2026, 1, 11, 9, 0, tzinfo=timezone.utc)

        await _create_run(
            run_id=f"d1a-{uuid4()}",
            user_id=user_id,
            graph_id=graph_id,
            status=AgentExecutionStatus.COMPLETED,
            cost_cents=100,
            started_at=day1,
        )
        await _create_run(
            run_id=f"d1b-{uuid4()}",
            user_id=user_id,
            graph_id=graph_id,
            status=AgentExecutionStatus.COMPLETED,
            cost_cents=250,
            started_at=day1,
        )
        await _create_run(
            run_id=f"d2-{uuid4()}",
            user_id=user_id,
            graph_id=graph_id,
            status=AgentExecutionStatus.COMPLETED,
            cost_cents=400,
            started_at=day2,
        )

        summary = await get_user_cost_summary(
            user_id=user_id,
            since=day1 - timedelta(hours=1),
            until=day2 + timedelta(hours=1),
        )

        assert [d.date for d in summary.daily] == [
            date(2026, 1, 10),
            date(2026, 1, 11),
        ]
        assert summary.daily[0].cost_cents == 350
        assert summary.daily[0].run_count == 2
        assert summary.daily[1].cost_cents == 400
        assert summary.daily[1].run_count == 1
    finally:
        await _cleanup(user_id, [graph_id])


@pytest.mark.asyncio(loop_scope="session")
async def test_daily_buckets_follow_created_at_not_started_at(
    server: SpinTestServer,
):
    # A queued-then-started run lands in the createdAt bucket, not the
    # startedAt bucket — locks the index-hit invariant documented in
    # backend/data/execution_cost_summary.py get_user_cost_summary docstring.
    user_id = f"cs-created-{uuid4()}"
    graph_id = f"cs-g-c-{uuid4()}"
    await _create_user(user_id)
    await _create_graph(graph_id, user_id)
    try:
        created_day = datetime(2026, 2, 5, 23, 30, tzinfo=timezone.utc)
        started_day = datetime(2026, 2, 7, 9, 0, tzinfo=timezone.utc)

        await _create_run(
            run_id=f"q-{uuid4()}",
            user_id=user_id,
            graph_id=graph_id,
            status=AgentExecutionStatus.COMPLETED,
            cost_cents=600,
            started_at=started_day,
            created_at=created_day,
        )

        summary = await get_user_cost_summary(
            user_id=user_id,
            since=created_day - timedelta(hours=1),
            until=started_day + timedelta(hours=1),
        )

        assert [d.date for d in summary.daily] == [date(2026, 2, 5)]
        assert summary.daily[0].cost_cents == 600
    finally:
        await _cleanup(user_id, [graph_id])
