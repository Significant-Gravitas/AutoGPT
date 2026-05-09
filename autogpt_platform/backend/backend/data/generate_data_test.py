"""Tests for the SQL-aggregated user execution summary."""

import logging
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from prisma.enums import AgentExecutionStatus
from prisma.errors import UniqueViolationError
from prisma.models import AgentGraph, AgentGraphExecution, User

from backend.data import generate_data
from backend.data.generate_data import (
    _resolve_agent_name,
    get_user_execution_summary_data,
)
from backend.util.json import SafeJson
from backend.util.test import SpinTestServer

logger = logging.getLogger(__name__)


async def _create_test_user(user_id: str) -> None:
    try:
        await User.prisma().create(
            data={
                "id": user_id,
                "email": f"test-{user_id}@example.com",
                "name": f"Test User {user_id[:8]}",
            }
        )
    except UniqueViolationError:
        # Idempotent test setup: row already exists from a prior run.
        pass


async def _create_graph(graph_id: str, user_id: str, name: str) -> None:
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
        # Idempotent test setup: row already exists from a prior run.
        pass


async def _create_exec(
    exec_id: str,
    user_id: str,
    graph_id: str,
    status: AgentExecutionStatus,
    cost_cents: int,
    walltime_s: float,
    created_at: datetime,
) -> None:
    await AgentGraphExecution.prisma().create(
        data={
            "id": exec_id,
            "agentGraphId": graph_id,
            "agentGraphVersion": 1,
            "executionStatus": status,
            "userId": user_id,
            "createdAt": created_at,
            "stats": SafeJson({"cost": cost_cents, "walltime": walltime_s}),
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
async def test_summary_empty_returns_zero_stats(server: SpinTestServer):
    user_id = f"sum-empty-{uuid4()}"
    await _create_test_user(user_id)
    try:
        now = datetime.now(timezone.utc)
        stats = await get_user_execution_summary_data(
            user_id, now - timedelta(days=1), now
        )
        assert stats.total_executions == 0
        assert stats.successful_runs == 0
        assert stats.failed_runs == 0
        assert stats.total_credits_used == 0
        assert stats.total_execution_time == 0
        assert stats.average_execution_time == 0
        assert stats.cost_breakdown == {}
        assert stats.most_used_agent == "No agents used"
    finally:
        await _cleanup(user_id, [])


@pytest.mark.asyncio(loop_scope="session")
async def test_summary_aggregates_by_status_and_graph(server: SpinTestServer):
    user_id = f"sum-agg-{uuid4()}"
    graph_a = f"graph-a-{uuid4()}"
    graph_b = f"graph-b-{uuid4()}"
    await _create_test_user(user_id)
    await _create_graph(graph_a, user_id, "Alpha")
    await _create_graph(graph_b, user_id, "Beta")

    now = datetime.now(timezone.utc)
    in_window = now - timedelta(hours=1)

    try:
        # Graph A: 2 successful (50 + 75 cents), 1 failed (25 cents)
        await _create_exec(
            f"a1-{uuid4()}",
            user_id,
            graph_a,
            AgentExecutionStatus.COMPLETED,
            50,
            1.5,
            in_window,
        )
        await _create_exec(
            f"a2-{uuid4()}",
            user_id,
            graph_a,
            AgentExecutionStatus.COMPLETED,
            75,
            2.5,
            in_window,
        )
        await _create_exec(
            f"a3-{uuid4()}",
            user_id,
            graph_a,
            AgentExecutionStatus.FAILED,
            25,
            0.5,
            in_window,
        )
        # Graph B: 1 terminated (10 cents) — counted as failed
        await _create_exec(
            f"b1-{uuid4()}",
            user_id,
            graph_b,
            AgentExecutionStatus.TERMINATED,
            10,
            0.1,
            in_window,
        )

        stats = await get_user_execution_summary_data(
            user_id, now - timedelta(hours=2), now
        )

        assert stats.total_executions == 4
        assert stats.successful_runs == 2
        assert stats.failed_runs == 2  # 1 FAILED + 1 TERMINATED
        assert stats.total_credits_used == pytest.approx((50 + 75 + 25 + 10) / 100)
        assert stats.total_execution_time == pytest.approx(1.5 + 2.5 + 0.5 + 0.1)
        assert stats.average_execution_time == pytest.approx(
            (1.5 + 2.5 + 0.5 + 0.1) / 4
        )
        # Most-used agent is Graph A (3 executions vs Graph B's 1)
        assert stats.most_used_agent == "Alpha"
        # Cost breakdown keyed by agent name
        assert stats.cost_breakdown == {
            "Alpha": pytest.approx((50 + 75 + 25) / 100),
            "Beta": pytest.approx(10 / 100),
        }
    finally:
        await _cleanup(user_id, [graph_a, graph_b])


@pytest.mark.asyncio(loop_scope="session")
async def test_summary_sums_costs_on_agent_name_collision(server: SpinTestServer):
    # Two distinct agentGraphIds resolving to the same display name must sum
    # into the same cost_breakdown bucket, not overwrite each other.
    user_id = f"sum-collide-{uuid4()}"
    graph_a = f"graph-c1-{uuid4()}"
    graph_b = f"graph-c2-{uuid4()}"
    await _create_test_user(user_id)
    await _create_graph(graph_a, user_id, "Scraper")
    await _create_graph(graph_b, user_id, "Scraper")

    now = datetime.now(timezone.utc)
    in_window = now - timedelta(hours=1)

    try:
        await _create_exec(
            f"c1-{uuid4()}",
            user_id,
            graph_a,
            AgentExecutionStatus.COMPLETED,
            30,
            1.0,
            in_window,
        )
        await _create_exec(
            f"c2-{uuid4()}",
            user_id,
            graph_b,
            AgentExecutionStatus.COMPLETED,
            70,
            2.0,
            in_window,
        )

        stats = await get_user_execution_summary_data(
            user_id, now - timedelta(hours=2), now
        )

        assert stats.cost_breakdown == {
            "Scraper": pytest.approx((30 + 70) / 100),
        }
        assert stats.total_credits_used == pytest.approx(100 / 100)
    finally:
        await _cleanup(user_id, [graph_a, graph_b])


@pytest.mark.asyncio(loop_scope="session")
async def test_summary_excludes_out_of_window(server: SpinTestServer):
    user_id = f"sum-window-{uuid4()}"
    graph_a = f"graph-w-{uuid4()}"
    await _create_test_user(user_id)
    await _create_graph(graph_a, user_id, "InWindow")

    now = datetime.now(timezone.utc)
    in_window = now - timedelta(hours=1)
    out_of_window = now - timedelta(days=10)

    try:
        await _create_exec(
            f"in-{uuid4()}",
            user_id,
            graph_a,
            AgentExecutionStatus.COMPLETED,
            100,
            1.0,
            in_window,
        )
        await _create_exec(
            f"out-{uuid4()}",
            user_id,
            graph_a,
            AgentExecutionStatus.COMPLETED,
            200,
            2.0,
            out_of_window,
        )

        stats = await get_user_execution_summary_data(
            user_id, now - timedelta(hours=2), now
        )
        assert stats.total_executions == 1
        assert stats.total_credits_used == pytest.approx(100 / 100)
    finally:
        await _cleanup(user_id, [graph_a])


@pytest.mark.asyncio
async def test_resolve_agent_name_falls_back_on_exception(
    monkeypatch: pytest.MonkeyPatch,
):
    # Direct unit test: when get_graph_metadata raises, _resolve_agent_name
    # logs and returns the short-id fallback.
    async def boom(graph_id: str, version=None):
        raise RuntimeError("simulated DB blip")

    monkeypatch.setattr(generate_data, "get_graph_metadata", boom)
    name = await _resolve_agent_name("abcdef0123456789")
    assert name == "Agent abcdef01"


@pytest.mark.asyncio
async def test_resolve_agent_name_falls_back_when_metadata_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    # When get_graph_metadata returns None (e.g. graph not found),
    # _resolve_agent_name must produce a short-id label.
    async def returns_none(graph_id: str, version=None):
        return None

    monkeypatch.setattr(generate_data, "get_graph_metadata", returns_none)
    name = await _resolve_agent_name("abcdef0123456789")
    assert name == "Agent abcdef01"
