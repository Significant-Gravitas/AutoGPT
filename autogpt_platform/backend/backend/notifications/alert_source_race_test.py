import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from prisma.enums import AlertCause

from backend.data import alerts
from backend.util.exceptions import DatabaseError


@pytest.mark.asyncio
async def test_first_source_alert_waits_for_delete_and_rejects_stale_source() -> None:
    deletion_finished = asyncio.Event()
    raise_waiting_on_graph = asyncio.Event()
    live_source = SimpleNamespace(
        id="execution-1",
        userId="user-1",
        organizationId="org-1",
        teamId="team-1",
        agentGraphId="graph-1",
        isDeleted=False,
    )
    deleted_source = SimpleNamespace(**{**vars(live_source), "isDeleted": True})
    execution_client = SimpleNamespace(
        find_unique=AsyncMock(side_effect=[live_source, deleted_source])
    )
    condition_client = SimpleNamespace(
        find_unique=AsyncMock(),
        create=AsyncMock(),
    )

    @asynccontextmanager
    async def graph_barrier(graph_ids):
        assert graph_ids == ["graph-1"]
        raise_waiting_on_graph.set()
        await deletion_finished.wait()
        yield

    with (
        patch.object(
            alerts.AgentGraphExecution,
            "prisma",
            return_value=execution_client,
        ),
        patch.object(alerts.AlertCondition, "prisma", return_value=condition_client),
        patch.object(alerts, "agent_graph_attachment_barriers", graph_barrier),
    ):
        first_raise = asyncio.create_task(
            alerts.raise_alert_condition(
                user_id="user-1",
                cause=AlertCause.LOW_BALANCE,
                cause_key="low-balance:execution-1",
                data={"balance": 0},
                organization_id="org-1",
                team_id="team-1",
                source_graph_execution_id="execution-1",
            )
        )
        await raise_waiting_on_graph.wait()
        await asyncio.sleep(0)
        assert not first_raise.done()

        deletion_finished.set()
        with pytest.raises(DatabaseError, match="no longer live"):
            await first_raise

    condition_client.find_unique.assert_not_awaited()
    condition_client.create.assert_not_awaited()
