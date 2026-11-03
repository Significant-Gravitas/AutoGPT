"""An expert's runs read their reviews and outputs side by side: each stand-in
answers only once both are being awaited.

The roster and an expert's page keep reading their runs before their
credentials, because the credentials read can seed an expert's grants: a
request that fails before then must seed nothing."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import prisma.models
import pytest

from backend.api.features.experts import experts_db
from backend.util.testing import Rendezvous

USER_ID = "user-1"


def _expert_prisma(**methods) -> MagicMock:
    client = MagicMock()
    for name, method in methods.items():
        setattr(client, name, method)
    return client


@pytest.mark.asyncio
async def test_a_roster_whose_runs_cannot_be_read_seeds_no_grants() -> None:
    rows = [SimpleNamespace(id="e1"), SimpleNamespace(id="e2")]
    providers = AsyncMock(return_value={})
    with (
        patch.object(
            prisma.models.Expert,
            "prisma",
            return_value=_expert_prisma(find_many=AsyncMock(return_value=rows)),
        ),
        patch.object(
            experts_db,
            "_latest_runs",
            AsyncMock(side_effect=RuntimeError("runs read failed")),
        ),
        patch.object(experts_db, "_weekly_spends", AsyncMock(return_value={})),
        patch.object(experts_db, "_credential_providers", providers),
        pytest.raises(RuntimeError, match="runs read failed"),
    ):
        await experts_db.list_experts(USER_ID)

    providers.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_experts_page_whose_run_cannot_be_read_seeds_no_grants() -> None:
    providers = AsyncMock(return_value={})
    with (
        patch.object(
            prisma.models.Expert,
            "prisma",
            return_value=_expert_prisma(
                find_first=AsyncMock(return_value=SimpleNamespace(id="e1"))
            ),
        ),
        patch.object(
            experts_db,
            "_latest_runs",
            AsyncMock(side_effect=RuntimeError("runs read failed")),
        ),
        patch.object(experts_db, "get_weekly_spend", AsyncMock(return_value=0)),
        patch.object(experts_db, "_credential_providers", providers),
        pytest.raises(RuntimeError, match="runs read failed"),
    ):
        await experts_db.get_expert(USER_ID, "e1", include_credentials=True)

    providers.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_experts_runs_read_reviews_and_outputs_side_by_side() -> None:
    meeting = Rendezvous(2)
    expert_row = SimpleNamespace(id="e1", Workflows=[])
    executions = [
        SimpleNamespace(id="x1", agentGraphId="g1"),
        SimpleNamespace(id="x2", agentGraphId="g1"),
    ]
    reviews = _expert_prisma(
        find_many=AsyncMock(
            side_effect=meeting.returning([SimpleNamespace(graphExecId="x2")])
        )
    )

    def to_run(execution, workflow, output_type, output_key, *, needs_review):
        return (execution.id, output_type, output_key, needs_review)

    with (
        patch.object(
            prisma.models.Expert,
            "prisma",
            return_value=_expert_prisma(find_first=AsyncMock(return_value=expert_row)),
        ),
        patch.object(
            prisma.models.AgentGraphExecution,
            "prisma",
            return_value=_expert_prisma(find_many=AsyncMock(return_value=executions)),
        ),
        patch.object(prisma.models.PendingHumanReview, "prisma", return_value=reviews),
        patch.object(
            experts_db,
            "_classify_run_outputs",
            AsyncMock(side_effect=meeting.returning({"x1": ("text", "hello")})),
        ),
        patch.object(experts_db, "_to_expert_run", to_run),
    ):
        runs = await experts_db.list_expert_runs(USER_ID, "e1")

    assert runs == [("x1", "text", "hello", False), ("x2", "unknown", None, True)]
