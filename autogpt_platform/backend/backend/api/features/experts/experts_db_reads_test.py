"""The roster, an expert's page and its runs read what each needs side by side:
every stand-in below answers only once all of them are being awaited."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import prisma.models
import pytest
from pydantic import BaseModel

from backend.api.features.experts import experts_db
from backend.util.testing import Rendezvous

USER_ID = "user-1"


class _ExpertView(BaseModel):
    """What ``_to_model`` is given, so the reads' answers are visible."""

    id: str
    latest_run: str | None = None
    weekly_spend: int = 0
    credential_count: int = 0
    credential_providers: list[str] = []


def _to_view(row, latest_run=None, weekly_spend=0) -> _ExpertView:
    return _ExpertView(id=row.id, latest_run=latest_run, weekly_spend=weekly_spend)


def _expert_prisma(**methods) -> MagicMock:
    client = MagicMock()
    for name, method in methods.items():
        setattr(client, name, method)
    return client


@pytest.mark.asyncio
async def test_the_roster_reads_runs_spend_and_credentials_side_by_side() -> None:
    meeting = Rendezvous(3)
    rows = [SimpleNamespace(id="e1"), SimpleNamespace(id="e2")]
    experts = _expert_prisma(find_many=AsyncMock(return_value=rows))
    with (
        patch.object(prisma.models.Expert, "prisma", return_value=experts),
        patch.object(experts_db, "_to_model", _to_view),
        patch.object(
            experts_db,
            "_latest_runs",
            AsyncMock(side_effect=meeting.returning({"e1": "run-1"})),
        ),
        patch.object(
            experts_db,
            "_weekly_spends",
            AsyncMock(side_effect=meeting.returning({"e1": 5, "e2": 7})),
        ),
        patch.object(
            experts_db,
            "_credential_providers",
            AsyncMock(
                side_effect=meeting.returning({"e2": ["github", "github", "slack"]})
            ),
        ) as providers,
    ):
        roster = await experts_db.list_experts(USER_ID)

    assert [view.model_dump() for view in roster] == [
        _ExpertView(id="e1", latest_run="run-1", weekly_spend=5).model_dump(),
        _ExpertView(
            id="e2",
            weekly_spend=7,
            credential_count=3,
            credential_providers=["github", "slack"],
        ).model_dump(),
    ]
    providers.assert_awaited_once_with(USER_ID, rows)


@pytest.mark.asyncio
async def test_an_experts_page_reads_its_run_spend_and_credentials_side_by_side() -> (
    None
):
    meeting = Rendezvous(3)
    row = SimpleNamespace(id="e1")
    experts = _expert_prisma(find_first=AsyncMock(return_value=row))
    with (
        patch.object(prisma.models.Expert, "prisma", return_value=experts),
        patch.object(experts_db, "_to_model", _to_view),
        patch.object(
            experts_db,
            "_latest_runs",
            AsyncMock(side_effect=meeting.returning({"e1": "run-1"})),
        ),
        patch.object(
            experts_db, "get_weekly_spend", AsyncMock(side_effect=meeting.returning(9))
        ),
        patch.object(
            experts_db,
            "_credential_providers",
            AsyncMock(side_effect=meeting.returning({"e1": ["notion"]})),
        ),
    ):
        expert = await experts_db.get_expert(USER_ID, "e1", include_credentials=True)

    assert expert is not None
    assert (
        expert.model_dump()
        == _ExpertView(
            id="e1",
            latest_run="run-1",
            weekly_spend=9,
            credential_count=1,
            credential_providers=["notion"],
        ).model_dump()
    )


@pytest.mark.asyncio
async def test_an_expert_without_credentials_reads_no_grants() -> None:
    meeting = Rendezvous(2)
    experts = _expert_prisma(
        find_first=AsyncMock(return_value=SimpleNamespace(id="e1"))
    )
    providers = AsyncMock(return_value={})
    with (
        patch.object(prisma.models.Expert, "prisma", return_value=experts),
        patch.object(experts_db, "_to_model", _to_view),
        patch.object(
            experts_db, "_latest_runs", AsyncMock(side_effect=meeting.returning({}))
        ),
        patch.object(
            experts_db, "get_weekly_spend", AsyncMock(side_effect=meeting.returning(4))
        ),
        patch.object(experts_db, "_credential_providers", providers),
    ):
        expert = await experts_db.get_expert(USER_ID, "e1")

    assert expert is not None
    assert expert.weekly_spend == 4
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
