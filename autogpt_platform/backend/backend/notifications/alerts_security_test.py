from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.enums import AlertConditionStatus
from prisma.models import AlertCondition

from backend.data.alerts import (
    alert_condition_sources_are_live,
    finalize_alert_delivery,
    finalize_briefing_delivery,
)
from backend.data.notifications import NotificationScope

NOW = datetime(2026, 8, 26, 12, 0, tzinfo=timezone.utc)


def _condition(*, deleted: bool = False, organization_id: str | None = "org-1"):
    execution = SimpleNamespace(
        id="exec-1",
        userId="user-1",
        organizationId=organization_id,
        teamId="team-1" if organization_id else None,
        agentGraphId="graph-1",
        isDeleted=deleted,
    )
    return SimpleNamespace(
        id="condition-1",
        userId="user-1",
        organizationId=organization_id,
        teamId="team-1" if organization_id else None,
        sourceGraphExecutionId="exec-1",
        SourceGraphExecution=execution,
    )


@pytest.mark.asyncio
async def test_soft_deleted_alert_source_is_not_deliverable() -> None:
    client = MagicMock()
    client.find_many = AsyncMock(return_value=[_condition(deleted=True)])
    with patch.object(AlertCondition, "prisma", return_value=client):
        assert not await alert_condition_sources_are_live(
            "user-1",
            ["condition-1"],
            [NotificationScope(organization_id="org-1", team_id="team-1")],
        )


@pytest.mark.asyncio
async def test_exact_live_alert_source_is_deliverable() -> None:
    client = MagicMock()
    client.find_many = AsyncMock(return_value=[_condition()])
    with patch.object(AlertCondition, "prisma", return_value=client):
        assert await alert_condition_sources_are_live(
            "user-1",
            ["condition-1"],
            [NotificationScope(organization_id="org-1", team_id="team-1")],
        )


@pytest.mark.asyncio
async def test_alert_finalizer_is_exact_and_compare_and_swap() -> None:
    client = MagicMock()
    client.update_many = AsyncMock(return_value=1)
    scope = NotificationScope(organization_id="org-1", team_id="team-1")
    with patch.object(AlertCondition, "prisma", return_value=client):
        await finalize_alert_delivery("user-1", ["condition-1"], [scope], NOW)

    where = client.update_many.await_args.kwargs["where"]
    assert where == {
        "id": {"in": ["condition-1"]},
        "userId": "user-1",
        "status": AlertConditionStatus.PENDING,
        "OR": [{"organizationId": "org-1", "teamId": "team-1"}],
    }


@pytest.mark.asyncio
async def test_older_briefing_cannot_move_cadence_backwards() -> None:
    tx = MagicMock()
    tx.alertcondition.update_many = AsyncMock()
    tx.user.update_many = AsyncMock()
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=tx)
    context.__aexit__ = AsyncMock(return_value=False)
    fake_prisma = MagicMock()
    fake_prisma.tx.return_value = context
    with patch("backend.data.alerts.prisma", fake_prisma):
        await finalize_briefing_delivery("user-1", [], [NotificationScope()], NOW, NOW)

    where = tx.user.update_many.await_args.kwargs["where"]
    assert where["OR"] == [
        {"lastBriefingAt": None},
        {"lastBriefingAt": {"lt": NOW}},
    ]
    assert tx.user.update_many.await_args.kwargs["data"] == {"lastBriefingAt": NOW}
