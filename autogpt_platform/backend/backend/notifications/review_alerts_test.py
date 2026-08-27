from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.notifications.review_alerts import sync_awaiting_review


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("metadata_org", "metadata_team", "expected_name"),
    [
        ("org-1", "team-1", "Named Agent"),
        ("org-other", "team-other", "Agent graph-12"),
    ],
)
async def test_review_alert_uses_graph_name_only_for_matching_scope(
    mocker, metadata_org, metadata_team, expected_name
):
    query = AsyncMock()
    query.find_many.return_value = [
        SimpleNamespace(
            createdAt=datetime(2026, 8, 27, 12, 30, tzinfo=timezone.utc),
            organizationId="org-1",
            teamId="team-1",
        )
    ]
    mocker.patch(
        "backend.notifications.review_alerts.PendingHumanReview.prisma",
        autospec=True,
        return_value=query,
    )
    mocker.patch(
        "backend.notifications.review_alerts.get_graph_metadata",
        new=AsyncMock(
            return_value=SimpleNamespace(
                name="Named Agent",
                organization_id=metadata_org,
                team_id=metadata_team,
            )
        ),
    )
    raise_alert = mocker.patch(
        "backend.notifications.review_alerts.alerts_db.raise_alert_condition",
        new_callable=AsyncMock,
    )

    await sync_awaiting_review("user-1", "graph-123456", "exec-1")

    raised = raise_alert.await_args.kwargs
    assert raised["data"]["agent"] == expected_name
    assert raised["organization_id"] == "org-1"
    assert raised["team_id"] == "team-1"
