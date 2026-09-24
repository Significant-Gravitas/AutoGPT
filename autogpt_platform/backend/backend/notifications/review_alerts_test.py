from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.notifications import review_alerts


@pytest.fixture
def alerts(mocker):
    db = MagicMock(
        resolve_alert_condition=AsyncMock(), raise_alert_condition=AsyncMock()
    )
    mocker.patch.object(review_alerts, "alerts_db", db)
    return db


def _waiting(mocker, rows):
    find_many = AsyncMock(return_value=rows)
    mocker.patch.object(
        review_alerts.PendingHumanReview,
        "prisma",
        return_value=MagicMock(find_many=find_many),
    )
    return find_many


@pytest.mark.asyncio
async def test_a_chat_alert_counts_old_shape_rows_and_clears_the_old_key(
    mocker, alerts
):
    find_many = _waiting(mocker, [])

    await review_alerts.sync_awaiting_review("u1", session_id="s1")

    where = find_many.await_args.kwargs["where"]
    assert {"sessionId": "s1"} in where["OR"]
    assert {"graphExecId": "copilot-session-s1"} in where["OR"]
    resolved = [c.args[1] for c in alerts.resolve_alert_condition.await_args_list]
    assert resolved == ["awaiting_review:copilot-session-s1", "awaiting_review:chat:s1"]


@pytest.mark.asyncio
async def test_a_waiting_chat_review_raises_a_chat_alert(mocker, alerts):
    _waiting(mocker, [MagicMock(createdAt=datetime(2026, 9, 24, tzinfo=timezone.utc))])

    await review_alerts.sync_awaiting_review("u1", session_id="s1")

    kwargs = alerts.raise_alert_condition.await_args.kwargs
    assert kwargs["cause_key"] == "awaiting_review:chat:s1"
    assert kwargs["data"]["cta_path"] == "/copilot?sessionId=s1"
