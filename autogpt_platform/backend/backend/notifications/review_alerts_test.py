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
    """The waiting reviews as the alert reads them: their number, and the
    oldest one's `createdAt`. Never the rows themselves, which hold payloads."""
    count = AsyncMock(return_value=len(rows))
    mocker.patch.object(
        review_alerts.PendingHumanReview,
        "prisma",
        return_value=MagicMock(spec_set=["count"], count=count),
    )
    find_first = AsyncMock(return_value=rows[0] if rows else None)
    mocker.patch.object(
        review_alerts._OldestWaiting,
        "prisma",
        return_value=MagicMock(spec_set=["find_first"], find_first=find_first),
    )
    return count


@pytest.mark.asyncio
async def test_a_chat_alert_counts_old_shape_rows_and_clears_the_old_key(
    mocker, alerts
):
    count = _waiting(mocker, [])

    await review_alerts.sync_awaiting_review("u1", chat_session_id="s1")

    where = count.await_args.kwargs["where"]
    assert {"chatSessionId": "s1"} in where["OR"]
    assert {"graphExecId": "copilot-session-s1"} in where["OR"]
    resolved = [c.args[1] for c in alerts.resolve_alert_condition.await_args_list]
    assert resolved == ["awaiting_review:copilot-session-s1", "awaiting_review:chat:s1"]


@pytest.mark.asyncio
async def test_a_waiting_chat_review_raises_a_chat_alert(mocker, alerts):
    _waiting(mocker, [MagicMock(createdAt=datetime(2026, 9, 24, tzinfo=timezone.utc))])

    await review_alerts.sync_awaiting_review("u1", chat_session_id="s1")

    kwargs = alerts.raise_alert_condition.await_args.kwargs
    assert kwargs["cause_key"] == "awaiting_review:chat:s1"
    assert kwargs["data"]["cta_path"] == "/copilot?sessionId=s1"


@pytest.mark.asyncio
async def test_an_agent_alert_counts_its_waiting_reviews_without_loading_them(
    mocker, alerts
):
    oldest = MagicMock(createdAt=datetime(2026, 9, 24, 8, 5, tzinfo=timezone.utc))
    count = _waiting(mocker, [oldest, MagicMock(), MagicMock()])
    mocker.patch.object(
        review_alerts, "get_graph_metadata", AsyncMock(return_value=None)
    )

    await review_alerts.sync_awaiting_review("u1", "graph-12345")

    assert count.await_args.kwargs["where"] == {
        "graphId": "graph-12345",
        "userId": "u1",
        "status": review_alerts.ReviewStatus.WAITING,
    }
    kwargs = alerts.raise_alert_condition.await_args.kwargs
    assert kwargs["cause_key"] == "awaiting_review:graph-12345"
    assert kwargs["data"]["count"] == 3
    assert kwargs["data"]["since_label"] == "24 Sep 2026, 08:05 UTC"
    # The alert reads the oldest review through a model that leaves `payload` out.
    assert set(review_alerts._OldestWaiting.model_fields) == {"createdAt"}
