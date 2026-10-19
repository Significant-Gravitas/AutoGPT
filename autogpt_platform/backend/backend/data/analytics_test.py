"""DB-backed tests for the execution accuracy trends query (#15640)."""

import logging
from datetime import datetime, timedelta, timezone
from statistics import mean
from uuid import uuid4

import pytest
from prisma.enums import AgentExecutionStatus
from prisma.models import AgentGraph, AgentGraphExecution, User

from backend.data.analytics import get_accuracy_trends_and_alerts
from backend.util.json import SafeJson
from backend.util.test import SpinTestServer

logger = logging.getLogger(__name__)


@pytest.fixture
def day_anchor() -> datetime:
    now = datetime.now(timezone.utc)
    today = now.replace(hour=12, minute=0, second=0, microsecond=0)
    if today > now:
        today -= timedelta(days=1)
    return today


def _day(days_ago: int, anchor: datetime) -> datetime:
    return anchor - timedelta(days=days_ago)


async def _setup() -> tuple[str, str]:
    user_id = f"acc-{uuid4()}"
    graph_id = f"acc-graph-{uuid4()}"
    await User.prisma().create(
        data={
            "id": user_id,
            "email": f"{user_id}@example.com",
            "name": "Accuracy Trends Test",
        }
    )
    await AgentGraph.prisma().create(
        data={
            "id": graph_id,
            "version": 1,
            "name": "Accuracy Trends Test",
            "description": "test",
            "userId": user_id,
            "isActive": True,
        }
    )
    return user_id, graph_id


async def _run(
    user_id: str, graph_id: str, created_at: datetime, score: float | None
) -> None:
    stats: dict[str, object] = {"cost": 0}
    if score is not None:
        stats["correctness_score"] = score
    await AgentGraphExecution.prisma().create(
        data={
            "id": str(uuid4()),
            "agentGraphId": graph_id,
            "agentGraphVersion": 1,
            "executionStatus": AgentExecutionStatus.COMPLETED,
            "userId": user_id,
            "createdAt": created_at,
            "startedAt": created_at,
            "endedAt": created_at + timedelta(seconds=1),
            "stats": SafeJson(stats),
        }
    )


async def _cleanup(user_id: str, graph_id: str) -> None:
    try:
        await AgentGraphExecution.prisma().delete_many(where={"userId": user_id})
        await AgentGraph.prisma().delete_many(where={"id": graph_id})
        await User.prisma().delete_many(where={"id": user_id})
    except Exception as exc:
        logger.warning("cleanup for %s failed: %s", user_id, exc)


# Scored days, oldest first: a steadily falling score.
_SCORED_DAYS = {5: 0.9, 4: 0.8, 3: 0.7, 2: 0.6, 1: 0.5}


async def _seed_falling_series(
    user_id: str, graph_id: str, anchor: datetime, *, unscored_today: bool
) -> None:
    for days_ago, score in _SCORED_DAYS.items():
        await _run(user_id, graph_id, _day(days_ago, anchor), score)
        # An unscored run on a scored day must not dilute that day's score.
        await _run(user_id, graph_id, _day(days_ago, anchor) + timedelta(hours=1), None)
    if unscored_today:
        # A day with only unscored runs must not become a row.
        await _run(user_id, graph_id, _day(0, anchor), None)
        await _run(user_id, graph_id, _day(0, anchor) + timedelta(minutes=5), None)


@pytest.mark.asyncio(loop_scope="session")
async def test_historical_data_is_oldest_first_and_scored_days_only(
    server: SpinTestServer,
    day_anchor: datetime,
):
    user_id, graph_id = await _setup()
    try:
        await _seed_falling_series(user_id, graph_id, day_anchor, unscored_today=True)

        result = await get_accuracy_trends_and_alerts(
            graph_id=graph_id,
            days_back=30,
            drop_threshold=10.0,
            include_historical=True,
        )

        history = result.historical_data
        assert history is not None
        dates = [h.date for h in history]
        assert dates == sorted(dates)
        assert len(set(dates)) == len(dates)
        assert len(history) == len(_SCORED_DAYS)
        assert all(h.daily_score is not None for h in history)

        expected_scores = [s * 100 for s in _SCORED_DAYS.values()]
        assert [h.daily_score for h in history] == pytest.approx(expected_scores)

        # The 3-day average is the mean of the last 3 *scored* days.
        assert history[-1].three_day_avg == pytest.approx(mean(expected_scores[-3:]))

        # latest_data is the most recent scored day, not the unscored today.
        assert result.latest_data.date == history[-1].date
        assert result.latest_data.daily_score == pytest.approx(50.0)
        assert result.latest_data.three_day_avg == pytest.approx(60.0)
    finally:
        await _cleanup(user_id, graph_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_latest_and_alert_match_with_and_without_history(
    server: SpinTestServer,
    day_anchor: datetime,
):
    """The monitor calls without history (LIMIT 1); it must see the same
    latest row and alert as the chart's call."""
    user_id, graph_id = await _setup()
    try:
        await _seed_falling_series(user_id, graph_id, day_anchor, unscored_today=False)

        with_history = await get_accuracy_trends_and_alerts(
            graph_id=graph_id, days_back=30, include_historical=True
        )
        without_history = await get_accuracy_trends_and_alerts(
            graph_id=graph_id, days_back=30, include_historical=False
        )

        assert without_history.historical_data is None
        assert without_history.latest_data == with_history.latest_data
        assert with_history.historical_data is not None
        assert without_history.latest_data == with_history.historical_data[-1]

        # Falling series: 3-day avg 60 vs 7-day avg 70 -> ~14.3% drop >= 10%.
        assert with_history.alert is not None
        assert without_history.alert is not None
        assert without_history.alert.drop_percent == pytest.approx(
            (70.0 - 60.0) / 70.0 * 100
        )
        assert without_history.alert.three_day_avg == pytest.approx(60.0)
        assert without_history.alert.seven_day_avg == pytest.approx(70.0)
    finally:
        await _cleanup(user_id, graph_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_flat_series_with_unscored_today_has_no_alert(
    server: SpinTestServer,
    day_anchor: datetime,
):
    user_id, graph_id = await _setup()
    try:
        for days_ago in range(1, 8):
            await _run(user_id, graph_id, _day(days_ago, day_anchor), 0.8)
        await _run(user_id, graph_id, _day(0, day_anchor), None)

        result = await get_accuracy_trends_and_alerts(
            graph_id=graph_id, days_back=30, drop_threshold=10.0
        )

        assert result.alert is None
        assert result.latest_data.daily_score == pytest.approx(80.0)
        assert result.latest_data.date.date() == _day(1, day_anchor).date()
    finally:
        await _cleanup(user_id, graph_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_only_unscored_runs_returns_empty_latest(
    server: SpinTestServer, day_anchor: datetime
):
    user_id, graph_id = await _setup()
    try:
        await _run(user_id, graph_id, _day(1, day_anchor), None)
        await _run(user_id, graph_id, _day(0, day_anchor), None)

        result = await get_accuracy_trends_and_alerts(
            graph_id=graph_id, days_back=30, include_historical=True
        )

        assert result.alert is None
        assert result.latest_data.daily_score is None
        assert result.latest_data.three_day_avg is None
        assert result.historical_data is None
    finally:
        await _cleanup(user_id, graph_id)
