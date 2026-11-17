from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.data import subscription_trial


@pytest.mark.asyncio
async def test_unreadable_trial_row_is_left_out_of_the_last_call_listing(caplog):
    rows = [
        SimpleNamespace(id="trial-1", userId="user-1"),
        SimpleNamespace(id="trial-2", userId="user-2"),
    ]
    good = MagicMock()

    def from_db(row):
        if row.id == "trial-1":
            raise ValueError("offer no longer validates")
        return good

    now = datetime.now(UTC)
    find_many = AsyncMock(return_value=rows)
    with (
        patch.object(
            subscription_trial.SubscriptionTrial,
            "prisma",
            return_value=MagicMock(find_many=find_many),
        ),
        patch.object(subscription_trial.TrialState, "from_db", side_effect=from_db),
        caplog.at_level("ERROR", logger=subscription_trial.logger.name),
    ):
        got = await subscription_trial.list_trials_ending_unrenewed(
            now, now + timedelta(hours=7)
        )
    assert got == [good]
    assert "trial-1" in caplog.text
    where = find_many.await_args.kwargs["where"]
    assert where["cancelAtPeriodEnd"] is True
    assert where["status"] == "trialing"
    assert where["convertedAt"] is None
    assert where["stripeSubscriptionId"] == {"not": None}
    assert where["endsAt"] == {"gt": now, "lte": now + timedelta(hours=7)}
