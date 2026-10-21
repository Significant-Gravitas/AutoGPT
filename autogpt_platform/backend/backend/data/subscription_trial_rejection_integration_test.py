import pytest
from prisma.models import SubscriptionTrial

from backend.data import subscription_trial_integration_fixtures as fixtures
from backend.data.subscription_trial import get_subscription_trial
from backend.data.subscription_trial_rejection import TrialRejectionReason

enrollment = fixtures.enrollment
pytestmark = fixtures.pytestmark


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", list(TrialRejectionReason))
async def test_rejection_reason_round_trips_through_database(enrollment, reason):
    await SubscriptionTrial.prisma().update(
        where={"id": enrollment.id},
        data={"status": "canceled", "rejectionReason": reason},
    )
    trial = await get_subscription_trial(enrollment.user_id)
    assert trial is not None
    assert trial.rejection_reason == reason
    assert not trial.active
