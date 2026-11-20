"""Service messages are about the customer's account, not a promotion, so the
volume knob doesn't apply to them."""

import pytest
from prisma.enums import BriefingFrequency, NotificationType

from backend.data.notifications import NotificationPreference
from backend.notifications.preferences import SERVICE_MESSAGES, wants_notification

EVERYTHING_OFF = NotificationPreference(
    user_id="u1",
    email="sam@example.com",
    briefing_frequency=BriefingFrequency.OFF,
    alerts_enabled=False,
    store_verdicts_enabled=False,
)
EVERYTHING_ON = NotificationPreference(
    user_id="u1",
    email="sam@example.com",
    briefing_frequency=BriefingFrequency.WEEKLY,
    alerts_enabled=True,
    store_verdicts_enabled=True,
)


@pytest.mark.parametrize("notification_type", sorted(SERVICE_MESSAGES, key=str))
def test_billing_messages_are_sent_even_to_someone_who_opted_out(notification_type):
    assert wants_notification(EVERYTHING_OFF, notification_type)


def test_the_briefing_follows_the_frequency():
    assert wants_notification(EVERYTHING_ON, NotificationType.BRIEFING)
    assert not wants_notification(EVERYTHING_OFF, NotificationType.BRIEFING)


def test_alerts_only_means_alerts_survive_the_digest_being_off():
    alerts_only = EVERYTHING_OFF.model_copy(update={"alerts_enabled": True})
    assert not wants_notification(alerts_only, NotificationType.BRIEFING)
    assert wants_notification(alerts_only, NotificationType.ALERT)


def test_store_verdicts_have_their_own_switch():
    assert wants_notification(EVERYTHING_ON, NotificationType.VERDICT)
    assert not wants_notification(EVERYTHING_OFF, NotificationType.VERDICT)
