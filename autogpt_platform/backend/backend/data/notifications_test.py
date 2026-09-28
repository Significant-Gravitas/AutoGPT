"""Regression test for AUTOGPT-SERVER-5E4.

Reading a user's notification preference must never fail because of the
format of an email address that is already stored. Pydantic's EmailStr
rejects reserved domains such as `.test`, which turned every preference
lookup for such a user into a 500.
"""

from unittest.mock import MagicMock

import pytest
from prisma.enums import BriefingFrequency
from pydantic import ValidationError

from backend.data.notifications import (
    AudienceAction,
    AudienceEventModel,
    NotificationPreference,
    NotificationPreferenceDTO,
)
from backend.data.user import _preference_from_user

RESERVED_EMAIL = "test-1@autogpt.test"


def test_notification_preference_accepts_reserved_domain():
    preference = NotificationPreference(user_id="u1", email=RESERVED_EMAIL)
    assert preference.email == RESERVED_EMAIL


def test_notification_preference_dto_still_rejects_invalid_email():
    with pytest.raises(ValidationError):
        NotificationPreferenceDTO(
            email="not-an-email",
            briefing_frequency=BriefingFrequency.WEEKLY,
            alerts_enabled=True,
            store_verdicts_enabled=True,
        )


def test_audience_event_accepts_reserved_domain():
    event = AudienceEventModel(
        action=AudienceAction.ENROLL_TOUR, email=RESERVED_EMAIL, user_id="u1"
    )
    assert event.email == RESERVED_EMAIL


def test_preference_from_user_accepts_reserved_domain():
    user = MagicMock(
        id="u1",
        email=RESERVED_EMAIL,
        briefingFrequency=BriefingFrequency.WEEKLY,
        alertsEnabled=True,
        notifyOnStoreVerdict=True,
        maxEmailsPerDay=3,
    )
    preference = _preference_from_user(user)
    assert preference.email == RESERVED_EMAIL
    assert preference.user_id == "u1"
