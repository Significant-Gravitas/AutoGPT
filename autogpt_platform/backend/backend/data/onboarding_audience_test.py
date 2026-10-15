"""The role picked in onboarding is queued for the account's MailerLite
subscriber, unless the account may not be in MailerLite at all, and none of
it can cost the profile."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.data import onboarding_audience
from backend.data.notifications import (
    AudienceAction,
    NotificationResult,
    SubscriberField,
)
from backend.data.onboarding_role import OnboardingRole
from backend.notifications import subscriber_fields

EMAIL = "sam@example.com"
OPTED_OUT = datetime(2026, 10, 2, tzinfo=UTC)


# The root conftest spins a full test server for every test via an autouse
# session fixture. These are unit tests over mocks, so shadow it for this
# module, as backend/data/db_test.py does.
@pytest.fixture(scope="session")
def server():
    yield None


@pytest.fixture(scope="session", autouse=True)
def graph_cleanup():
    yield


def _user(**overrides) -> SimpleNamespace:
    fields = dict(email=EMAIL, timezone="America/Chicago", marketing_opt_out_at=None)
    return SimpleNamespace(**{**fields, **overrides})


@pytest.fixture
def queued(monkeypatch):
    queue = AsyncMock(return_value=NotificationResult(success=True))
    monkeypatch.setattr(subscriber_fields, "queue_audience_change", queue)
    monkeypatch.setattr(
        onboarding_audience, "get_user_by_id", AsyncMock(return_value=_user())
    )
    return queue


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "role, fields",
    [
        (OnboardingRole(choice="HR/People"), ("HR / People", None)),
        (OnboardingRole(choice="Other", other="Dentist"), ("Other", "Dentist")),
    ],
)
async def test_the_pick_is_queued_as_labelled(queued, role, fields):
    await onboarding_audience.queue_onboarding_role("user-1", role)
    event = queued.await_args.args[0]
    assert event.action is AudienceAction.ONBOARDING_PROFILE
    assert event.email == EMAIL
    assert event.fields == {
        SubscriberField.ROLE: fields[0],
        SubscriberField.ROLE_OTHER: fields[1],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "user",
    [
        _user(marketing_opt_out_at=OPTED_OUT),
        _user(timezone="Asia/Tehran"),
        _user(email="sam@firma.ru"),
    ],
    ids=["opted-out", "iran", "russia"],
)
async def test_nothing_is_queued_for_an_account_mailerlite_may_not_hold(
    queued, monkeypatch, user
):
    monkeypatch.setattr(
        onboarding_audience, "get_user_by_id", AsyncMock(return_value=user)
    )
    await onboarding_audience.queue_onboarding_role(
        "user-1", OnboardingRole(choice="Marketing")
    )
    queued.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failure_never_reaches_the_profile(queued, monkeypatch):
    monkeypatch.setattr(
        onboarding_audience,
        "get_user_by_id",
        AsyncMock(side_effect=RuntimeError("db down")),
    )
    await onboarding_audience.queue_onboarding_role(
        "user-1", OnboardingRole(choice="Marketing")
    )
    queued.assert_not_awaited()
