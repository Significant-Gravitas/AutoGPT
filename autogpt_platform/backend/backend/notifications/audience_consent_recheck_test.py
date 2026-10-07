"""The audience consumer re-reads the opt-out right before each MailerLite
write, so a change queued just before a refusal is never applied."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.data.notifications import AudienceAction, SubscriberField
from backend.notifications import mailerlite, subscriber_fields
from backend.notifications.notifications import NotificationManager

EMAIL = "sam@example.com"

HANDLERS = {
    AudienceAction.ENROLL_TOUR: "enroll_in_onboarding",
    AudienceAction.ADD_CHANGELOG: "add_to_changelog",
    AudienceAction.REMOVE_CHANGELOG: "remove_from_changelog",
    AudienceAction.ADD_TRIAL: "add_to_trial",
    AudienceAction.REMOVE_TRIAL: "remove_from_trial",
    AudienceAction.UPDATE_FIELDS: "update_fields",
    AudienceAction.SIGNUP: "record_signup",
    AudienceAction.CHECKOUT_OPENED: "record_checkout_opened",
    AudienceAction.UNSUBSCRIBE: "unsubscribe",
}
WRITES = [action for action in AudienceAction if action != AudienceAction.UNSUBSCRIBE]


@pytest.fixture(autouse=True)
def handlers(monkeypatch) -> dict[AudienceAction, AsyncMock]:
    monkeypatch.setattr(mailerlite, "configured", lambda: True)
    mocks = {}
    for action, name in HANDLERS.items():
        mocks[action] = AsyncMock()
        monkeypatch.setattr(mailerlite, name, mocks[action])
    return mocks


async def _consume(action: AudienceAction) -> bool:
    event = subscriber_fields.audience_event(
        action, EMAIL, "user-1", {SubscriberField.STATUS: "signed"}
    )
    assert event is not None
    return await NotificationManager._process_audience_change(
        MagicMock(), event.model_dump_json()
    )


def test_every_action_is_covered():
    assert set(HANDLERS) == set(AudienceAction)


@pytest.mark.asyncio
@pytest.mark.parametrize("action", WRITES, ids=lambda a: a.value)
async def test_a_change_for_an_opted_out_account_is_dropped(
    audience_consent: SimpleNamespace, handlers, action, caplog
):
    audience_consent.is_marketing_opted_out.return_value = True

    with caplog.at_level("DEBUG", logger="backend.notifications.consent"):
        assert await _consume(action)

    audience_consent.is_marketing_opted_out.assert_awaited_once_with("user-1")
    handlers[action].assert_not_awaited()
    assert mailerlite.pseudonym(EMAIL) in caplog.text
    assert EMAIL not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("action", WRITES, ids=lambda a: a.value)
async def test_a_change_for_an_opted_in_account_is_applied(
    audience_consent: SimpleNamespace, handlers, action
):
    assert await _consume(action)

    audience_consent.is_marketing_opted_out.assert_awaited_once_with("user-1")
    handlers[action].assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("opted_out", [True, False])
async def test_the_unsubscribe_always_goes_through(
    audience_consent: SimpleNamespace, handlers, opted_out
):
    """It carries the refusal, so it is the one write an opted-out account
    gets; there is nothing to re-read."""
    audience_consent.is_marketing_opted_out.return_value = opted_out

    assert await _consume(AudienceAction.UNSUBSCRIBE)

    handlers[AudienceAction.UNSUBSCRIBE].assert_awaited_once()
    assert handlers[AudienceAction.UNSUBSCRIBE].await_args.args[0] == EMAIL
    audience_consent.is_marketing_opted_out.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failed_read_retries_rather_than_writes(
    audience_consent: SimpleNamespace, handlers
):
    """Unknown consent is never taken as consent: the message is retried."""
    audience_consent.is_marketing_opted_out.side_effect = RuntimeError("rpc down")

    with pytest.raises(RuntimeError):
        await _consume(AudienceAction.CHECKOUT_OPENED)

    handlers[AudienceAction.CHECKOUT_OPENED].assert_not_awaited()


@pytest.mark.asyncio
async def test_without_mailerlite_nothing_is_read(
    audience_consent: SimpleNamespace, handlers, monkeypatch
):
    monkeypatch.setattr(mailerlite, "configured", lambda: False)

    assert await _consume(AudienceAction.CHECKOUT_OPENED)

    audience_consent.is_marketing_opted_out.assert_not_awaited()
    handlers[AudienceAction.CHECKOUT_OPENED].assert_not_awaited()
