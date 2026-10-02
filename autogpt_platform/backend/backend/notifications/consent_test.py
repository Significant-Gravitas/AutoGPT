"""Someone who refused marketing never enters MailerLite, and the skip is
logged at debug level by pseudonym, never by address."""

import logging
from datetime import UTC, datetime

import pytest

from backend.data.notifications import AudienceAction
from backend.data.user import BillingEmailRecipient
from backend.notifications import consent
from backend.notifications.mailerlite import _pseudonym

EMAIL = "sam@example.com"
OPTED_OUT = datetime(2026, 10, 2, 12, 0, tzinfo=UTC)


def _user(opted_out_at: datetime | None) -> BillingEmailRecipient:
    return BillingEmailRecipient(
        id="user-1", email=EMAIL, marketing_opt_out_at=opted_out_at
    )


def test_marketing_is_allowed_until_they_opt_out():
    assert consent.marketing_allowed(_user(None))
    assert not consent.marketing_allowed(_user(OPTED_OUT))


def test_an_allowed_change_logs_nothing(caplog):
    with caplog.at_level(logging.DEBUG, logger=consent.__name__):
        assert consent.audience_change_allowed(
            _user(None), AudienceAction.CHECKOUT_OPENED
        )
    assert caplog.records == []


@pytest.mark.parametrize("action", list(AudienceAction))
def test_a_refused_change_is_logged_by_pseudonym_at_debug(caplog, action):
    with caplog.at_level(logging.DEBUG, logger=consent.__name__):
        assert not consent.audience_change_allowed(_user(OPTED_OUT), action)
    (record,) = caplog.records
    assert record.levelno == logging.DEBUG
    assert _pseudonym(EMAIL) in record.getMessage()
    assert action.value in record.getMessage()
    assert EMAIL not in caplog.text


def test_the_skip_never_reaches_an_info_log(caplog):
    with caplog.at_level(logging.INFO, logger=consent.__name__):
        consent.log_opted_out_skip(EMAIL, "checkout_opened")
    assert caplog.records == []
