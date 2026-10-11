"""Someone who refused marketing never enters MailerLite, and the skip is
logged at debug level by pseudonym, never by address."""

import logging
import pickle
from datetime import UTC, datetime

import pytest

from backend.data.model import User
from backend.data.notifications import AudienceAction
from backend.data.user import BillingEmailRecipient
from backend.notifications import consent
from backend.notifications.mailerlite import pseudonym

EMAIL = "sam@example.com"
OPTED_OUT = datetime(2026, 10, 2, 12, 0, tzinfo=UTC)


def _user(opted_out_at: datetime | None) -> BillingEmailRecipient:
    return BillingEmailRecipient(
        id="user-1", email=EMAIL, marketing_opt_out_at=opted_out_at
    )


def test_marketing_is_allowed_until_they_opt_out():
    assert consent.marketing_allowed(_user(None))
    assert not consent.marketing_allowed(_user(OPTED_OUT))


CONSENT_FIELDS = (
    "terms_accepted_at",
    "terms_version",
    "marketing_opt_out_at",
    "marketing_opt_out_source",
)


def _cached_before_consent() -> User:
    """A `User` as the shared cache hands it back when the previous release
    pickled it, before the consent fields existed."""
    user = User(id="user-1", email=EMAIL, created_at=OPTED_OUT, updated_at=OPTED_OUT)
    for field in CONSENT_FIELDS:
        del user.__dict__[field]
    return pickle.loads(pickle.dumps(user))


def test_a_user_cached_before_the_consent_fields_is_skipped_without_raising():
    """During a rolling deploy its consent is unknown, so it fails closed: the
    MailerLite change is skipped and the backfills catch it up later."""
    stale = _cached_before_consent()
    assert not set(CONSENT_FIELDS) & set(stale.__dict__)

    assert not consent.marketing_allowed(stale)
    assert not consent.audience_change_allowed(stale, AudienceAction.ADD_TRIAL)


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
    assert pseudonym(EMAIL) in record.getMessage()
    assert action.value in record.getMessage()
    assert EMAIL not in caplog.text


def test_the_skip_never_reaches_an_info_log(caplog):
    with caplog.at_level(logging.INFO, logger=consent.__name__):
        consent.log_opted_out_skip(EMAIL, "checkout_opened")
    assert caplog.records == []
