"""A checkout opener's MailerLite segmentation reaches the PostHog person with
the same values, and `$set`/`$set_once` keep MailerLite's merge rules without
reading the person first."""

from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import Mock

import pytest

from backend.data.notifications import SubscriberField
from backend.data.onboarding_role import OnboardingRole
from backend.notifications import posthog_audience
from backend.notifications.audience_enrichment import (
    checkout_fields,
    merge_with_held,
    role_fields,
)
from backend.notifications.posthog_audience import (
    CHECKOUT_OPENED_AT,
    MIRRORED,
    NOT_MIRRORED,
    PersonUpdate,
)
from backend.util import posthog_client

CREATED = datetime(2026, 7, 14, 9, 0, tzinfo=UTC)
OPENED = 1788305400
OPENED_ISO = "2026-09-01T23:30:00+00:00"
COUNTRY_KEYS = {"country", "country_code", "country_source"}
# Mirrored under MailerLite's own key and value; the open goes as a moment.
SHARED = MIRRORED - {SubscriberField.CHECKOUT_OPENED}


@pytest.fixture
def capture(monkeypatch: pytest.MonkeyPatch) -> Mock:
    client = Mock()
    monkeypatch.setattr(posthog_client, "get_posthog_client", lambda: client)
    return client.capture


def _fields(**overrides):
    return checkout_fields(
        **{
            "email": "sam@acme.io",
            "created_at": CREATED,
            "opened_at": OPENED,
            "signin_providers": ["credential"],
            "timezone": "America/Chicago",
            **overrides,
        }
    )


def _sent(update: PersonUpdate) -> dict[str, str]:
    return {**update.to_set_once, **update.to_set}


def _person(*updates: PersonUpdate) -> dict[str, str]:
    """What PostHog holds after `updates`, in order: `$set` replaces, and
    `$set_once` only fills in what the person doesn't hold yet."""
    person: dict[str, str] = {}
    for update in updates:
        for key, value in update.to_set_once.items():
            person.setdefault(key, value)
        person.update(update.to_set)
    return person


# ── every account ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "email, providers, expected",
    [
        (
            "sam@acme.io",
            ["credential"],
            {"email_type": "business", "signin_method": "email"},
        ),
        (
            "sam@gmail.com",
            ["credential", "google"],
            {"email_type": "personal", "signin_method": "google"},
        ),
        ("sam@uni.edu", [], {"email_type": "education"}),
    ],
)
def test_an_account_gets_its_email_type_and_sign_in_method(email, providers, expected):
    update = posthog_audience.account_update(email, providers)
    assert update.to_set == expected
    assert update.to_set_once == {}


@pytest.mark.parametrize(
    "signup_method, expected", [("email", "email"), ("google", "google"), (None, None)]
)
def test_a_new_account_is_sent_from_its_signup_method(
    capture: Mock, signup_method, expected
):
    posthog_audience.set_account_properties("user-1", "sam@acme.io", signup_method)
    kwargs = capture.call_args.kwargs
    assert kwargs["distinct_id"] == "user-1"
    assert kwargs["event"] == "$set"
    person = kwargs["properties"]["$set"]
    assert person["email_type"] == "business"
    assert person.get("signin_method") == expected


# ── checkout openers ─────────────────────────────────────────────────────────


def test_every_mailerlite_field_is_mirrored_or_left_to_another_sender():
    """A field added to SubscriberField fails this until it is placed, so it
    can't skip PostHog unnoticed."""
    assert MIRRORED | NOT_MIRRORED == set(SubscriberField)
    assert not MIRRORED & NOT_MIRRORED


def test_an_opener_gets_every_mirrored_field_and_nothing_else():
    """With every signal a checkout can carry, and the role on top: the
    mirrored fields arrive with MailerLite's values under its keys, the open as
    a moment rather than MailerLite's day, and nothing another sender owns."""
    fields = {
        **_fields(stripe_country="AT", ip_country="US", signin_providers=["google"]),
        **role_fields(OnboardingRole(choice="Other", other="Dentist")),
    }
    person = _person(posthog_audience.checkout_update(fields, OPENED))
    for field in SHARED:
        assert person[field.value] == fields[field]
    assert person[CHECKOUT_OPENED_AT] == OPENED_ISO
    assert person.keys() == {f.value for f in SHARED} | {CHECKOUT_OPENED_AT}


@pytest.mark.parametrize(
    "countries, source, replaces",
    [
        (dict(stripe_country="DE", ip_country="US"), "stripe", True),
        (dict(ip_country="DE"), "ip", False),
        ({}, "timezone", False),
    ],
)
def test_only_a_billing_country_replaces_the_one_held(countries, source, replaces):
    update = posthog_audience.checkout_update(_fields(**countries), OPENED)
    held, other = (
        (update.to_set, update.to_set_once)
        if replaces
        else (update.to_set_once, update.to_set)
    )
    assert held["country_source"] == source
    assert COUNTRY_KEYS <= held.keys()
    assert not COUNTRY_KEYS & other.keys()


def test_an_opener_with_no_country_signal_gets_no_country():
    sent = _sent(posthog_audience.checkout_update(_fields(timezone="UTC"), OPENED))
    assert not COUNTRY_KEYS & sent.keys()


@pytest.mark.parametrize(
    "countries, value, replaces",
    [(dict(ip_country="AT"), "yes", True), (dict(ip_country="US"), "no", False)],
)
def test_exclude_de_at_yes_replaces_and_no_only_fills_in(countries, value, replaces):
    update = posthog_audience.checkout_update(_fields(**countries), OPENED)
    held = update.to_set if replaces else update.to_set_once
    assert held["exclude_de_at"] == value


@pytest.mark.parametrize(
    "opened_at",
    [
        OPENED,
        datetime(2026, 9, 1, 23, 30),
        datetime(2026, 9, 2, 1, 30, tzinfo=timezone(timedelta(hours=2))),
    ],
    ids=["stripe-timestamp", "naive-utc", "other-offset"],
)
def test_the_open_is_an_iso_utc_moment_that_only_fills_in(opened_at):
    update = posthog_audience.checkout_update(_fields(), opened_at)
    assert update.to_set_once[CHECKOUT_OPENED_AT] == OPENED_ISO
    assert CHECKOUT_OPENED_AT not in update.to_set


def test_an_open_without_a_timestamp_of_its_own_is_now():
    before = datetime.now(UTC).replace(microsecond=0)
    update = posthog_audience.checkout_update(_fields(opened_at=None), None)
    opened = datetime.fromisoformat(update.to_set_once[CHECKOUT_OPENED_AT])
    assert before <= opened <= datetime.now(UTC)


def test_the_backfills_first_open_replaces_the_one_held():
    """Stripe holds every checkout, so its first is the first open, even
    where the live code saw a later one first."""
    live = posthog_audience.checkout_update(_fields(), OPENED + 86400)
    backfill = posthog_audience.checkout_update(_fields(), OPENED, first_open=True)
    assert backfill.to_set[CHECKOUT_OPENED_AT] == OPENED_ISO
    assert _person(live, backfill)[CHECKOUT_OPENED_AT] == OPENED_ISO


def test_a_run_of_checkouts_ends_where_mailerlite_does():
    """Opened from Austria, completed billed to the US, opened again from
    France: MailerLite keeps the billing country, the Austrian signal and the
    first open, and so does PostHog, without reading the person."""
    checkouts = [
        (OPENED, dict(ip_country="AT")),
        (OPENED + 600, dict(stripe_country="US")),
        (OPENED + 86400, dict(ip_country="FR")),
    ]
    held: dict[str, str | None] = {}
    updates = []
    for opened_at, countries in checkouts:
        fields = _fields(opened_at=opened_at, **countries)
        merged = merge_with_held(fields, held)
        held.update({field.value: value for field, value in merged.items()})
        updates.append(posthog_audience.checkout_update(fields, opened_at))
    person = _person(*updates)
    for field in SHARED:
        assert person[field.value] == held[field.value]
    assert person["country_source"] == "stripe"
    assert person["exclude_de_at"] == "yes"
    assert person[CHECKOUT_OPENED_AT] == OPENED_ISO


def test_posthog_keeps_the_first_ip_country_where_mailerlite_moves_on():
    """The one documented difference: PostHog can't read what it holds, so a
    later IP country doesn't replace an earlier IP or timezone one."""
    first = posthog_audience.checkout_update(_fields(ip_country="US"), OPENED)
    later = posthog_audience.checkout_update(_fields(ip_country="CA"), OPENED + 60)
    assert _person(first, later)["country_code"] == "US"


# ── sending ──────────────────────────────────────────────────────────────────


def test_an_update_is_one_set_event(capture: Mock):
    assert posthog_audience.send(
        "user-1",
        PersonUpdate(
            to_set={"email_type": "business"}, to_set_once={"exclude_de_at": "no"}
        ),
    )
    capture.assert_called_once()
    kwargs = capture.call_args.kwargs
    assert kwargs["distinct_id"] == "user-1"
    assert kwargs["event"] == "$set"
    assert kwargs["properties"]["$set"] == {"email_type": "business"}
    assert kwargs["properties"]["$set_once"] == {"exclude_de_at": "no"}


def test_only_the_parts_with_values_are_sent(capture: Mock):
    posthog_audience.send("user-1", PersonUpdate(to_set={"email_type": "business"}))
    assert "$set_once" not in capture.call_args.kwargs["properties"]


def test_an_empty_update_sends_nothing_and_counts_as_done(capture: Mock):
    assert posthog_audience.send("user-1", PersonUpdate())
    capture.assert_not_called()


@pytest.mark.parametrize(
    "returned, side_effect",
    [(None, None), (None, RuntimeError("posthog down"))],
    ids=["dropped", "raised"],
)
def test_an_update_the_client_did_not_queue_is_reported_not_raised(
    capture: Mock, returned, side_effect
):
    capture.return_value = returned
    capture.side_effect = side_effect
    update = PersonUpdate(to_set={"email_type": "business"})
    assert posthog_audience.send("user-1", update) is False


def test_an_update_goes_to_the_client_it_is_given(capture: Mock):
    own = Mock()
    update = PersonUpdate(to_set={"email_type": "business"})
    assert posthog_audience.send("user-1", update, client=own)
    own.capture.assert_called_once()
    capture.assert_not_called()


def test_a_checkout_opener_is_sent_their_checkout_update(capture: Mock):
    fields = _fields(ip_country="AT")
    posthog_audience.set_checkout_properties("user-1", fields, OPENED)
    properties = capture.call_args.kwargs["properties"]
    expected = posthog_audience.checkout_update(fields, OPENED)
    assert properties["$set"] == expected.to_set
    assert properties["$set_once"] == expected.to_set_once


@pytest.mark.parametrize(
    "broken, call",
    [
        (
            "account_update",
            lambda: posthog_audience.set_account_properties(
                "user-1", "sam@acme.io", "email"
            ),
        ),
        (
            "checkout_update",
            lambda: posthog_audience.set_checkout_properties(
                "user-1", _fields(), OPENED
            ),
        ),
    ],
    ids=["signup", "checkout"],
)
def test_a_failure_is_logged_and_never_reaches_the_caller(
    capture: Mock, monkeypatch, caplog, broken, call
):
    """Analytics must not fail a signup or cost an opener their MailerLite
    change."""
    monkeypatch.setattr(posthog_audience, broken, Mock(side_effect=RuntimeError()))
    call()
    capture.assert_not_called()
    assert "user-1" in caplog.text
