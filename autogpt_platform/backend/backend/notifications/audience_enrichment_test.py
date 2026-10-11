"""What GTM segments checkout openers on: country from the strongest source,
Germany and Austria caught by any signal and never released, and nothing
MailerLite holds made worse by a later or weaker write."""

from datetime import UTC, datetime

import pytest

from backend.data.notifications import SubscriberField
from backend.notifications import audience_enrichment as enrichment

CREATED = datetime(2026, 7, 14, 9, 0, tzinfo=UTC)
OPENED = datetime(2026, 7, 15, 9, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    "timezone, country",
    [
        ("Asia/Kolkata", "IN"),
        # Old names browsers still report, followed through tzdata's links.
        ("Asia/Calcutta", "IN"),
        ("Asia/Katmandu", "NP"),
        ("Europe/Kiev", "UA"),
        # tzdata links Stockholm to Berlin; the reported name must win.
        ("Europe/Stockholm", "SE"),
        ("Europe/Berlin", "DE"),
        ("Europe/Busingen", "DE"),
        ("Europe/Vienna", "AT"),
        ("America/Chicago", "US"),
        # A link that points across a border keeps its own country.
        ("Atlantic/Jan_Mayen", "SJ"),
        (" Asia/Kolkata ", "IN"),
        ("UTC", None),
        ("Etc/GMT+3", None),
        ("not-set", None),
        ("", None),
        (None, None),
        ("Mars/Olympus_Mons", None),
    ],
)
def test_a_timezone_names_its_country(timezone, country):
    assert enrichment.timezone_country(timezone) == country


def test_country_names_read_well_in_a_segment_filter():
    assert enrichment.country_name("GB") == "United Kingdom"
    assert enrichment.country_name("KR") == "South Korea"
    assert enrichment.country_name("US") == "United States"
    assert enrichment.country_name("IN") == "India"


@pytest.mark.parametrize(
    "value, code",
    [("us", "US"), (" DE ", "DE"), ("XX", None), ("", None), (None, None)],
)
def test_only_known_country_codes_are_kept(value, code):
    assert enrichment.country_code(value) == code


@pytest.mark.parametrize(
    "email, kind",
    [
        ("sam@gmail.com", "personal"),
        ("sam@GMAIL.com", "personal"),
        ("sam@yahoo.co.uk", "personal"),
        ("sam@qq.com", "personal"),
        ("sam@acme.io", "business"),
        ("sam@acme.co.uk", "business"),
        ("sam@mit.edu", "education"),
        ("sam@iitb.ac.in", "education"),
        ("sam@pucp.edu.pe", "education"),
    ],
)
def test_an_address_is_business_personal_or_education(email, kind):
    assert enrichment.email_type(email) == kind


@pytest.mark.parametrize(
    "email, de_at",
    [
        ("sam@firma.de", True),
        ("sam@uni.ac.at", True),
        ("sam@shop.co.at", True),
        ("sam@acme.com", False),
        ("sam@made.design", False),
    ],
)
def test_a_de_or_at_address_counts_whatever_sits_under_it(email, de_at):
    assert enrichment.email_points_at_de_at(email) is de_at


@pytest.mark.parametrize(
    "providers, method",
    [
        (["google"], "google"),
        (["credential", "google"], "google"),
        (["credential"], "email"),
        (["github"], "github"),
        ([], None),
    ],
)
def test_google_wins_the_signin_method(providers, method):
    assert enrichment.signin_method(providers) == method


def _fields(**kwargs) -> dict:
    defaults = dict(
        email="sam@acme.com",
        created_at=CREATED,
        opened_at=OPENED,
        signin_providers=["google"],
        timezone="Asia/Kolkata",
    )
    return enrichment.checkout_fields(**{**defaults, **kwargs})


def test_a_checkout_opener_gets_every_field():
    assert _fields() == {
        SubscriberField.STATUS: "signed",
        SubscriberField.SIGNUP: "2026-07-14",
        SubscriberField.CHECKOUT_OPENED: "2026-07-15",
        SubscriberField.EMAIL_TYPE: "business",
        SubscriberField.SIGNIN_METHOD: "google",
        SubscriberField.COUNTRY: "India",
        SubscriberField.COUNTRY_CODE: "IN",
        SubscriberField.COUNTRY_SOURCE: "timezone",
        SubscriberField.EXCLUDE_DE_AT: "no",
    }


@pytest.mark.parametrize(
    "sources, country, source",
    [
        (dict(stripe_country="US", ip_country="GB"), "US", "stripe"),
        (dict(ip_country="GB"), "GB", "ip"),
        (dict(stripe_country="??", ip_country="GB"), "GB", "ip"),
        ({}, "IN", "timezone"),
    ],
)
def test_country_comes_from_the_strongest_source(sources, country, source):
    fields = _fields(**sources)
    assert fields[SubscriberField.COUNTRY_CODE] == country
    assert fields[SubscriberField.COUNTRY_SOURCE] == source


def test_without_any_country_signal_the_country_is_left_alone():
    fields = _fields(timezone="UTC", signin_providers=[])
    for field in (
        SubscriberField.COUNTRY,
        SubscriberField.COUNTRY_CODE,
        SubscriberField.COUNTRY_SOURCE,
        SubscriberField.SIGNIN_METHOD,
    ):
        assert field not in fields


@pytest.mark.parametrize(
    "signals",
    [
        dict(timezone="Europe/Berlin"),
        dict(timezone="Europe/Vienna"),
        dict(ip_country="AT"),
        # The billing address is German even though the IP and timezone say UK.
        dict(stripe_country="DE", ip_country="GB", timezone="Europe/London"),
        # A weaker signal still excludes when a stronger one names elsewhere.
        dict(stripe_country="US", timezone="Europe/Berlin"),
        dict(email="sam@firma.de", timezone="America/Chicago"),
    ],
)
def test_any_de_or_at_signal_excludes(signals):
    assert _fields(**signals)[SubscriberField.EXCLUDE_DE_AT] == "yes"


def _held(**fields) -> dict:
    return {SubscriberField[k.upper()].value: v for k, v in fields.items()}


def test_a_status_mailerlite_holds_is_kept_from_signed():
    merged = enrichment.merge_with_held(_fields(), _held(status="subscribed"))
    assert SubscriberField.STATUS not in merged


def test_a_known_status_replaces_the_held_one_when_asked():
    merged = enrichment.merge_with_held(
        _fields(status="in_trial"),
        _held(status="signed"),
        keep_held_status=False,
    )
    assert merged[SubscriberField.STATUS] == "in_trial"


@pytest.mark.parametrize(
    "held, kept",
    [("2026-07-01", True), ("2026-07-15", True), ("2026-07-20T10:00:00", False)],
)
def test_the_first_checkout_date_is_kept(held, kept):
    merged = enrichment.merge_with_held(_fields(), _held(checkout_opened=held))
    assert (SubscriberField.CHECKOUT_OPENED not in merged) is kept


def test_exclude_de_at_is_never_released():
    merged = enrichment.merge_with_held(_fields(), _held(exclude_de_at="yes"))
    assert SubscriberField.EXCLUDE_DE_AT not in merged


@pytest.mark.parametrize(
    "held",
    [
        # Ours, from Stripe, written before the exclusion ever was.
        {"country": "Germany", "country_code": "DE", "country_source": "stripe"},
        {"country_code": "at"},
        # Typed by hand into MailerLite's built-in field.
        {"country": "Deutschland"},
        {"country": " Austria "},
    ],
)
def test_a_held_german_or_austrian_country_excludes(held):
    """The live guess (India by timezone) says no, but MailerLite already
    holds a German or Austrian country: that is a signal, so it excludes."""
    merged = enrichment.merge_with_held(_fields(), held)
    assert merged[SubscriberField.EXCLUDE_DE_AT] == "yes"


def test_a_held_country_elsewhere_does_not_exclude():
    merged = enrichment.merge_with_held(_fields(), {"country": "United States"})
    assert merged[SubscriberField.EXCLUDE_DE_AT] == "no"


def test_a_weaker_country_never_replaces_a_stronger_one():
    held = _held(country="Germany", country_code="DE", country_source="stripe")
    merged = enrichment.merge_with_held(_fields(ip_country="GB"), held)
    for field in (
        SubscriberField.COUNTRY,
        SubscriberField.COUNTRY_CODE,
        SubscriberField.COUNTRY_SOURCE,
    ):
        assert field not in merged


def test_a_stronger_country_replaces_a_weaker_one():
    held = _held(country="India", country_code="IN", country_source="timezone")
    merged = enrichment.merge_with_held(_fields(stripe_country="US"), held)
    assert merged[SubscriberField.COUNTRY_CODE] == "US"


def test_a_hand_typed_country_is_replaced():
    merged = enrichment.merge_with_held(_fields(), {"country": "USA"})
    assert merged[SubscriberField.COUNTRY] == "India"
