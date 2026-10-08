"""The notification service puts a checkout opener in the checkout openers
group with the fields GTM segments on, and a later or weaker event never
makes MailerLite's copy worse."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.data.notifications import AudienceAction
from backend.notifications import mailerlite, subscriber_fields
from backend.notifications.audience_enrichment import checkout_fields
from backend.notifications.notifications import NotificationManager

EMAIL = "sam@example.com"
CREATED = datetime(2026, 7, 14, 9, 0, tzinfo=UTC)


@pytest.fixture
def mailerlite_configured(monkeypatch):
    fake = SimpleNamespace(
        config=SimpleNamespace(
            mailerlite_checkout_group_id="grp_checkout",
            mailerlite_changelog_group_id="grp_changelog",
            mailerlite_trial_group_id="",
        ),
        secrets=SimpleNamespace(mailerlite_api_token="token"),
    )
    monkeypatch.setattr(mailerlite, "settings", fake)
    monkeypatch.setattr(mailerlite, "_fields_ready", False)
    monkeypatch.setattr(mailerlite, "_checkout_off_logged", False)
    return fake


def _response(status: int, body: dict | None = None) -> MagicMock:
    response = MagicMock(status=status)
    response.json.return_value = body or {}
    return response


class _FakeMailerLite:
    """Subscribers as MailerLite holds them: an upsert merges its fields and
    groups into whatever is already there."""

    def __init__(self, held: dict[str, dict] | None = None):
        self.subscribers = held or {}
        self.groups: dict[str, set[str]] = {}
        self.posts: list[dict] = []

    async def get(self, url: str, **_) -> MagicMock:
        if "/fields" in url:
            return _response(
                200,
                {
                    "data": [
                        {"key": f.value, "type": t}
                        for f, t in mailerlite.FIELD_TYPES.items()
                    ],
                    "meta": {"last_page": 1},
                },
            )
        held = self.subscribers.get(url.rsplit("/", 1)[-1])
        if held is None:
            return _response(404)
        return _response(200, {"data": {"id": "ml_1", "fields": dict(held)}})

    async def post(self, url: str, json: dict, **_) -> MagicMock:
        self.posts.append(json)
        held = self.subscribers.setdefault(json["email"], {})
        held.update(json.get("fields") or {})
        self.groups.setdefault(json["email"], set()).update(json.get("groups") or [])
        return _response(200)


def _event(**kwargs):
    defaults = dict(
        email=EMAIL,
        created_at=CREATED,
        opened_at=datetime(2026, 7, 15, tzinfo=UTC),
        signin_providers=["google"],
        timezone="Asia/Kolkata",
    )
    fields = checkout_fields(**{**defaults, **kwargs})
    return subscriber_fields.audience_event(
        AudienceAction.CHECKOUT_OPENED, EMAIL, "user-1", fields
    )


async def _consume(*events) -> None:
    for event in events:
        assert await NotificationManager._process_audience_change(
            MagicMock(), event.model_dump_json()
        )


# ── the handler ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_checkout_opener_joins_the_group_with_every_field(
    mailerlite_configured,
):
    ml = _FakeMailerLite()
    with patch.object(mailerlite, "_client", return_value=ml):
        await _consume(_event(ip_country="IN"))
    assert ml.groups[EMAIL] == {"grp_checkout"}
    assert ml.subscribers[EMAIL] == {
        "subscription_status": "signed",
        "signup_date": "2026-07-14",
        "checkout_opened_date": "2026-07-15",
        "email_type": "business",
        "signin_method": "google",
        "country": "India",
        "country_code": "IN",
        "country_source": "ip",
        "exclude_de_at": "no",
    }


@pytest.mark.asyncio
async def test_without_a_checkout_group_openers_are_dropped_not_dead_lettered(
    mailerlite_configured, caplog
):
    """The group is the switch: off, nothing is written and the message is
    acknowledged, said once; the backfill brings in anyone missed."""
    mailerlite_configured.config.mailerlite_checkout_group_id = ""
    ml = _FakeMailerLite()
    ml.post = AsyncMock(side_effect=ml.post)
    with (
        patch.object(mailerlite, "_client", return_value=ml),
        caplog.at_level("INFO"),
    ):
        await _consume(_event(), _event())
    ml.post.assert_not_awaited()
    assert caplog.text.count("MAILERLITE_CHECKOUT_GROUP_ID is not set") == 1


@pytest.mark.asyncio
async def test_a_later_checkout_never_makes_mailerlite_worse(mailerlite_configured):
    """A paying customer opening checkout again, from a UK IP in a UK
    timezone, keeps their status, their first open date, their billing
    country and their exclusion."""
    held = {
        "subscription_status": "subscribed",
        "checkout_opened_date": "2026-07-01",
        "country": "Germany",
        "country_code": "DE",
        "country_source": "stripe",
        "exclude_de_at": "yes",
    }
    ml = _FakeMailerLite({EMAIL: dict(held)})
    with patch.object(mailerlite, "_client", return_value=ml):
        await _consume(_event(ip_country="GB", timezone="Europe/London"))
    for key, value in held.items():
        assert ml.subscribers[EMAIL][key] == value
    assert ml.groups[EMAIL] == {"grp_checkout"}


@pytest.mark.asyncio
async def test_the_billing_country_upgrades_a_timezone_guess(mailerlite_configured):
    ml = _FakeMailerLite(
        {
            EMAIL: {
                "country_code": "IN",
                "country_source": "timezone",
                "exclude_de_at": "no",
            }
        }
    )
    with patch.object(mailerlite, "_client", return_value=ml):
        await _consume(_event(stripe_country="AT"))
    assert ml.subscribers[EMAIL]["country"] == "Austria"
    assert ml.subscribers[EMAIL]["country_source"] == "stripe"
    assert ml.subscribers[EMAIL]["exclude_de_at"] == "yes"


@pytest.mark.asyncio
async def test_every_audience_action_has_a_handler(monkeypatch):
    """An action the consumer cannot route would raise on every delivery."""
    monkeypatch.setattr(mailerlite, "configured", lambda: True)
    handlers = {}
    for name in (
        "enroll_in_onboarding",
        "add_to_changelog",
        "remove_from_changelog",
        "add_to_trial",
        "remove_from_trial",
        "update_fields",
        "record_signup",
        "record_checkout_opened",
        "unsubscribe",
    ):
        handlers[name] = AsyncMock()
        monkeypatch.setattr(mailerlite, name, handlers[name])
    for action in AudienceAction:
        event = subscriber_fields.audience_event(action, EMAIL, "user-1")
        await _consume(event)
    assert all(h.await_count == 1 for h in handlers.values())
