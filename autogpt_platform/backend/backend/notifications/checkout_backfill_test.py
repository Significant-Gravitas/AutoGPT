"""The backfill puts existing checkout openers where the live event would
have, writes only the difference, never makes MailerLite's copy worse, and
writes one subscriber at a time so MailerLite stores what it accepts."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.data.notifications import SubscriberField
from backend.notifications import checkout_backfill, mailerlite
from backend.notifications.mailerlite_backfill import Subscription
from backend.notifications.mailerlite_field_backfill import Person

OPENED = int(datetime(2026, 7, 15, 9, 0, tzinfo=UTC).timestamp())
OPTED_OUT = datetime(2026, 10, 2, tzinfo=UTC)


def _opener(email="sam@acme.com", subscriptions=None, **kwargs):
    return checkout_backfill.Opener(
        person=Person(
            user_id="user-1",
            email=email,
            created_at=datetime(2026, 7, 14, tzinfo=UTC),
            subscriptions=subscriptions or [],
            stripe_customer_id="cus_1",
            timezone=kwargs.pop("timezone", "Asia/Kolkata"),
            marketing_opt_out_at=kwargs.pop("opted_out_at", None),
        ),
        opened_at=OPENED,
        signin_providers=["google"],
        **kwargs,
    )


def _held(opener) -> dict:
    """Exactly what a full write of this opener leaves in MailerLite."""
    return {
        field.value: value for field, value in checkout_backfill.wanted(opener).items()
    }


def test_a_new_opener_is_created_in_the_group_with_every_field():
    opener = _opener()
    plan = checkout_backfill.plan([opener], current={}, members={})
    (change,) = plan.changes
    assert change.new and change.joins
    assert change.fields[SubscriberField.COUNTRY_CODE] == "IN"
    assert change.fields[SubscriberField.COUNTRY_SOURCE] == "timezone"
    assert change.fields[SubscriberField.CHECKOUT_OPENED] == "2026-07-15"
    assert change.fields[SubscriberField.STATUS] == "signed"


def test_an_opener_already_in_place_is_left_alone():
    opener = _opener()
    plan = checkout_backfill.plan(
        [opener], current={"sam@acme.com": _held(opener)}, members={"sam@acme.com": "1"}
    )
    assert plan.changes == []
    assert plan.openers == 1


def test_an_existing_subscriber_only_joins_the_group():
    opener = _opener()
    plan = checkout_backfill.plan(
        [opener], current={"sam@acme.com": _held(opener)}, members={}
    )
    (change,) = plan.changes
    assert change.joins and not change.new and change.fields == {}


def test_the_status_comes_from_stripe_not_the_signed_default():
    opener = _opener(subscriptions=[Subscription(status="active", start_date=OPENED)])
    held = {**_held(_opener()), "subscription_status": "signed"}
    plan = checkout_backfill.plan(
        [opener], current={"sam@acme.com": held}, members={"sam@acme.com": "1"}
    )
    (change,) = plan.changes
    assert change.fields[SubscriberField.STATUS] == "subscribed"


def test_a_billing_country_is_used_and_never_downgraded_to_the_timezone():
    with_billing = _opener(stripe_country="DE")
    plan = checkout_backfill.plan([with_billing], current={}, members={})
    assert plan.changes[0].fields[SubscriberField.COUNTRY_CODE] == "DE"
    assert plan.changes[0].fields[SubscriberField.EXCLUDE_DE_AT] == "yes"
    assert plan.exclude_de_at == 1

    held = {
        **_held(_opener()),
        "country": "Germany",
        "country_code": "DE",
        "country_source": "stripe",
        "exclude_de_at": "yes",
    }
    plan = checkout_backfill.plan(
        [_opener()], current={"sam@acme.com": held}, members={"sam@acme.com": "1"}
    )
    assert plan.changes == []
    assert plan.countries == {"DE": 1}


def test_an_address_mailerlite_would_refuse_is_skipped():
    plan = checkout_backfill.plan([_opener(email="sam@site.test")], {}, {})
    assert plan.changes == [] and plan.invalid == 1


@pytest.mark.asyncio
async def test_each_opener_is_one_paced_upsert_into_the_group(monkeypatch, caplog):
    sleep = AsyncMock()
    monkeypatch.setattr(checkout_backfill.asyncio, "sleep", sleep)
    plan = checkout_backfill.plan(
        [_opener(email="a@acme.com"), _opener(email="b@acme.com")], {}, {}
    )
    refusal = {"message": "The given data was invalid.", "errors": {"email": ["no"]}}
    client = MagicMock(
        post=AsyncMock(side_effect=[MagicMock(status=200), _refused(refusal)])
    )
    with (
        patch.object(checkout_backfill, "_client", return_value=client),
        patch.object(
            checkout_backfill, "_find_subscriber", AsyncMock(return_value=None)
        ),
        caplog.at_level("WARNING"),
    ):
        result = await checkout_backfill.apply(plan.changes, "grp_checkout")
    assert result == (1, 1, 0)
    first = client.post.await_args_list[0]
    assert first.args[0].endswith("/subscribers")
    assert first.kwargs["json"]["groups"] == ["grp_checkout"]
    assert first.kwargs["json"]["fields"]["country_code"] == "IN"
    sleep.assert_awaited_once_with(checkout_backfill.WRITE_INTERVAL_SECONDS)
    assert "The given data was invalid." in caplog.text
    assert "b@acme.com" not in caplog.text


@pytest.mark.asyncio
async def test_a_live_write_made_during_the_run_is_not_overwritten(monkeypatch):
    """Planned from an early snapshot as Indian by timezone; while the run
    went on, the live event recorded a German billing address. The write is
    merged against a fresh read, so Germany and the exclusion stay."""
    monkeypatch.setattr(checkout_backfill.asyncio, "sleep", AsyncMock())
    opener = _opener()
    plan = checkout_backfill.plan([opener], current={}, members={})
    fresh = {
        **_held(opener),
        "country": "Germany",
        "country_code": "DE",
        "country_source": "stripe",
        "exclude_de_at": "yes",
    }
    client = MagicMock(post=AsyncMock(return_value=MagicMock(status=200)))
    with (
        patch.object(checkout_backfill, "_client", return_value=client),
        patch.object(
            checkout_backfill,
            "_find_subscriber",
            AsyncMock(return_value={"fields": fresh}),
        ),
    ):
        result = await checkout_backfill.apply(plan.changes, "grp_checkout")
    assert result == (1, 0, 0)
    body = client.post.await_args.kwargs["json"]
    assert body["groups"] == ["grp_checkout"]
    assert "fields" not in body


@pytest.mark.asyncio
async def test_someone_already_up_to_date_is_skipped(monkeypatch):
    monkeypatch.setattr(checkout_backfill.asyncio, "sleep", AsyncMock())
    opener = _opener()
    plan = checkout_backfill.plan(
        [opener], current={"sam@acme.com": {}}, members={"sam@acme.com": "1"}
    )
    client = MagicMock(post=AsyncMock())
    with (
        patch.object(checkout_backfill, "_client", return_value=client),
        patch.object(
            checkout_backfill,
            "_find_subscriber",
            AsyncMock(return_value={"fields": _held(opener)}),
        ),
    ):
        result = await checkout_backfill.apply(plan.changes, "grp_checkout")
    assert result == (0, 0, 1)
    client.post.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failed_reread_is_counted_and_retried_later(monkeypatch, caplog):
    monkeypatch.setattr(checkout_backfill.asyncio, "sleep", AsyncMock())
    plan = checkout_backfill.plan([_opener()], current={}, members={})
    client = MagicMock(post=AsyncMock())
    with (
        patch.object(checkout_backfill, "_client", return_value=client),
        patch.object(
            checkout_backfill,
            "_find_subscriber",
            AsyncMock(side_effect=RuntimeError("MailerLite down")),
        ),
        caplog.at_level("WARNING"),
    ):
        result = await checkout_backfill.apply(plan.changes, "grp_checkout")
    assert result == (0, 1, 0)
    client.post.assert_not_awaited()
    assert "sam@acme.com" not in caplog.text


def _refused(body: dict) -> MagicMock:
    response = MagicMock(status=422)
    response.json.return_value = body
    return response


def test_the_built_in_country_is_written_but_never_created():
    assert SubscriberField.COUNTRY not in mailerlite.FIELD_TYPES
    assert mailerlite.field_type(SubscriberField.COUNTRY) == "text"
    assert mailerlite.FIELD_TYPES[SubscriberField.CHECKOUT_OPENED] == "date"


@pytest.mark.asyncio
async def test_a_failed_write_is_counted_and_the_run_goes_on(monkeypatch, caplog):
    """A network error on one person's write must not end a run of
    thousands: it is counted, logged without the address, and the next
    person is still written."""
    monkeypatch.setattr(checkout_backfill.asyncio, "sleep", AsyncMock())
    plan = checkout_backfill.plan(
        [_opener(email="a@acme.com"), _opener(email="b@acme.com")], {}, {}
    )
    client = MagicMock(
        post=AsyncMock(
            side_effect=[RuntimeError("connection reset"), MagicMock(status=200)]
        )
    )
    with (
        patch.object(checkout_backfill, "_client", return_value=client),
        patch.object(
            checkout_backfill, "_find_subscriber", AsyncMock(return_value=None)
        ),
        caplog.at_level("WARNING"),
    ):
        result = await checkout_backfill.apply(plan.changes, "grp_checkout")
    assert result == (1, 1, 0)
    assert client.post.await_count == 2
    assert "a@acme.com" not in caplog.text


ACTIVE = [Subscription(id="sub_1", status="active", start_date=OPENED)]


async def _apply_with(monkeypatch, plan, *, held, refresh):
    monkeypatch.setattr(checkout_backfill.asyncio, "sleep", AsyncMock())
    client = MagicMock(post=AsyncMock(return_value=MagicMock(status=200)))
    with (
        patch.object(checkout_backfill, "_client", return_value=client),
        patch.object(
            checkout_backfill, "_find_subscriber", AsyncMock(return_value=held)
        ),
    ):
        result = await checkout_backfill.apply(
            plan.changes, "grp_checkout", refresh=refresh
        )
    return result, client


@pytest.mark.asyncio
async def test_the_status_written_is_stripes_now_not_the_plans(monkeypatch):
    """Planned hours earlier as a signup; the trial converted since. The
    write re-reads Stripe, so it says subscribed, not the snapshot's signed."""
    plan = checkout_backfill.plan([_opener()], current={}, members={})
    assert plan.changes[0].fields[SubscriberField.STATUS] == "signed"
    refresh = AsyncMock(return_value=ACTIVE)
    result, client = await _apply_with(monkeypatch, plan, held=None, refresh=refresh)
    assert result == (1, 0, 0)
    refresh.assert_awaited_once_with("cus_1")
    fields = client.post.await_args.kwargs["json"]["fields"]
    assert fields["subscription_status"] == "subscribed"
    assert fields["subscription_started_date"] == "2026-07-15"


@pytest.mark.asyncio
async def test_a_newer_status_the_live_event_wrote_is_not_overwritten(monkeypatch):
    """The live lifecycle event already wrote subscribed during the run; the
    snapshot still said signed. Stripe now agrees with the live write, so
    nothing is left to write."""
    # Not in MailerLite when planned (already in the group by the time it
    # runs, joined by the live event), so the plan wants the snapshot's signed.
    plan = checkout_backfill.plan(
        [_opener()], current={}, members={"sam@acme.com": "1"}
    )
    assert plan.changes[0].fields[SubscriberField.STATUS] == "signed"
    live = _held(_opener(subscriptions=ACTIVE))
    result, client = await _apply_with(
        monkeypatch,
        plan,
        held={"fields": live},
        refresh=AsyncMock(return_value=ACTIVE),
    )
    assert result == (0, 0, 1)
    client.post.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failed_stripe_refresh_is_counted_and_retried_later(monkeypatch):
    plan = checkout_backfill.plan([_opener()], current={}, members={})
    result, client = await _apply_with(
        monkeypatch,
        plan,
        held=None,
        refresh=AsyncMock(side_effect=RuntimeError("Stripe down")),
    )
    assert result == (0, 1, 0)
    client.post.assert_not_awaited()


def _with_an_opted_out_opener() -> checkout_backfill.OpenerPlan:
    """Held by MailerLite but outside the group, and German: every reason to
    be written and tallied, had they not opted out."""
    opted_out = _opener(
        email="out@acme.com", stripe_country="DE", opted_out_at=OPTED_OUT
    )
    return checkout_backfill.plan(
        [opted_out, _opener()], current={"out@acme.com": {}}, members={}
    )


def test_an_opted_out_opener_is_counted_but_never_planned_or_tallied():
    plan = _with_an_opted_out_opener()
    assert [c.opener.person.email for c in plan.changes] == ["sam@acme.com"]
    assert plan.openers == 2
    assert plan.opted_out == 1
    assert plan.invalid == 0
    assert plan.countries == {"IN": 1}
    assert plan.exclude_de_at == 0
    assert sum(plan.email_types.values()) == 1


@pytest.mark.asyncio
async def test_an_opted_out_opener_is_never_written(monkeypatch):
    plan = _with_an_opted_out_opener()
    result, client = await _apply_with(monkeypatch, plan, held=None, refresh=None)
    assert result == (1, 0, 0)
    written = [c.kwargs["json"]["email"] for c in client.post.await_args_list]
    assert written == ["sam@acme.com"]
