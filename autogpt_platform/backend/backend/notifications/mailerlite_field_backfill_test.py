"""The field backfill gives every account the status and dates the live code
would have, writes only what MailerLite does not already hold, and so resumes
an interrupted run by simply running again."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.data.notifications import SubscriberField, SubscriptionStatus
from backend.notifications import mailerlite, mailerlite_backfill
from backend.notifications import mailerlite_field_backfill as backfill
from backend.notifications.mailerlite_backfill import Subscription
from backend.notifications.mailerlite_field_backfill import Person

S = SubscriptionStatus
F = SubscriberField
# 1 Aug, 1 Sep, 10 Sep and 20 Sep 2026, noon UTC.
AUG, SEP, SEP10, SEP20 = 1785585600, 1788264000, 1789041600, 1789905600
CREATED = datetime(2026, 6, 1, 9, 0, tzinfo=UTC)


def _day(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, UTC).strftime("%Y-%m-%d")


def _paid(status: str, **over) -> Subscription:
    return Subscription(status=status, start_date=AUG, **over)


def _trial(status: str, **over) -> Subscription:
    return Subscription(
        status=status, from_trial=True, trial_start=AUG, trial_end=SEP, **over
    )


def _person(email: str = "a@x.io", *subscriptions: Subscription) -> Person:
    return Person(
        user_id=f"u-{email}",
        email=email,
        created_at=CREATED,
        subscriptions=list(subscriptions),
    )


@pytest.mark.parametrize(
    "subscriptions, status",
    [
        ([], S.SIGNED),
        # Checkout never finished: nothing started.
        ([_paid("incomplete_expired")], S.SIGNED),
        ([_paid("active")], S.SUBSCRIBED),
        ([_paid("past_due")], S.SUBSCRIBED),
        ([_paid("active", cancel_at_period_end=True)], S.SUBSCRIPTION_CANCELED),
        ([_paid("canceled")], S.SUBSCRIPTION_ENDED),
        ([_paid("unpaid")], S.SUBSCRIPTION_ENDED),
        ([_trial("trialing")], S.IN_TRIAL),
        ([_trial("trialing", cancel_at_period_end=True)], S.TRIAL_CANCELED),
        # Its first payment never cleared.
        ([_trial("past_due")], S.TRIAL_CANCELED),
        ([_trial("canceled")], S.TRIAL_CANCELED),
        ([_trial("active")], S.SUBSCRIBED),
        ([_trial("canceled", converted=True)], S.SUBSCRIPTION_ENDED),
        # A paid subscription outranks a trial, and a live one an ended one.
        ([_paid("canceled"), _trial("trialing")], S.IN_TRIAL),
        ([_trial("canceled"), _paid("active")], S.SUBSCRIBED),
    ],
)
def test_standing(subscriptions, status):
    assert backfill.standing(subscriptions)[0] is status


def test_every_field_is_set_so_nothing_stale_survives():
    status, fields = backfill.desired(_person("a@x.io"))
    assert status is S.SIGNED
    assert fields == {
        F.STATUS: "signed",
        F.SIGNUP: "2026-06-01",
        F.TRIAL_STARTED: None,
        F.SUBSCRIPTION_STARTED: None,
        F.SUBSCRIPTION_CANCELED: None,
        F.SUBSCRIPTION_ENDED: None,
    }


def test_the_dates_come_from_stripe():
    _, fields = backfill.desired(
        _person(
            "a@x.io",
            _trial(
                "canceled",
                converted=True,
                cancel_at_period_end=True,
                canceled_at=SEP10,
                ended_at=SEP20,
            ),
        )
    )
    assert fields[F.TRIAL_STARTED] == _day(AUG)
    # A converted trial's paid subscription began when the trial ended.
    assert fields[F.SUBSCRIPTION_STARTED] == _day(SEP)
    assert fields[F.SUBSCRIPTION_CANCELED] == _day(SEP10)
    assert fields[F.SUBSCRIPTION_ENDED] == _day(SEP20)


def test_a_missing_timestamp_is_left_empty_not_today():
    _, fields = backfill.desired(_person("a@x.io", Subscription(status="canceled")))
    assert fields[F.SUBSCRIPTION_ENDED] is None


def test_only_the_difference_is_written():
    held = {
        "a@x.io": {
            "subscription_status": "signed",
            # MailerLite may hand a date back with a time part.
            "signup_date": "2026-06-01 00:00:00",
            "trial_started_date": None,
            "subscription_started_date": "",
            "subscription_canceled_date": None,
            "subscription_ended_date": None,
        }
    }
    people = [_person("A@x.io"), _person("b@x.io", _paid("active"))]
    result = backfill.plan(people, held)
    assert [c.person.email for c in result.changes] == ["b@x.io"]
    [change] = result.changes
    assert change.new is True
    assert change.fields[F.STATUS] == "subscribed"
    assert result.statuses[S.SIGNED] == 1
    assert result.statuses[S.SUBSCRIBED] == 1


def test_a_changed_status_writes_just_that_field():
    held = {"a@x.io": {k.value: v for k, v in backfill.desired(_person())[1].items()}}
    held["a@x.io"]["subscription_status"] = "in_trial"
    [change] = backfill.plan([_person()], held).changes
    assert change.fields == {F.STATUS: "signed"}
    assert change.new is False


def test_an_address_mailerlite_would_refuse_is_counted_not_sent():
    result = backfill.plan([_person("sam@site.test")], {})
    assert result.invalid == 1
    assert result.changes == []


def test_a_second_run_finds_nothing_to_do():
    people = [_person("a@x.io"), _person("b@x.io", _trial("trialing"))]
    first = backfill.plan(people, {})
    held = {
        c.person.email: {k.value: v for k, v in c.fields.items()} for c in first.changes
    }
    assert backfill.plan(people, held).changes == []


@pytest.fixture
def configured(monkeypatch):
    fake = SimpleNamespace(secrets=SimpleNamespace(mailerlite_api_token="token"))
    monkeypatch.setattr(mailerlite, "settings", fake)
    sleep = AsyncMock()
    monkeypatch.setattr(backfill.asyncio, "sleep", sleep)
    return sleep


def _response(status: int, body: dict) -> MagicMock:
    response = MagicMock(status=status)
    response.json.return_value = body
    return response


@pytest.mark.asyncio
async def test_apply_batches_upserts_at_the_import_pace(configured, monkeypatch):
    client = MagicMock()
    client.post = AsyncMock(
        side_effect=lambda url, **kw: _response(
            200,
            {"responses": [{"code": 200} for _ in kw["json"]["requests"]]},
        )
    )
    monkeypatch.setattr(mailerlite_backfill, "_client", lambda: client)
    changes = backfill.plan([_person(f"p{i}@x.io") for i in range(120)], {}).changes
    progress = []

    ok, failed = await backfill.apply(changes, lambda d, t: progress.append(d))

    batches = [c.kwargs["json"]["requests"] for c in client.post.await_args_list]
    assert [len(b) for b in batches] == [50, 50, 20]
    assert batches[0][0] == {
        "method": "POST",
        "path": "api/subscribers",
        "body": {
            "email": "p0@x.io",
            "fields": {k.value: v for k, v in changes[0].fields.items()},
        },
    }
    assert [c.args[0] for c in configured.await_args_list] == [
        mailerlite_backfill.UPSERT_BATCH_INTERVAL_SECONDS
    ] * 2
    assert (ok, failed) == (120, 0)
    assert progress == [50, 100, 120]


@pytest.mark.asyncio
async def test_apply_counts_a_refused_upsert_without_logging_the_address(
    configured, monkeypatch, caplog
):
    client = MagicMock()
    refusal = {
        "message": "The given data was invalid.",
        "errors": {"email": ["bad@x.io is not a deliverable address."]},
    }
    client.post = AsyncMock(
        return_value=_response(200, {"responses": [{"code": 422, "body": refusal}]})
    )
    monkeypatch.setattr(mailerlite_backfill, "_client", lambda: client)
    changes = backfill.plan([_person("bad@x.io")], {}).changes

    assert await backfill.apply(changes) == (0, 1)
    assert "bad@x.io" not in caplog.text
    assert " at .io with 422 " in caplog.text
    assert "The given data was invalid." in caplog.text
    assert "is not a deliverable address" in caplog.text


@pytest.mark.asyncio
async def test_read_current_pages_every_status(configured, monkeypatch):
    pages = {
        ("active", None): {
            "data": [
                {
                    "email": "A@x.io",
                    "fields": {"subscription_status": "signed", "city": "Leeds"},
                }
            ],
            "meta": {"next_cursor": "c2"},
        },
        ("active", "c2"): {"data": [{"email": "b@x.io", "fields": {}}], "meta": {}},
        ("unsubscribed", None): {
            "data": [{"email": "u@x.io", "fields": None}],
            "meta": {},
        },
    }

    async def get(url, **kw):
        _, _, query = url.partition("?")
        params = dict(p.split("=", 1) for p in query.split("&"))
        key = (params["filter%5Bstatus%5D"], params.get("cursor"))
        return _response(200, pages.get(key, {"data": [], "meta": {}}))

    client = MagicMock(get=AsyncMock(side_effect=get))
    monkeypatch.setattr(backfill, "_client", lambda: client)

    current = await backfill.read_current()

    assert set(current) == {"a@x.io", "b@x.io", "u@x.io"}
    assert current["a@x.io"]["subscription_status"] == "signed"
    assert "city" not in current["a@x.io"]


@pytest.mark.asyncio
async def test_read_current_stops_on_a_repeated_cursor(configured, monkeypatch):
    """As for the audience read: a repeated cursor is an error, not the end."""
    page = {"data": [{"email": "a@x.io", "fields": {}}], "meta": {"next_cursor": "c2"}}
    calls = 0

    async def get(url, **kw):
        nonlocal calls
        calls += 1
        if calls > 10:
            raise AssertionError("the reader requested the same page forever")
        return _response(200, page)

    client = MagicMock(get=AsyncMock(side_effect=get))
    monkeypatch.setattr(backfill, "_client", lambda: client)

    with pytest.raises(mailerlite.MailerLiteError, match="repeated"):
        await backfill.read_current()
    assert client.get.await_count == 2


def test_without_create_only_existing_subscribers_are_planned():
    """A Stripe customer is not proof of a checkout: the billing portal makes
    one too. So `mailerlite-backfill` never creates anyone, and someone
    MailerLite does not hold gets no change at all."""
    portal_only = _person("portal@x.io")
    existing = _person("held@x.io", _paid("active"))
    plan = backfill.plan(
        [portal_only, existing],
        {"held@x.io": {"subscription_status": "signed"}},
        create=False,
    )
    assert [c.person.email for c in plan.changes] == ["held@x.io"]
    assert not any(c.new for c in plan.changes)


@pytest.mark.parametrize("create", [True, False])
def test_an_opted_out_person_is_counted_and_never_written(create):
    """A field write creates the subscriber, so someone who refused marketing
    is left out even when MailerLite already holds them with stale fields."""
    opted_out = _person("out@x.io", _paid("active")).model_copy(
        update={"marketing_opt_out_at": CREATED}
    )
    plan = backfill.plan(
        [opted_out, _person("held@x.io", _paid("active"))],
        {
            "out@x.io": {"subscription_status": "signed"},
            "held@x.io": {"subscription_status": "signed"},
        },
        create=create,
    )
    assert [c.person.email for c in plan.changes] == ["held@x.io"]
    assert plan.opted_out == 1
    assert plan.invalid == 0
    assert plan.statuses[S.SUBSCRIBED] == 1
