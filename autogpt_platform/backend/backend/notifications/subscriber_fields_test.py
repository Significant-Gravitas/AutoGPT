"""Customers carry their status and dates into MailerLite, and neither a
MailerLite outage nor an address it refuses ever costs the checkout or the
billing email the update rides along with. A signup alone queues nothing."""

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.data import user as user_data
from backend.data.notifications import (
    AudienceAction,
    NotificationResult,
    SubscriberField,
    SubscriptionStatus,
)
from backend.notifications import mailerlite
from backend.notifications import notifications as delivery
from backend.notifications import subscriber_fields
from backend.notifications.notifications import NotificationManager

EMAIL = "sam@example.com"


@pytest.fixture
def fields_on(monkeypatch):
    queued = AsyncMock(return_value=NotificationResult(success=True))
    monkeypatch.setattr(subscriber_fields, "queue_audience_change", queued)
    return queued


def test_a_stripe_timestamp_is_its_utc_day():
    # 23:30 UTC on 1 Sep 2026 is already 2 Sep further east; MailerLite gets UTC.
    assert subscriber_fields.mailerlite_date(1788305400) == "2026-09-01"
    assert (
        subscriber_fields.mailerlite_date(datetime(2026, 6, 24, 23, 59, tzinfo=UTC))
        == "2026-06-24"
    )


def test_a_naive_datetime_is_read_as_utc():
    assert subscriber_fields.mailerlite_date(datetime(2026, 6, 24, 1, 0)) == (
        "2026-06-24"
    )


def test_a_new_subscription_clears_the_last_ones_cancellation_and_end():
    assert subscriber_fields.subscribed(1788305400) == {
        SubscriberField.STATUS: "subscribed",
        SubscriberField.SUBSCRIPTION_STARTED: "2026-09-01",
        SubscriberField.SUBSCRIPTION_CANCELED: None,
        SubscriberField.SUBSCRIPTION_ENDED: None,
    }


SIGNED = subscriber_fields.signed(datetime(2026, 9, 29, 10, 0, tzinfo=UTC))


@pytest.mark.asyncio
async def test_a_field_change_is_queued_without_any_mailerlite_settings(fields_on):
    """It is queued from the API server, which never holds the MailerLite
    token: the notification service decides whether it goes anywhere."""
    await subscriber_fields.queue_fields(
        "user-1", EMAIL, SIGNED, AudienceAction.CHECKOUT_OPENED
    )
    event = fields_on.await_args.args[0]
    assert event.action is AudienceAction.CHECKOUT_OPENED
    assert event.fields == {
        SubscriberField.STATUS: SubscriptionStatus.SIGNED.value,
        SubscriberField.SIGNUP: "2026-09-29",
    }


@pytest.mark.asyncio
async def test_a_reserved_domain_is_skipped_not_retried(fields_on):
    await subscriber_fields.queue_fields("user-1", "sam@site.test", SIGNED)
    fields_on.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        AsyncMock(return_value=NotificationResult(success=False, message="down")),
        AsyncMock(side_effect=RuntimeError("broker gone")),
    ],
)
async def test_a_queue_failure_is_reported_never_raised(
    monkeypatch, fields_on, failure
):
    monkeypatch.setattr(subscriber_fields, "queue_audience_change", failure)
    await subscriber_fields.queue_fields("user-1", EMAIL, SIGNED)
    failure.assert_awaited_once()


# ── signup hook ────────────────────────────────────────────────────────────


def _prisma(existing):
    created = SimpleNamespace(
        id="user-1",
        email=EMAIL,
        createdAt=datetime(2026, 9, 29, tzinfo=UTC),
        name=None,
    )
    return (
        MagicMock(
            user=MagicMock(
                find_unique=AsyncMock(return_value=existing),
                create=AsyncMock(return_value=created),
            )
        ),
        created,
    )


async def _get_or_create(existing):
    prisma, created = _prisma(existing)
    scheduled: list[asyncio.Task] = []
    create_task = asyncio.create_task

    def recording(coro, **kwargs):
        task = create_task(coro, **kwargs)
        scheduled.append(task)
        return task

    with (
        patch.object(user_data, "prisma", prisma),
        patch.object(user_data, "_ensure_user_profile", AsyncMock()),
        patch.object(user_data, "ensure_personal_org", AsyncMock(return_value=False)),
        patch.object(user_data, "schedule_posthog_lifecycle_sync", MagicMock()),
        patch.object(user_data.User, "from_db", MagicMock()),
        patch.object(user_data, "UserCreationResult", MagicMock()),
        patch("asyncio.create_task", recording),
    ):
        await user_data.get_or_create_user_with_status(
            {"sub": "user-1", "email": EMAIL}
        )
        # Whatever signup scheduled in the background runs before the patches
        # go, so a MailerLite queue it reached would be seen.
        await asyncio.gather(*scheduled, return_exceptions=True)
    return created


@pytest.mark.asyncio
async def test_a_new_account_is_not_sent_to_mailerlite(fields_on):
    """Only checkout openers belong in MailerLite: a signup alone queues
    nothing, so it can never create a subscriber."""
    await _get_or_create(None)
    fields_on.assert_not_awaited()
    for name in ("_sync_signup", "queue_signup", "queue_fields", "_signup_sync_tasks"):
        assert not hasattr(user_data, name)


# ── the MailerLite client ──────────────────────────────────────────────────


@pytest.fixture
def mailerlite_configured(monkeypatch):
    fake = SimpleNamespace(
        config=SimpleNamespace(
            mailerlite_changelog_group_id="grp_changelog",
            mailerlite_trial_group_id="",
        ),
        secrets=SimpleNamespace(mailerlite_api_token="token"),
    )
    monkeypatch.setattr(mailerlite, "settings", fake)
    monkeypatch.setattr(mailerlite, "_fields_ready", False)
    return fake


def _response(status: int, body: dict | None = None) -> MagicMock:
    response = MagicMock(status=status)
    response.json.return_value = body or {}
    return response


def _all_fields() -> dict:
    return {
        "data": [
            {"key": f.value, "type": t} for f, t in mailerlite.FIELD_TYPES.items()
        ],
        "meta": {"last_page": 1},
    }


STATUS = {
    SubscriberField.STATUS: "subscribed",
    SubscriberField.SUBSCRIPTION_ENDED: None,
}


@pytest.mark.asyncio
async def test_a_group_add_carries_the_fields_in_the_same_upsert(
    mailerlite_configured,
):
    client = MagicMock(
        get=AsyncMock(return_value=_response(200, _all_fields())),
        post=AsyncMock(return_value=_response(200)),
    )
    with patch.object(mailerlite, "_client", return_value=client):
        await mailerlite.add_to_changelog(EMAIL, STATUS)
    assert client.post.await_args.kwargs["json"] == {
        "email": EMAIL,
        "groups": ["grp_changelog"],
        "fields": {
            "subscription_status": "subscribed",
            "subscription_ended_date": None,
        },
    }


@pytest.mark.asyncio
async def test_missing_fields_are_created_once_before_the_first_write(
    mailerlite_configured,
):
    client = MagicMock(
        get=AsyncMock(return_value=_response(200, {"data": [], "meta": {}})),
        post=AsyncMock(
            side_effect=lambda url, **kw: _response(
                201,
                (
                    {"data": {"key": kw["json"].get("name")}}
                    if url.endswith("/fields")
                    else {}
                ),
            )
        ),
    )
    with patch.object(mailerlite, "_client", return_value=client):
        await mailerlite.update_fields(EMAIL, STATUS)
        await mailerlite.update_fields(EMAIL, STATUS)
    created = [
        c.kwargs["json"]
        for c in client.post.await_args_list
        if c.args[0].endswith("/fields")
    ]
    assert created == [
        {"name": f.value, "type": t} for f, t in mailerlite.FIELD_TYPES.items()
    ]
    client.get.assert_awaited_once()
    upserts = [
        c for c in client.post.await_args_list if c.args[0].endswith("/subscribers")
    ]
    assert len(upserts) == 2
    assert "groups" not in upserts[0].kwargs["json"]


@pytest.mark.asyncio
async def test_a_field_under_another_key_fails_loudly(mailerlite_configured):
    client = MagicMock(
        get=AsyncMock(return_value=_response(200, {"data": [], "meta": {}})),
        post=AsyncMock(return_value=_response(201, {"data": {"key": "status_2"}})),
    )
    with (
        patch.object(mailerlite, "_client", return_value=client),
        pytest.raises(mailerlite.MailerLiteError),
    ):
        await mailerlite.update_fields(EMAIL, STATUS)


@pytest.mark.asyncio
async def test_reading_fields_stops_on_an_empty_page(mailerlite_configured):
    """An empty page is the end, whatever last_page claims."""
    client = MagicMock(
        get=AsyncMock(
            side_effect=[
                _response(200, _all_fields() | {"meta": {"last_page": 1000}}),
                _response(200, {"data": [], "meta": {"last_page": 1000}}),
            ]
            + [AssertionError("requested a page after an empty one")] * 1000
        ),
    )
    with patch.object(mailerlite, "_client", return_value=client):
        fields = await mailerlite.read_fields()
    assert set(fields) == {f.value for f in mailerlite.FIELD_TYPES}
    assert client.get.await_count == 2


@pytest.mark.asyncio
async def test_reading_fields_stops_when_last_page_keeps_growing(
    mailerlite_configured, monkeypatch
):
    monkeypatch.setattr(mailerlite, "_MAX_FIELD_PAGES", 3)
    calls = 0

    def page(url, **kw):
        nonlocal calls
        calls += 1
        if calls > 10:
            raise AssertionError("still paging")
        return _response(
            200,
            {
                "data": [{"key": f"f{calls}", "type": "text"}],
                "meta": {"last_page": calls + 1},
            },
        )

    client = MagicMock(get=AsyncMock(side_effect=page))
    with (
        patch.object(mailerlite, "_client", return_value=client),
        pytest.raises(mailerlite.MailerLiteError, match="pages"),
    ):
        await mailerlite.read_fields()
    assert calls == 3


@pytest.mark.asyncio
async def test_fields_land_even_while_the_group_is_unconfigured(
    mailerlite_configured,
):
    """The status is not the group's to hold hostage: it is written, and the
    group change is then dead-lettered (see below)."""
    mailerlite_configured.config.mailerlite_changelog_group_id = ""
    client = MagicMock(
        get=AsyncMock(return_value=_response(200, _all_fields())),
        post=AsyncMock(return_value=_response(200)),
    )
    with (
        patch.object(mailerlite, "_client", return_value=client),
        pytest.raises(mailerlite.MailerLiteNotConfigured),
    ):
        await mailerlite.add_to_changelog(EMAIL, STATUS)
    assert client.post.await_args.kwargs["json"]["fields"] == {
        "subscription_status": "subscribed",
        "subscription_ended_date": None,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", [mailerlite.add_to_trial, mailerlite.remove_from_trial]
)
async def test_without_a_trial_group_only_the_fields_are_written(
    mailerlite_configured, change
):
    """The trial group is optional, so a trial change without one writes the
    status and is done, instead of retrying into the dead-letter queue."""
    client = MagicMock(
        get=AsyncMock(return_value=_response(200, _all_fields())),
        post=AsyncMock(return_value=_response(200)),
        delete=AsyncMock(),
    )
    with patch.object(mailerlite, "_client", return_value=client):
        await change(EMAIL, STATUS)
    assert client.post.await_args.kwargs["json"] == {
        "email": EMAIL,
        "fields": {
            "subscription_status": "subscribed",
            "subscription_ended_date": None,
        },
    }
    client.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_removal_writes_the_fields_too(mailerlite_configured):
    client = MagicMock(
        get=AsyncMock(
            side_effect=[
                _response(200, _all_fields()),
                _response(200, {"data": {"id": "ml_1"}}),
            ]
        ),
        post=AsyncMock(return_value=_response(200)),
        delete=AsyncMock(return_value=_response(204)),
    )
    with patch.object(mailerlite, "_client", return_value=client):
        await mailerlite.remove_from_changelog(EMAIL, STATUS)
    assert client.post.await_args.kwargs["json"]["fields"]["subscription_status"] == (
        "subscribed"
    )
    client.delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_no_fields_means_no_field_calls(mailerlite_configured):
    client = MagicMock(post=AsyncMock(return_value=_response(201)))
    with patch.object(mailerlite, "_client", return_value=client):
        await mailerlite.add_to_changelog(EMAIL)
        await mailerlite.update_fields(EMAIL, {})
    assert client.post.await_args.kwargs["json"] == {
        "email": EMAIL,
        "groups": ["grp_changelog"],
    }
    client.post.assert_awaited_once()


# ── ordering against the checkout ──────────────────────────────────────────


class _FakeMailerLite:
    """Subscribers as MailerLite holds them: an upsert merges its fields into
    whatever is already there, so the last write wins."""

    def __init__(self, held: dict[str, dict] | None = None):
        self.subscribers = held or {}

    async def get(self, url: str, **_) -> MagicMock:
        if "/fields" in url:
            return _response(200, _all_fields())
        held = self.subscribers.get(url.rsplit("/", 1)[-1])
        if held is None:
            return _response(404)
        return _response(200, {"data": {"id": "ml_1", "fields": dict(held)}})

    async def post(self, url: str, json: dict, **_) -> MagicMock:
        held = self.subscribers.setdefault(json["email"], {})
        held.update(json.get("fields") or {})
        return _response(200)


async def _consume(*events) -> None:
    for event in events:
        assert await NotificationManager._process_audience_change(
            MagicMock(), event.model_dump_json()
        )


async def _signup_event(fields_on):
    """A signup queued before signups stopped being sent, still in the queue
    or replayed from the dead-letter queue."""
    return subscriber_fields.audience_event(
        AudienceAction.SIGNUP, EMAIL, "user-1", SIGNED
    )


def _checkout_event():
    return subscriber_fields.audience_event(
        AudienceAction.ENROLL_TOUR,
        EMAIL,
        "user-1",
        subscriber_fields.subscribed(1788305400),
    )


@pytest.mark.asyncio
async def test_a_signup_synced_after_the_checkout_does_not_undo_it(
    mailerlite_configured, fields_on
):
    """The signup is queued from a background task, so nothing orders it
    before the account's first checkout. Landing second, its `signed` must not
    replace the `subscribed` the checkout wrote; the signup date still lands."""
    mailerlite_configured.config.mailerlite_onboarding_group_id = "grp_tour"
    signup = await _signup_event(fields_on)
    ml = _FakeMailerLite()
    with patch.object(mailerlite, "_client", return_value=ml):
        await _consume(_checkout_event(), signup)
    assert ml.subscribers[EMAIL]["subscription_status"] == "subscribed"
    assert ml.subscribers[EMAIL]["signup_date"] == "2026-09-29"


@pytest.mark.asyncio
async def test_a_stale_signup_never_creates_a_subscriber(
    mailerlite_configured, fields_on
):
    """A signup still queued from before signups stopped being sent must not
    bring someone who never opened checkout into MailerLite."""
    signup = await _signup_event(fields_on)
    ml = _FakeMailerLite()
    ml.post = AsyncMock(side_effect=ml.post)
    with patch.object(mailerlite, "_client", return_value=ml):
        await _consume(signup)
    ml.post.assert_not_awaited()
    assert EMAIL not in ml.subscribers


@pytest.mark.asyncio
async def test_a_signup_gives_a_subscriber_with_no_status_signed(
    mailerlite_configured, fields_on
):
    """Someone already on the newsletter has no status of ours yet."""
    signup = await _signup_event(fields_on)
    ml = _FakeMailerLite({EMAIL: {"subscription_status": None, "city": "Leeds"}})
    with patch.object(mailerlite, "_client", return_value=ml):
        await _consume(signup)
    assert ml.subscribers[EMAIL]["subscription_status"] == "signed"


# ── a required group without an ID ─────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action", [AudienceAction.ENROLL_TOUR, AudienceAction.REMOVE_CHANGELOG]
)
async def test_a_missing_required_group_is_dead_lettered_at_once(
    mailerlite_configured, monkeypatch, caplog, action
):
    """No retry makes a setting appear, so the change goes to the dead-letter
    queue on the first attempt, to be replayed once the ID is set. The fields
    are written once and the problem is said once."""
    mailerlite_configured.config.mailerlite_onboarding_group_id = ""
    mailerlite_configured.config.mailerlite_changelog_group_id = ""
    event = subscriber_fields.audience_event(
        action, EMAIL, "user-1", subscriber_fields.subscribed(1788305400)
    )
    message = MagicMock(
        body=event.model_dump_json().encode(), ack=AsyncMock(), reject=AsyncMock()
    )
    ml = _FakeMailerLite()
    ml.post = AsyncMock(side_effect=ml.post)
    sleep = AsyncMock()
    monkeypatch.setattr(delivery.asyncio, "sleep", sleep)
    with (
        patch.object(mailerlite, "_client", return_value=ml),
        caplog.at_level("WARNING"),
    ):
        manager = NotificationManager.__new__(NotificationManager)
        await manager._process_message_with_retry(
            message, manager._process_audience_change, delivery.AUDIENCE_QUEUE
        )
    message.reject.assert_awaited_once_with(requeue=False)
    message.ack.assert_not_awaited()
    sleep.assert_not_awaited()
    assert ml.post.await_count == 1
    assert ml.subscribers[EMAIL]["subscription_status"] == "subscribed"
    assert caplog.text.count("group ID is not configured") == 1
    assert EMAIL not in caplog.text
