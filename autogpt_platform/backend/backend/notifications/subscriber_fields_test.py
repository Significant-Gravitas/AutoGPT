"""Every account carries its status and dates into MailerLite, from signup on,
and neither a MailerLite outage nor an address it refuses ever costs the
signup or the billing email the update rides along with."""

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
from backend.notifications import mailerlite, subscriber_fields

EMAIL = "sam@example.com"


@pytest.fixture
def fields_on(monkeypatch):
    monkeypatch.setattr(
        subscriber_fields,
        "settings",
        SimpleNamespace(secrets=SimpleNamespace(mailerlite_api_token="token")),
    )
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


@pytest.mark.asyncio
async def test_a_signup_queues_signed_with_the_account_creation_day(fields_on):
    await subscriber_fields.queue_signup(
        "user-1", EMAIL, datetime(2026, 9, 29, 10, 0, tzinfo=UTC)
    )
    event = fields_on.await_args.args[0]
    assert event.action is AudienceAction.UPDATE_FIELDS
    assert event.fields == {
        SubscriberField.STATUS: SubscriptionStatus.SIGNED.value,
        SubscriberField.SIGNUP: "2026-09-29",
    }


@pytest.mark.asyncio
async def test_without_a_token_nothing_is_queued(monkeypatch, fields_on):
    monkeypatch.setattr(
        subscriber_fields,
        "settings",
        SimpleNamespace(secrets=SimpleNamespace(mailerlite_api_token="")),
    )
    await subscriber_fields.queue_signup("user-1", EMAIL, datetime.now(UTC))
    fields_on.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_reserved_domain_is_skipped_not_retried(fields_on):
    await subscriber_fields.queue_signup("user-1", "sam@site.test", datetime.now(UTC))
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
    await subscriber_fields.queue_signup("user-1", EMAIL, datetime.now(UTC))
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


async def _get_or_create(existing, queue_signup):
    prisma, created = _prisma(existing)
    with (
        patch.object(user_data, "prisma", prisma),
        patch.object(user_data, "_ensure_user_profile", AsyncMock()),
        patch.object(user_data, "ensure_personal_org", AsyncMock()),
        patch.object(user_data.User, "from_db", MagicMock()),
        patch.object(user_data, "UserCreationResult", MagicMock()),
        patch.object(user_data, "queue_signup", queue_signup),
    ):
        await user_data.get_or_create_user_with_status(
            {"sub": "user-1", "email": EMAIL}
        )
        # Let the background task run.
        for task in list(user_data._signup_sync_tasks):
            await task
    return created


@pytest.mark.asyncio
async def test_a_new_account_is_queued_for_mailerlite():
    queue_signup = AsyncMock()
    created = await _get_or_create(None, queue_signup)
    queue_signup.assert_awaited_once_with("user-1", EMAIL, created.createdAt)


@pytest.mark.asyncio
async def test_an_existing_account_is_not_queued_again():
    queue_signup = AsyncMock()
    await _get_or_create(SimpleNamespace(id="user-1", email=EMAIL), queue_signup)
    queue_signup.assert_not_awaited()


@pytest.mark.asyncio
async def test_signup_never_fails_because_the_sync_does():
    def broken(*_):
        raise RuntimeError("no event loop for you")

    await _get_or_create(None, MagicMock(side_effect=broken))


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
async def test_fields_land_even_while_the_group_is_unconfigured(
    mailerlite_configured,
):
    """The status is not the group's to hold hostage: it is written, and the
    group change then retries as it always has."""
    client = MagicMock(
        get=AsyncMock(return_value=_response(200, _all_fields())),
        post=AsyncMock(return_value=_response(200)),
    )
    with (
        patch.object(mailerlite, "_client", return_value=client),
        pytest.raises(mailerlite.MailerLiteNotConfigured),
    ):
        await mailerlite.add_to_trial(EMAIL, STATUS)
    assert client.post.await_args.kwargs["json"]["fields"] == {
        "subscription_status": "subscribed",
        "subscription_ended_date": None,
    }


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
