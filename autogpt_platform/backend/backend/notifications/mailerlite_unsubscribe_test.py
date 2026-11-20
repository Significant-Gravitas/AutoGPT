"""A refusal reaches someone MailerLite already has, and never creates anyone.

`unsubscribe` is the one MailerLite write queued for an opted-out account, so
it must not go through the upsert every other write uses: that would create
the very subscriber the refusal rules out.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.data.notifications import AudienceAction
from backend.notifications import mailerlite, subscriber_fields
from backend.notifications.notifications import NotificationManager

EMAIL = "sam@example.com"


@pytest.fixture(autouse=True)
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
    return fake


def _response(status: int, body: dict | None = None) -> MagicMock:
    response = MagicMock(status=status)
    response.json.return_value = body or {}
    return response


def _client(held: dict | None, put_status: int = 200) -> MagicMock:
    """MailerLite holding `held` for EMAIL (None: nobody)."""
    client = MagicMock()
    client.get = AsyncMock(
        return_value=_response(404) if held is None else _response(200, {"data": held})
    )
    client.put = AsyncMock(return_value=_response(put_status))
    client.post = AsyncMock(return_value=_response(200))
    client.delete = AsyncMock(return_value=_response(200))
    return client


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["active", "unconfirmed", None])
async def test_a_subscriber_is_marked_unsubscribed_by_id(status):
    client = _client({"id": "ml_1", "status": status})
    with patch.object(mailerlite, "_client", return_value=client):
        await mailerlite.unsubscribe(EMAIL)

    client.get.assert_awaited_once()
    assert client.get.await_args.args[0].endswith(f"/subscribers/{EMAIL}")
    client.put.assert_awaited_once()
    assert client.put.await_args.args[0].endswith("/subscribers/ml_1")
    assert client.put.await_args.kwargs["json"] == {"status": "unsubscribed"}
    client.post.assert_not_awaited()
    client.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_someone_mailerlite_does_not_have_is_never_created(caplog):
    client = _client(None)
    with (
        patch.object(mailerlite, "_client", return_value=client),
        caplog.at_level("INFO"),
    ):
        await mailerlite.unsubscribe(EMAIL)

    client.put.assert_not_awaited()
    client.post.assert_not_awaited()
    assert "nothing to unsubscribe" in caplog.text
    assert EMAIL not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["unsubscribed", "bounced", "junk"])
async def test_a_subscriber_mailerlite_already_does_not_mail_is_left_alone(status):
    client = _client({"id": "ml_1", "status": status})
    with patch.object(mailerlite, "_client", return_value=client):
        await mailerlite.unsubscribe(EMAIL)

    client.put.assert_not_awaited()
    client.post.assert_not_awaited()


@pytest.mark.asyncio
async def test_fields_are_never_written_for_someone_who_refused():
    client = _client({"id": "ml_1", "status": "active"})
    with patch.object(mailerlite, "_client", return_value=client):
        await mailerlite.unsubscribe(EMAIL, {next(iter(mailerlite.FIELD_TYPES)): "x"})

    assert client.put.await_args.kwargs["json"] == {"status": "unsubscribed"}
    client.post.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_subscriber_deleted_since_the_lookup_is_done():
    client = _client({"id": "ml_1", "status": "active"}, put_status=404)
    with patch.object(mailerlite, "_client", return_value=client):
        await mailerlite.unsubscribe(EMAIL)


@pytest.mark.asyncio
async def test_a_failed_write_raises_so_the_queue_retries():
    client = _client({"id": "ml_1", "status": "active"}, put_status=500)
    with (
        patch.object(mailerlite, "_client", return_value=client),
        pytest.raises(mailerlite.MailerLiteError) as raised,
    ):
        await mailerlite.unsubscribe(EMAIL)

    assert EMAIL not in str(raised.value)
    assert mailerlite.pseudonym(EMAIL) in str(raised.value)


@pytest.mark.asyncio
async def test_without_a_token_it_is_not_configured(mailerlite_configured):
    mailerlite_configured.secrets.mailerlite_api_token = ""
    with pytest.raises(mailerlite.MailerLiteNotConfigured):
        await mailerlite.unsubscribe(EMAIL)


@pytest.mark.asyncio
async def test_the_consumer_routes_an_unsubscribe_to_it(monkeypatch):
    handler = AsyncMock()
    monkeypatch.setattr(mailerlite, "unsubscribe", handler)
    event = subscriber_fields.audience_event(
        AudienceAction.UNSUBSCRIBE, EMAIL, "user-1"
    )
    assert event is not None

    assert await NotificationManager._process_audience_change(
        MagicMock(), event.model_dump_json()
    )

    handler.assert_awaited_once_with(EMAIL, None)
