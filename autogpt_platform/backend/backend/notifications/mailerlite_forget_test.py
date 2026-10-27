"""A deleted account's subscriber is forgotten by ID, which erases their data,
and someone MailerLite does not have is already gone."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.notifications import mailerlite

EMAIL = "sam@example.com"


@pytest.fixture(autouse=True)
def mailerlite_configured(monkeypatch):
    fake = SimpleNamespace(
        config=SimpleNamespace(),
        secrets=SimpleNamespace(mailerlite_api_token="token"),
    )
    monkeypatch.setattr(mailerlite, "settings", fake)


def _response(status: int, body: dict | None = None) -> MagicMock:
    response = MagicMock(status=status)
    response.json.return_value = body or {}
    return response


def _client(held: dict | None, post_status: int = 200) -> MagicMock:
    client = MagicMock()
    client.get = AsyncMock(
        return_value=_response(404) if held is None else _response(200, {"data": held})
    )
    client.post = AsyncMock(return_value=_response(post_status))
    client.delete = AsyncMock(return_value=_response(204))
    return client


@pytest.mark.asyncio
async def test_a_subscriber_is_forgotten_by_id():
    client = _client({"id": "ml_1", "status": "active"})
    with patch.object(mailerlite, "_client", return_value=client):
        assert await mailerlite.forget_subscriber(EMAIL) is True

    client.post.assert_awaited_once()
    assert client.post.await_args.args[0].endswith("/subscribers/ml_1/forget")
    client.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_nobody_to_forget_is_a_no_op():
    client = _client(None)
    with patch.object(mailerlite, "_client", return_value=client):
        assert await mailerlite.forget_subscriber(EMAIL) is False

    client.post.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_subscriber_gone_since_the_lookup_counts_as_gone():
    client = _client({"id": "ml_1"}, post_status=404)
    with patch.object(mailerlite, "_client", return_value=client):
        assert await mailerlite.forget_subscriber(EMAIL) is False


@pytest.mark.asyncio
async def test_a_failed_forget_raises_without_the_address():
    client = _client({"id": "ml_1"}, post_status=500)
    with patch.object(mailerlite, "_client", return_value=client):
        with pytest.raises(mailerlite.MailerLiteError) as raised:
            await mailerlite.forget_subscriber(EMAIL)

    assert EMAIL not in str(raised.value)


@pytest.mark.asyncio
async def test_forgetting_needs_a_token(monkeypatch):
    monkeypatch.setattr(
        mailerlite,
        "settings",
        SimpleNamespace(secrets=SimpleNamespace(mailerlite_api_token="")),
    )
    with pytest.raises(mailerlite.MailerLiteNotConfigured):
        await mailerlite.forget_subscriber(EMAIL)
