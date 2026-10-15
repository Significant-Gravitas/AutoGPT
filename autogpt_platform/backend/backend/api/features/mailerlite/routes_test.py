"""MailerLite's unsubscribe webhook records a refusal, and only a call signed
with the webhook's secret can."""

import hashlib
import hmac
import json
from collections.abc import Iterator
from unittest.mock import AsyncMock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
from fastapi.routing import APIRoute

from backend.api.rest_api import app as real_app

from . import routes
from .routes import router

SECRET = "4jQ3Y4UlLI"
PATH = "/mailerlite/webhook"
EMAIL = "sam@example.com"

app = fastapi.FastAPI()
app.include_router(router)
client = fastapi.testclient.TestClient(app)


@pytest.fixture(scope="session")
def server() -> None:
    """Pure route logic; no live stack."""
    return None


@pytest.fixture(scope="session", autouse=True)
def graph_cleanup() -> Iterator[None]:
    yield


@pytest.fixture(autouse=True)
def secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(routes.settings.secrets, "mailerlite_webhook_secret", SECRET)


@pytest.fixture
def record(mocker: pytest_mock.MockFixture) -> AsyncMock:
    return mocker.patch.object(
        routes,
        "record_marketing_opt_out_by_email",
        new=AsyncMock(return_value="user-1"),
    )


def _unsubscribed(email: str = EMAIL) -> dict:
    """Shaped like MailerLite's documented single-event payload."""
    return {
        "id": "100000000000000000",
        "email": email,
        "status": "unsubscribed",
        "unsubscribed_at": "2026-10-07T08:26:04.000000Z",
        "fields": {"name": ""},
        "event": "subscriber.unsubscribed",
        "account_id": 0,
    }


def _post(payload: object, secret: str = SECRET, signature: str | None = None):
    body = json.dumps(payload).encode()
    if signature is None:
        signature = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return client.post(
        PATH,
        content=body,
        headers={"Content-Type": "application/json", "Signature": signature},
    )


def test_an_unsubscribe_records_the_opt_out(record: AsyncMock) -> None:
    response = _post(_unsubscribed())

    assert response.status_code == 200
    record.assert_awaited_once_with(EMAIL, "email_unsubscribe")


def test_a_batched_unsubscribe_records_each_one(record: AsyncMock) -> None:
    payload = {
        "events": [
            {"type": "subscriber.unsubscribed", "subscriber": {"email": EMAIL}},
            {"type": "campaign.open", "subscriber": {"email": "other@example.com"}},
            {
                "type": "subscriber.unsubscribed",
                "subscriber": {"email": "third@example.com"},
            },
        ],
        "total": 3,
    }

    assert _post(payload).status_code == 200

    assert [c.args[0] for c in record.await_args_list] == [
        EMAIL,
        "third@example.com",
    ]


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({**_unsubscribed(), "event": "subscriber.created"}, id="other"),
        pytest.param({**_unsubscribed(), "email": None}, id="no-email"),
        pytest.param({}, id="empty"),
    ],
)
def test_anything_else_is_acknowledged_and_ignored(
    record: AsyncMock, payload: dict
) -> None:
    assert _post(payload).status_code == 200
    record.assert_not_awaited()


def test_an_unknown_or_repeated_address_is_still_acknowledged(
    record: AsyncMock,
) -> None:
    """MailerLite retries anything but a 2xx, and a retry changes nothing."""
    record.return_value = None

    assert _post(_unsubscribed()).status_code == 200
    assert _post(_unsubscribed()).status_code == 200


@pytest.mark.parametrize(
    "signature",
    [
        pytest.param("", id="missing"),
        pytest.param("0" * 64, id="wrong"),
        pytest.param(
            hmac.new(b"other-secret", b"{}", hashlib.sha256).hexdigest(),
            id="other-secret",
        ),
    ],
)
def test_a_badly_signed_call_is_refused(record: AsyncMock, signature: str) -> None:
    response = _post(_unsubscribed(), signature=signature)

    assert response.status_code == 401
    record.assert_not_awaited()


def test_a_non_ascii_signature_is_refused_not_a_crash(record: AsyncMock) -> None:
    response = client.post(
        PATH,
        content=json.dumps(_unsubscribed()).encode(),
        headers={"Content-Type": "application/json", "Signature": b"\xe9"},
    )

    assert response.status_code == 401
    record.assert_not_awaited()


def test_the_signature_covers_the_exact_body(record: AsyncMock) -> None:
    """Signed over one body and sent with another: refused."""
    signed = json.dumps(_unsubscribed()).encode()
    signature = hmac.new(SECRET.encode(), signed, hashlib.sha256).hexdigest()

    response = client.post(
        PATH,
        content=json.dumps(_unsubscribed("victim@example.com")).encode(),
        headers={"Signature": signature},
    )

    assert response.status_code == 401
    record.assert_not_awaited()


def test_without_a_secret_every_call_is_refused(
    record: AsyncMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty key would let anyone sign, so unconfigured means closed."""
    monkeypatch.setattr(routes.settings.secrets, "mailerlite_webhook_secret", "")

    response = _post(_unsubscribed(), secret="")

    assert response.status_code == 503
    record.assert_not_awaited()


def test_a_signed_body_that_is_not_an_event_is_rejected(record: AsyncMock) -> None:
    body = b"not json"
    signature = hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()

    response = client.post(PATH, content=body, headers={"Signature": signature})

    assert response.status_code == 400
    record.assert_not_awaited()


def test_a_failed_write_is_not_acknowledged(record: AsyncMock) -> None:
    """A 5xx makes MailerLite retry, and the write is idempotent."""
    record.side_effect = RuntimeError("database down")
    failing = fastapi.testclient.TestClient(app, raise_server_exceptions=False)
    body = json.dumps(_unsubscribed()).encode()
    signature = hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()

    response = failing.post(PATH, content=body, headers={"Signature": signature})

    assert response.status_code == 500


def test_the_webhook_is_mounted_without_user_auth() -> None:
    """MailerLite carries no user credential; a dependency here would refuse
    every delivery. The signature is the only check."""
    route = next(
        r
        for r in real_app.routes
        if isinstance(r, APIRoute) and r.path == "/api/email/mailerlite/webhook"
    )
    assert route.methods == {"POST"}
    assert route.dependant.dependencies == []
    assert "security" not in real_app.openapi()["paths"][route.path]["post"]
