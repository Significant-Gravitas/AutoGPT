from unittest.mock import MagicMock

import fastapi
import fastapi.testclient
import pytest

from backend.api.features.auth_email import routes as auth_email_routes

app = fastapi.FastAPI()
app.include_router(auth_email_routes.auth_email_router, prefix="/api")
client = fastapi.testclient.TestClient(app, raise_server_exceptions=False)

VALID_BODY = {
    "type": "reset_password",
    "to": "user@example.com",
    "url": "https://platform.agpt.co/reset-password?token=abc",
}


@pytest.fixture
def send_mock(monkeypatch):
    monkeypatch.setattr(
        auth_email_routes.settings.secrets, "auth_email_token", "s3cret"
    )
    monkeypatch.setattr(
        auth_email_routes.settings.config,
        "frontend_base_url",
        "https://platform.agpt.co",
    )
    mock = MagicMock()
    monkeypatch.setattr(auth_email_routes._email_sender, "send_transactional", mock)
    return mock


def _post(body=None, token="s3cret"):
    headers = {"X-Auth-Email-Token": token} if token is not None else {}
    return client.post("/api/auth-email/send", json=body or VALID_BODY, headers=headers)


def test_valid_request_sends_and_returns_204(send_mock):
    res = _post()
    assert res.status_code == 204
    send_mock.assert_called_once()
    to, subject, body = send_mock.call_args.args
    assert to == "user@example.com"
    assert "Reset your AutoGPT Platform password" == subject
    assert VALID_BODY["url"] in body


def test_missing_token_is_401(send_mock):
    res = _post(token=None)
    assert res.status_code == 401
    send_mock.assert_not_called()


def test_wrong_token_is_401(send_mock):
    res = _post(token="not-the-secret")
    assert res.status_code == 401
    send_mock.assert_not_called()


def test_endpoint_disabled_when_secret_unset(send_mock, monkeypatch):
    monkeypatch.setattr(auth_email_routes.settings.secrets, "auth_email_token", "")
    res = _post()
    assert res.status_code == 503
    send_mock.assert_not_called()


def test_rejects_link_on_untrusted_host(send_mock):
    res = _post({**VALID_BODY, "url": "https://evil.example.com/reset"})
    assert res.status_code == 400
    send_mock.assert_not_called()


def test_rejects_non_https_link(send_mock):
    res = _post({**VALID_BODY, "url": "http://platform.agpt.co/reset"})
    assert res.status_code == 400
    send_mock.assert_not_called()


def test_allows_vercel_preview_link(send_mock):
    res = _post(
        {**VALID_BODY, "url": "https://autogpt-pr-13330.vercel.app/reset-password"}
    )
    assert res.status_code == 204
    send_mock.assert_called_once()


def test_rejects_unknown_type(send_mock):
    res = _post({**VALID_BODY, "type": "spam_blast"})
    assert res.status_code == 422  # pydantic Literal rejection
    send_mock.assert_not_called()
