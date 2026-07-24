import os
from unittest.mock import MagicMock

import fastapi
import fastapi.testclient
import pytest
from autogpt_libs.auth import config as auth_config
from autogpt_libs.auth.config import Settings as AuthSettings
from pytest_mock import MockerFixture

from backend.api.features.auth_email import routes as auth_email_routes

app = fastapi.FastAPI()
# Mirror the prefix rest_api.py mounts it under, so these tests exercise the
# same paths the service actually serves.
app.include_router(auth_email_routes.auth_email_router, prefix="/api/auth/email")
client = fastapi.testclient.TestClient(app, raise_server_exceptions=False)

VALID_BODY = {
    "type": "reset_password",
    "to": "user@example.com",
    "url": "https://platform.agpt.co/reset-password?token=abc",
}


@pytest.fixture
def send_mock(monkeypatch):
    """Bypass service-token auth (covered by autogpt_libs service_test) and
    capture the send forwarded to the notification service."""
    app.dependency_overrides[auth_email_routes.requires_auth_email_service] = (
        lambda: None
    )
    monkeypatch.setattr(
        auth_email_routes.settings.config,
        "frontend_base_url",
        "https://platform.agpt.co",
    )
    # Deterministic: no extra trusted origins unless a test sets them.
    monkeypatch.setattr(
        auth_email_routes.settings.config, "trusted_frontend_origins", []
    )
    mock = MagicMock()
    monkeypatch.setattr(
        auth_email_routes,
        "get_notification_manager_client",
        lambda: MagicMock(send_transactional_email=mock),
    )
    yield mock
    app.dependency_overrides.clear()


def _post(body=None):
    return client.post("/api/auth/email/send", json=body or VALID_BODY)


def test_valid_request_sends_and_returns_204(send_mock):
    res = _post()
    assert res.status_code == 204
    send_mock.assert_called_once()
    to, subject, body = send_mock.call_args.args
    assert to == "user@example.com"
    assert "Reset your AutoGPT Platform password" == subject
    assert VALID_BODY["url"] in body


def test_rejects_link_on_untrusted_host(send_mock):
    res = _post({**VALID_BODY, "url": "https://evil.example.com/reset"})
    assert res.status_code == 400
    send_mock.assert_not_called()


def test_rejects_scheme_mismatch(send_mock):
    # http origin != the trusted https frontend origin.
    res = _post({**VALID_BODY, "url": "http://platform.agpt.co/reset"})
    assert res.status_code == 400
    send_mock.assert_not_called()


def test_rejects_url_with_embedded_credentials(send_mock):
    # userinfo must not smuggle a different effective host past the check.
    res = _post(
        {**VALID_BODY, "url": "https://platform.agpt.co@evil.example.com/reset"}
    )
    assert res.status_code == 400
    send_mock.assert_not_called()


def test_rejects_arbitrary_vercel_host_without_config(send_mock):
    # No hardcoded *.vercel.app blanket anymore: an arbitrary vercel app is
    # rejected unless explicitly configured.
    res = _post({**VALID_BODY, "url": "https://attacker.vercel.app/reset"})
    assert res.status_code == 400
    send_mock.assert_not_called()


def test_allows_configured_preview_regex(send_mock, monkeypatch):
    # Cloud configures a tight preview pattern (not a blanket wildcard).
    monkeypatch.setattr(
        auth_email_routes.settings.config,
        "trusted_frontend_origins",
        [r"regex:https://autogpt-pr-\d+\.vercel\.app"],
    )
    res = _post(
        {**VALID_BODY, "url": "https://autogpt-pr-13330.vercel.app/reset-password"}
    )
    assert res.status_code == 204
    send_mock.assert_called_once()

    # A vercel host that doesn't match the pattern is still rejected.
    send_mock.reset_mock()
    res = _post({**VALID_BODY, "url": "https://autogpt-pr-13330.evil.app/x"})
    assert res.status_code == 400
    send_mock.assert_not_called()


def test_allows_exact_configured_origin(send_mock, monkeypatch):
    # Self-host style: an explicit extra origin (e.g. a custom domain).
    monkeypatch.setattr(
        auth_email_routes.settings.config,
        "trusted_frontend_origins",
        ["https://app.selfhosted.example:8443"],
    )
    res = _post(
        {**VALID_BODY, "url": "https://app.selfhosted.example:8443/reset?token=x"}
    )
    assert res.status_code == 204
    send_mock.assert_called_once()


def test_rejects_unknown_type(send_mock):
    res = _post({**VALID_BODY, "type": "spam_blast"})
    assert res.status_code == 422  # pydantic Literal rejection
    send_mock.assert_not_called()


def test_unauthenticated_request_is_401(mocker: MockerFixture, monkeypatch):
    """Without a frontend service token the route must refuse — proves the
    Security dependency is actually wired to the endpoint."""
    mocker.patch.dict(
        os.environ,
        {"JWT_JWKS_URL": "http://localhost:3000/api/auth/jwks"},
        clear=False,
    )
    mocker.patch.object(auth_config, "_settings", AuthSettings())
    send = MagicMock()
    monkeypatch.setattr(
        auth_email_routes,
        "get_notification_manager_client",
        lambda: MagicMock(send_transactional_email=send),
    )

    res = _post()

    assert res.status_code == 401
    send.assert_not_called()
