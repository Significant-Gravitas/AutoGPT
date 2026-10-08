import os
import re
from html import unescape
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
        lambda: MagicMock(send_email_or_raise=mock),
    )
    yield mock
    app.dependency_overrides.clear()


def _post(body=None):
    return client.post("/api/auth/email/send", json=body or VALID_BODY)


def test_valid_request_sends_and_returns_204(send_mock):
    res = _post()
    assert res.status_code == 204
    send_mock.assert_called_once()
    to, subject, body, text = send_mock.call_args.args
    assert to == "user@example.com"
    assert "Reset your AutoGPT Platform password" == subject
    assert VALID_BODY["url"] in body
    assert VALID_BODY["url"] in text


def test_set_password_email_says_why_it_came_and_when_it_expires(send_mock):
    res = _post({**VALID_BODY, "type": "set_password"})

    assert res.status_code == 204
    _, subject, body, text = send_mock.call_args.args
    assert subject == "Set your AutoGPT Platform password"
    for part in (body, text):
        assert "asked to sign up for, or verify" in part
        assert "set a password and finish signing up" in part
        assert "expires in 1 hour" in part


def test_set_password_email_warns_an_earlier_password_no_longer_works(send_mock):
    # Someone who resubmits the sign-up form gets this email, and the password
    # they signed up with stops working, even if they finish through the
    # first link.
    _post({**VALID_BODY, "type": "set_password"})

    _, _, body, text = send_mock.call_args.args
    assert "the password you chose then no longer works" in body
    assert "&#34;Forgot password&#34;" in body
    assert '"Forgot password"' in text


def test_reset_email_carries_no_set_password_note(send_mock):
    _post(VALID_BODY)

    _, _, body, text = send_mock.call_args.args
    assert "no longer works" not in body
    assert "no longer works" not in text


def test_verify_email_states_its_24_hour_expiry(send_mock):
    _post({**VALID_BODY, "type": "verify_email"})

    _, _, body, text = send_mock.call_args.args
    assert "expires in 24 hours" in body
    assert "expires in 24 hours" in text


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
        lambda: MagicMock(send_email_or_raise=send),
    )

    res = _post()

    assert res.status_code == 401
    send.assert_not_called()


def test_allows_origin_with_explicit_default_port(send_mock, monkeypatch):
    """An explicit :443 must not read as a different origin than the bare host.

    urlparse keeps an explicit default port, so without normalization the
    comparison fails closed and a legitimate reset email is never sent.
    """
    monkeypatch.setattr(
        auth_email_routes.settings.config,
        "trusted_frontend_origins",
        ["https://app.selfhosted.example"],
    )
    res = _post(
        {**VALID_BODY, "url": "https://app.selfhosted.example:443/reset?token=x"}
    )
    assert res.status_code == 204
    send_mock.assert_called_once()


def test_allows_configured_origin_carrying_a_default_port(send_mock, monkeypatch):
    """...and the same when the explicit port is on the configured side."""
    monkeypatch.setattr(
        auth_email_routes.settings.config,
        "trusted_frontend_origins",
        ["https://app.selfhosted.example:443"],
    )
    res = _post({**VALID_BODY, "url": "https://app.selfhosted.example/reset?token=x"})
    assert res.status_code == 204
    send_mock.assert_called_once()


def test_non_default_port_still_distinguishes_origins(send_mock, monkeypatch):
    """Normalizing default ports must not collapse genuinely different ones."""
    monkeypatch.setattr(
        auth_email_routes.settings.config,
        "trusted_frontend_origins",
        ["https://app.selfhosted.example:8443"],
    )
    res = _post({**VALID_BODY, "url": "https://app.selfhosted.example:9443/reset"})
    assert res.status_code == 400
    send_mock.assert_not_called()


def test_rejects_malformed_trusted_origin_regex_at_startup():
    """A bad pattern must fail at config time, not on every email send."""
    import pytest

    from backend.util.settings import Config

    with pytest.raises(ValueError, match="Invalid regex pattern"):
        Config(trusted_frontend_origins=["regex:https://(unclosed"])

    with pytest.raises(ValueError, match="cannot be empty"):
        Config(trusted_frontend_origins=["regex:"])


@pytest.mark.parametrize(
    "email_type,subject,button",
    [
        ("reset_password", "Reset your AutoGPT Platform password", "Reset password"),
        ("verify_email", "Verify your AutoGPT Platform email", "Verify email"),
        ("change_email", "Confirm your new AutoGPT Platform email", "Confirm email"),
        ("set_password", "Set your AutoGPT Platform password", "Set password"),
    ],
)
def test_auth_email_design_and_plain_text(send_mock, email_type, subject, button):
    url = 'https://platform.agpt.co/verify?token=a&callback="<script>alert(1)</script>'
    result = _post({**VALID_BODY, "type": email_type, "url": url})
    assert result.status_code == 204
    to, actual_subject, body, text = send_mock.call_args.args
    assert to == VALID_BODY["to"]
    assert actual_subject == subject
    assert body.count('bgcolor="#6144DF"') == 1
    assert 'width="560"' in body
    assert "font-family:Geist," in body
    assert 'alt="AutoGPT"' in body
    assert 'width="100" height="45"' in body
    assert all(
        'role="presentation"' in table and 'bgcolor="' in table
        for table in re.findall(r"<table\b[^>]*>", body)
    )
    assert all(
        src.startswith("https://")
        for src in re.findall(r'<img[^>]+src="([^"]+)"', body)
    )
    assert "<svg" not in body and "data:image" not in body
    assert "logo-light.png" in body
    assert "Otto, your personal Head of AI" not in body
    assert "autogpt-characters/" not in body
    assert "<script>" not in body
    assert "&amp;callback=&#34;&lt;script&gt;" in body
    assert url in text
    assert [unescape(href) for href in re.findall(r'href="([^"]+)"', body)].count(
        url
    ) == 2
    assert all(
        "—" not in part and "–" not in part for part in (actual_subject, body, text)
    )
    assert button in body and button in text
    assert "3rd Floor, 1 Ashley Road, Altrincham, WA14 2DT, UK" in body
    assert "3rd Floor, 1 Ashley Road, Altrincham, WA14 2DT, UK" in text
    assert "This is a service message about an AutoGPT account." in body
    assert "This is a service message about an AutoGPT account." in text
    assert "Unsubscribe" not in body
    assert "Email preferences" not in body
    assert "didn't request this" in text
    assert len(body.encode()) < 60_000
