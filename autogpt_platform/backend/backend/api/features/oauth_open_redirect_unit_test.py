"""Focused unit tests for OAuth authorize open-redirect hardening (#15047)."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from prisma.enums import APIKeyPermission

from backend.api.features.oauth import AuthorizeRequest, _error_redirect_url, authorize
from backend.data.auth.oauth import OAuthApplicationInfo


def _make_app(
    *, redirect_uris: list[str], is_active: bool = True
) -> OAuthApplicationInfo:
    now = datetime.now(timezone.utc)
    return OAuthApplicationInfo(
        id="app-1",
        name="Test App",
        description=None,
        logo_url=None,
        client_id="client-1",
        redirect_uris=redirect_uris,
        grant_types=["authorization_code"],
        scopes=[APIKeyPermission.EXECUTE_GRAPH],
        owner_id="user-1",
        is_active=is_active,
        created_at=now,
        updated_at=now,
    )


EVIL = "https://evil.example/phish"
GOOD = "https://example.com/callback"


def test_error_redirect_url_rejects_unregistered_uri():
    app = _make_app(redirect_uris=[GOOD])
    with pytest.raises(HTTPException) as exc_info:
        _error_redirect_url(app, EVIL, "state", "invalid_scope", "bad")
    assert exc_info.value.status_code == 400
    assert EVIL not in str(exc_info.value.detail)


def test_error_redirect_url_allows_registered_uri():
    app = _make_app(redirect_uris=[GOOD])
    result = _error_redirect_url(app, GOOD, "state", "invalid_scope", "bad")
    assert result.redirect_url.startswith(GOOD)
    assert EVIL not in result.redirect_url
    assert "error=invalid_scope" in result.redirect_url


@pytest.mark.asyncio
async def test_authorize_unknown_client_no_evil_redirect(monkeypatch):
    monkeypatch.setattr(
        "backend.api.features.oauth.get_oauth_application",
        AsyncMock(return_value=None),
    )
    request = AuthorizeRequest(
        client_id="unknown",
        redirect_uri=EVIL,
        scopes=["EXECUTE_GRAPH"],
        state="s",
        response_type="code",
        code_challenge="challenge",
        code_challenge_method="S256",
    )
    with pytest.raises(HTTPException) as exc_info:
        await authorize(request=request, user_id="user-1")
    assert exc_info.value.status_code == 400
    assert EVIL not in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_authorize_inactive_client_no_evil_redirect(monkeypatch):
    app = _make_app(redirect_uris=[GOOD], is_active=False)
    monkeypatch.setattr(
        "backend.api.features.oauth.get_oauth_application",
        AsyncMock(return_value=app),
    )
    request = AuthorizeRequest(
        client_id="client-1",
        redirect_uri=EVIL,
        scopes=["EXECUTE_GRAPH"],
        state="s",
        response_type="code",
        code_challenge="challenge",
        code_challenge_method="S256",
    )
    with pytest.raises(HTTPException) as exc_info:
        await authorize(request=request, user_id="user-1")
    assert exc_info.value.status_code == 400
    assert EVIL not in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_authorize_unsupported_response_type_with_evil_uri(monkeypatch):
    app = _make_app(redirect_uris=[GOOD])
    monkeypatch.setattr(
        "backend.api.features.oauth.get_oauth_application",
        AsyncMock(return_value=app),
    )
    request = AuthorizeRequest(
        client_id="client-1",
        redirect_uri=EVIL,
        scopes=["EXECUTE_GRAPH"],
        state="s",
        response_type="token",
        code_challenge="challenge",
        code_challenge_method="S256",
    )
    with pytest.raises(HTTPException) as exc_info:
        await authorize(request=request, user_id="user-1")
    assert exc_info.value.status_code == 400
    assert EVIL not in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_authorize_server_error_never_uses_unvalidated_uri(monkeypatch):
    """If an unexpected error happens after validation, only registered URI is used."""
    app = _make_app(redirect_uris=[GOOD])
    monkeypatch.setattr(
        "backend.api.features.oauth.get_oauth_application",
        AsyncMock(return_value=app),
    )
    monkeypatch.setattr(
        "backend.api.features.oauth.create_authorization_code",
        AsyncMock(side_effect=RuntimeError("boom")),
    )
    request = AuthorizeRequest(
        client_id="client-1",
        redirect_uri=GOOD,
        scopes=["EXECUTE_GRAPH"],
        state="s",
        response_type="code",
        code_challenge="challenge",
        code_challenge_method="S256",
    )
    result = await authorize(request=request, user_id="user-1")
    assert result.redirect_url.startswith(GOOD)
    assert "error=server_error" in result.redirect_url
    assert EVIL not in result.redirect_url
