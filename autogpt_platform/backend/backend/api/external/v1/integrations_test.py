"""External v1 credential creation rejects blank secrets (#15727)."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import fastapi
import fastapi.testclient
import pytest
from prisma.enums import APIKeyPermission

import backend.api.external.v1.integrations as integrations_mod
from backend.api.external.middleware import require_auth
from backend.api.external.v1.integrations import integrations_router
from backend.data.auth.base import APIAuthorizationInfo
from backend.data.model import (
    APIKeyCredentials,
    HostScopedCredentials,
    UserPasswordCredentials,
)

app = fastapi.FastAPI()
app.include_router(integrations_router)
client = fastapi.testclient.TestClient(app)


@pytest.fixture(autouse=True)
def setup_auth():
    async def fake_require_auth() -> APIAuthorizationInfo:
        return APIAuthorizationInfo(
            user_id="test-user",
            scopes=list(APIKeyPermission),
            type="api_key",
            created_at=datetime.now(timezone.utc),
        )

    app.dependency_overrides[require_auth] = fake_require_auth
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def create(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    mock = AsyncMock()
    monkeypatch.setattr(integrations_mod.creds_manager, "create", mock)
    return mock


@pytest.mark.parametrize("api_key", ["", "   ", "\t\n"])
def test_blank_api_key_rejected(create: AsyncMock, api_key: str):
    resp = client.post(
        "/integrations/openai/credentials",
        json={"type": "api_key", "title": "K", "api_key": api_key},
    )

    assert resp.status_code == 422
    assert resp.json()["detail"] == "API key must not be empty"
    create.assert_not_called()


def test_padded_api_key_stored_stripped(create: AsyncMock):
    resp = client.post(
        "/integrations/openai/credentials",
        json={"type": "api_key", "title": "K", "api_key": " sk-abc \n"},
    )

    assert resp.status_code == 201
    stored = create.await_args.args[1]
    assert isinstance(stored, APIKeyCredentials)
    assert stored.api_key.get_secret_value() == "sk-abc"


def test_blank_username_rejected_without_echoing_password(create: AsyncMock):
    resp = client.post(
        "/integrations/openai/credentials",
        json={
            "type": "user_password",
            "title": "U",
            "username": " ",
            "password": "hunter2-secret",
        },
    )

    assert resp.status_code == 422
    assert resp.json()["detail"] == "Username must not be empty"
    assert "hunter2-secret" not in resp.text
    create.assert_not_called()


def test_username_stripped_password_untouched(create: AsyncMock):
    resp = client.post(
        "/integrations/openai/credentials",
        json={
            "type": "user_password",
            "title": "U",
            "username": " alice ",
            "password": " pw ",
        },
    )

    assert resp.status_code == 201
    stored = create.await_args.args[1]
    assert isinstance(stored, UserPasswordCredentials)
    assert stored.username.get_secret_value() == "alice"
    assert stored.password.get_secret_value() == " pw "


@pytest.mark.parametrize(
    "host, headers",
    [
        (" ", {"Authorization": "Bearer x"}),
        ("api.example.com", {}),
        ("api.example.com", {"Authorization": " "}),
        ("api.example.com", {" ": "x"}),
    ],
)
def test_blank_host_scoped_rejected(create: AsyncMock, host: str, headers: dict):
    resp = client.post(
        "/integrations/http/credentials",
        json={"type": "host_scoped", "title": "H", "host": host, "headers": headers},
    )

    assert resp.status_code == 422
    create.assert_not_called()


def test_host_scoped_stripped(create: AsyncMock):
    resp = client.post(
        "/integrations/http/credentials",
        json={
            "type": "host_scoped",
            "title": "H",
            "host": " api.example.com ",
            "headers": {"Authorization": " Bearer x "},
        },
    )

    assert resp.status_code == 201
    stored = create.await_args.args[1]
    assert isinstance(stored, HostScopedCredentials)
    assert stored.host == "api.example.com"
    assert stored.get_headers_dict() == {"Authorization": "Bearer x"}
