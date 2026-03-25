"""Tests for credentials API security: no secret leakage, SDK defaults filtered."""

from unittest.mock import AsyncMock, patch

import fastapi
import fastapi.testclient
import pytest
from pydantic import SecretStr

from backend.api.features.integrations.router import router
from backend.data.model import (
    APIKeyCredentials,
    HostScopedCredentials,
    OAuth2Credentials,
    UserPasswordCredentials,
)

app = fastapi.FastAPI()
app.include_router(router)
client = fastapi.testclient.TestClient(app)

TEST_USER_ID = "test-user-id"


def _make_api_key_cred(cred_id: str = "cred-123", provider: str = "openai"):
    return APIKeyCredentials(
        id=cred_id,
        provider=provider,
        title="My API Key",
        api_key=SecretStr("sk-secret-key-value"),
    )


def _make_oauth2_cred(cred_id: str = "cred-456", provider: str = "github"):
    return OAuth2Credentials(
        id=cred_id,
        provider=provider,
        title="My OAuth",
        access_token=SecretStr("ghp_secret_token"),
        refresh_token=SecretStr("ghp_refresh_secret"),
        scopes=["repo", "user"],
        username="testuser",
    )


def _make_user_password_cred(cred_id: str = "cred-789", provider: str = "openai"):
    return UserPasswordCredentials(
        id=cred_id,
        provider=provider,
        title="My Login",
        username=SecretStr("admin"),
        password=SecretStr("s3cret-pass"),
    )


def _make_host_scoped_cred(cred_id: str = "cred-host", provider: str = "openai"):
    return HostScopedCredentials(
        id=cred_id,
        provider=provider,
        title="Host Cred",
        host="https://api.example.com",
        headers={"Authorization": SecretStr("Bearer top-secret")},
    )


def _make_sdk_default_cred(provider: str = "openai"):
    return APIKeyCredentials(
        id=f"{provider}-default",
        provider=provider,
        title=f"{provider} (default)",
        api_key=SecretStr("sk-platform-secret-key"),
    )


@pytest.fixture(autouse=True)
def setup_auth(mock_jwt_user):
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


class TestGetCredentialReturnsMetaOnly:
    """GET /{provider}/credentials/{cred_id} must not return secrets."""

    def test_api_key_credential_no_secret(self):
        cred = _make_api_key_cred()
        with (
            patch.object(router, "dependencies", []),
            patch("backend.api.features.integrations.router.creds_manager") as mock_mgr,
        ):
            mock_mgr.get = AsyncMock(return_value=cred)
            resp = client.get("/openai/credentials/cred-123")

        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "cred-123"
        assert data["provider"] == "openai"
        assert data["type"] == "api_key"
        assert "api_key" not in data
        assert "sk-secret-key-value" not in str(data)

    def test_oauth2_credential_no_secret(self):
        cred = _make_oauth2_cred()
        with patch(
            "backend.api.features.integrations.router.creds_manager"
        ) as mock_mgr:
            mock_mgr.get = AsyncMock(return_value=cred)
            resp = client.get("/github/credentials/cred-456")

        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "cred-456"
        assert data["scopes"] == ["repo", "user"]
        assert data["username"] == "testuser"
        assert "access_token" not in data
        assert "refresh_token" not in data
        assert "ghp_" not in str(data)

    def test_user_password_credential_no_secret(self):
        cred = _make_user_password_cred()
        with patch(
            "backend.api.features.integrations.router.creds_manager"
        ) as mock_mgr:
            mock_mgr.get = AsyncMock(return_value=cred)
            resp = client.get("/openai/credentials/cred-789")

        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "cred-789"
        assert "password" not in data
        assert "username" not in data or data["username"] is None
        assert "s3cret-pass" not in str(data)
        assert "admin" not in str(data)

    def test_host_scoped_credential_no_secret(self):
        cred = _make_host_scoped_cred()
        with patch(
            "backend.api.features.integrations.router.creds_manager"
        ) as mock_mgr:
            mock_mgr.get = AsyncMock(return_value=cred)
            resp = client.get("/openai/credentials/cred-host")

        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "cred-host"
        assert data["host"] == "https://api.example.com"
        assert "headers" not in data
        assert "top-secret" not in str(data)

    def test_get_credential_wrong_provider_returns_404(self):
        """Provider mismatch should return generic 404, not leak credential existence."""
        cred = _make_api_key_cred(provider="openai")
        with patch(
            "backend.api.features.integrations.router.creds_manager"
        ) as mock_mgr:
            mock_mgr.get = AsyncMock(return_value=cred)
            resp = client.get("/github/credentials/cred-123")

        assert resp.status_code == 404
        assert resp.json()["detail"] == "Credentials not found"

    def test_list_credentials_no_secrets(self):
        """List endpoint must not leak secrets in any credential."""
        creds = [_make_api_key_cred(), _make_oauth2_cred()]
        with patch(
            "backend.api.features.integrations.router.creds_manager"
        ) as mock_mgr:
            mock_mgr.store.get_all_creds = AsyncMock(return_value=creds)
            resp = client.get("/credentials")

        assert resp.status_code == 200
        raw = str(resp.json())
        assert "sk-secret-key-value" not in raw
        assert "ghp_secret_token" not in raw
        assert "ghp_refresh_secret" not in raw


class TestSdkDefaultCredentialsNotAccessible:
    """SDK default credentials (ID ending in '-default') must be hidden."""

    def test_get_sdk_default_returns_404(self):
        with patch(
            "backend.api.features.integrations.router.creds_manager"
        ) as mock_mgr:
            mock_mgr.get = AsyncMock()
            resp = client.get("/openai/credentials/openai-default")

        assert resp.status_code == 404
        mock_mgr.get.assert_not_called()

    def test_list_credentials_excludes_sdk_defaults(self):
        user_cred = _make_api_key_cred()
        sdk_cred = _make_sdk_default_cred("openai")
        with patch(
            "backend.api.features.integrations.router.creds_manager"
        ) as mock_mgr:
            mock_mgr.store.get_all_creds = AsyncMock(return_value=[user_cred, sdk_cred])
            resp = client.get("/credentials")

        assert resp.status_code == 200
        data = resp.json()
        ids = [c["id"] for c in data]
        assert "cred-123" in ids
        assert "openai-default" not in ids

    def test_list_by_provider_excludes_sdk_defaults(self):
        user_cred = _make_api_key_cred()
        sdk_cred = _make_sdk_default_cred("openai")
        with patch(
            "backend.api.features.integrations.router.creds_manager"
        ) as mock_mgr:
            mock_mgr.store.get_creds_by_provider = AsyncMock(
                return_value=[user_cred, sdk_cred]
            )
            resp = client.get("/openai/credentials")

        assert resp.status_code == 200
        data = resp.json()
        ids = [c["id"] for c in data]
        assert "cred-123" in ids
        assert "openai-default" not in ids

    def test_delete_sdk_default_returns_404(self):
        with patch(
            "backend.api.features.integrations.router.creds_manager"
        ) as mock_mgr:
            mock_mgr.store.get_creds_by_id = AsyncMock()
            resp = client.request("DELETE", "/openai/credentials/openai-default")

        assert resp.status_code == 404
        mock_mgr.store.get_creds_by_id.assert_not_called()


class TestCreateCredentialNoSecretInResponse:
    """POST /{provider}/credentials must not return secrets."""

    def test_create_api_key_no_secret_in_response(self):
        with patch(
            "backend.api.features.integrations.router.creds_manager"
        ) as mock_mgr:
            mock_mgr.create = AsyncMock()
            resp = client.post(
                "/openai/credentials",
                json={
                    "id": "new-cred",
                    "provider": "openai",
                    "type": "api_key",
                    "title": "New Key",
                    "api_key": "sk-newsecret",
                },
            )

        assert resp.status_code == 201
        data = resp.json()
        assert data["id"] == "new-cred"
        assert "api_key" not in data
        assert "sk-newsecret" not in str(data)

    def test_create_with_sdk_default_id_rejected(self):
        with patch(
            "backend.api.features.integrations.router.creds_manager"
        ) as mock_mgr:
            mock_mgr.create = AsyncMock()
            resp = client.post(
                "/openai/credentials",
                json={
                    "id": "openai-default",
                    "provider": "openai",
                    "type": "api_key",
                    "title": "Sneaky",
                    "api_key": "sk-evil",
                },
            )

        assert resp.status_code == 403
        mock_mgr.create.assert_not_called()
