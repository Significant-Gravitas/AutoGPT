"""Tests for credentials API security: no secret leakage, SDK defaults filtered."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import fastapi
import fastapi.testclient
import pytest
from pydantic import SecretStr

from backend.api.features.integrations.router import router
from backend.data.integrations import Webhook
from backend.data.model import (
    APIKeyCredentials,
    HostScopedCredentials,
    OAuth2Credentials,
    UserPasswordCredentials,
)
from backend.integrations.providers import ProviderName
from backend.util.exceptions import NotFoundError

app = fastapi.FastAPI()
app.include_router(router)


@app.exception_handler(NotFoundError)
async def _not_found_handler(
    request: fastapi.Request, exc: NotFoundError
) -> fastapi.responses.JSONResponse:
    """Mirror the production NotFoundError → 404 mapping from the REST app."""
    return fastapi.responses.JSONResponse(status_code=404, content={"detail": str(exc)})


client = fastapi.testclient.TestClient(app)

TEST_USER_ID = "test-user-id"
# The id the mock_jwt_user fixture authenticates as, i.e. what the endpoints
# must scope every store operation to.
JWT_USER_ID = "3e53486c-cf57-477e-ba2a-cb02dc828e1a"


def _make_webhook(
    webhook_id: str = "wh-123",
    user_id: str = TEST_USER_ID,
    provider: str = "github",
    credentials_id: str = "cred-456",
) -> Webhook:
    return Webhook(
        id=webhook_id,
        user_id=user_id,
        provider=ProviderName(provider),
        credentials_id=credentials_id,
        webhook_type="repo",
        resource="owner/repo",
        events=["push"],
        config={},
        secret="whsecret",
        provider_webhook_id="provider-wh-1",
    )


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


class TestManagedCredentials:
    """AutoGPT-managed credentials cannot be deleted by users."""

    def test_delete_is_managed_returns_403(self):
        cred = APIKeyCredentials(
            id="managed-cred-1",
            provider="agent_mail",
            title="AgentMail (managed by AutoGPT)",
            api_key=SecretStr("sk-managed-key"),
            is_managed=True,
        )
        with patch(
            "backend.api.features.integrations.router.creds_manager"
        ) as mock_mgr:
            mock_mgr.store.get_creds_by_id = AsyncMock(return_value=cred)
            resp = client.request("DELETE", "/agent_mail/credentials/managed-cred-1")

        assert resp.status_code == 403
        assert "AutoGPT-managed" in resp.json()["detail"]

    def test_list_credentials_includes_is_managed_field(self):
        managed = APIKeyCredentials(
            id="managed-1",
            provider="agent_mail",
            title="AgentMail (managed)",
            api_key=SecretStr("sk-key"),
            is_managed=True,
        )
        regular = APIKeyCredentials(
            id="regular-1",
            provider="openai",
            title="My Key",
            api_key=SecretStr("sk-key"),
        )
        with patch(
            "backend.api.features.integrations.router.creds_manager"
        ) as mock_mgr:
            mock_mgr.store.get_all_creds = AsyncMock(return_value=[managed, regular])
            resp = client.get("/credentials")

        assert resp.status_code == 200
        data = resp.json()
        managed_cred = next(c for c in data if c["id"] == "managed-1")
        regular_cred = next(c for c in data if c["id"] == "regular-1")
        assert managed_cred["is_managed"] is True
        assert regular_cred["is_managed"] is False


# ---------------------------------------------------------------------------
# Managed credential provisioning infrastructure
# ---------------------------------------------------------------------------


def _make_managed_cred(
    provider: str = "agent_mail", pod_id: str = "pod-abc"
) -> APIKeyCredentials:
    return APIKeyCredentials(
        id="managed-auto",
        provider=provider,
        title="AgentMail (managed by AutoGPT)",
        api_key=SecretStr("sk-pod-key"),
        is_managed=True,
        metadata={"pod_id": pod_id},
    )


def _make_store_mock(**kwargs) -> MagicMock:
    """Create a store mock with a working async ``locks()`` context manager."""

    @asynccontextmanager
    async def _noop_locked(key):
        yield

    locks_obj = MagicMock()
    locks_obj.locked = _noop_locked

    store = MagicMock(**kwargs)
    store.locks = AsyncMock(return_value=locks_obj)
    return store


class TestEnsureManagedCredentials:
    """Unit tests for the ensure/cleanup helpers in managed_credentials.py."""

    @pytest.mark.asyncio
    async def test_provisions_when_missing(self):
        """Provider.provision() is called when no managed credential exists."""
        from backend.integrations.managed_credentials import (
            _PROVIDERS,
            _provisioned_users,
            ensure_managed_credentials,
        )

        cred = _make_managed_cred()
        provider = MagicMock()
        provider.provider_name = "test_provider"
        provider.is_available = AsyncMock(return_value=True)
        provider.provision = AsyncMock(return_value=cred)

        store = _make_store_mock()
        store.has_managed_credential = AsyncMock(return_value=False)
        store.add_managed_credential = AsyncMock()

        saved = dict(_PROVIDERS)
        _PROVIDERS.clear()
        _PROVIDERS["test_provider"] = provider
        _provisioned_users.pop("user-1", None)
        try:
            await ensure_managed_credentials("user-1", store)
        finally:
            _PROVIDERS.clear()
            _PROVIDERS.update(saved)
            _provisioned_users.pop("user-1", None)

        provider.provision.assert_awaited_once_with("user-1", store)
        store.add_managed_credential.assert_awaited_once_with("user-1", cred)

    @pytest.mark.asyncio
    async def test_skips_when_already_exists(self):
        """Provider.provision() is NOT called when managed credential exists."""
        from backend.integrations.managed_credentials import (
            _PROVIDERS,
            _provisioned_users,
            ensure_managed_credentials,
        )

        provider = MagicMock()
        provider.provider_name = "test_provider"
        provider.is_available = AsyncMock(return_value=True)
        provider.provision = AsyncMock()

        store = _make_store_mock()
        store.has_managed_credential = AsyncMock(return_value=True)

        saved = dict(_PROVIDERS)
        _PROVIDERS.clear()
        _PROVIDERS["test_provider"] = provider
        _provisioned_users.pop("user-1", None)
        try:
            await ensure_managed_credentials("user-1", store)
        finally:
            _PROVIDERS.clear()
            _PROVIDERS.update(saved)
            _provisioned_users.pop("user-1", None)

        provider.provision.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_when_unavailable(self):
        """Provider.provision() is NOT called when provider is not available."""
        from backend.integrations.managed_credentials import (
            _PROVIDERS,
            _provisioned_users,
            ensure_managed_credentials,
        )

        provider = MagicMock()
        provider.provider_name = "test_provider"
        provider.is_available = AsyncMock(return_value=False)
        provider.provision = AsyncMock()

        store = _make_store_mock()
        store.has_managed_credential = AsyncMock()

        saved = dict(_PROVIDERS)
        _PROVIDERS.clear()
        _PROVIDERS["test_provider"] = provider
        _provisioned_users.pop("user-1", None)
        try:
            await ensure_managed_credentials("user-1", store)
        finally:
            _PROVIDERS.clear()
            _PROVIDERS.update(saved)
            _provisioned_users.pop("user-1", None)

        provider.provision.assert_not_awaited()
        store.has_managed_credential.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_provision_failure_does_not_propagate(self):
        """A failed provision is logged but does not raise."""
        from backend.integrations.managed_credentials import (
            _PROVIDERS,
            _provisioned_users,
            ensure_managed_credentials,
        )

        provider = MagicMock()
        provider.provider_name = "test_provider"
        provider.is_available = AsyncMock(return_value=True)
        provider.provision = AsyncMock(side_effect=RuntimeError("boom"))

        store = _make_store_mock()
        store.has_managed_credential = AsyncMock(return_value=False)

        saved = dict(_PROVIDERS)
        _PROVIDERS.clear()
        _PROVIDERS["test_provider"] = provider
        _provisioned_users.pop("user-1", None)
        try:
            await ensure_managed_credentials("user-1", store)
        finally:
            _PROVIDERS.clear()
            _PROVIDERS.update(saved)
            _provisioned_users.pop("user-1", None)

        # No exception raised — provisioning failure is swallowed.


class TestCleanupManagedCredentials:
    """Unit tests for cleanup_managed_credentials."""

    @pytest.mark.asyncio
    async def test_calls_deprovision_for_managed_creds(self):
        from backend.integrations.managed_credentials import (
            _PROVIDERS,
            cleanup_managed_credentials,
        )

        cred = _make_managed_cred()
        provider = MagicMock()
        provider.provider_name = "agent_mail"
        provider.deprovision = AsyncMock()

        store = MagicMock()
        store.get_all_creds = AsyncMock(return_value=[cred])

        saved = dict(_PROVIDERS)
        _PROVIDERS.clear()
        _PROVIDERS["agent_mail"] = provider
        try:
            await cleanup_managed_credentials("user-1", store)
        finally:
            _PROVIDERS.clear()
            _PROVIDERS.update(saved)

        provider.deprovision.assert_awaited_once_with("user-1", cred)

    @pytest.mark.asyncio
    async def test_skips_non_managed_creds(self):
        from backend.integrations.managed_credentials import (
            _PROVIDERS,
            cleanup_managed_credentials,
        )

        regular = _make_api_key_cred()
        provider = MagicMock()
        provider.provider_name = "openai"
        provider.deprovision = AsyncMock()

        store = MagicMock()
        store.get_all_creds = AsyncMock(return_value=[regular])

        saved = dict(_PROVIDERS)
        _PROVIDERS.clear()
        _PROVIDERS["openai"] = provider
        try:
            await cleanup_managed_credentials("user-1", store)
        finally:
            _PROVIDERS.clear()
            _PROVIDERS.update(saved)

        provider.deprovision.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_deprovision_failure_does_not_propagate(self):
        from backend.integrations.managed_credentials import (
            _PROVIDERS,
            cleanup_managed_credentials,
        )

        cred = _make_managed_cred()
        provider = MagicMock()
        provider.provider_name = "agent_mail"
        provider.deprovision = AsyncMock(side_effect=RuntimeError("boom"))

        store = MagicMock()
        store.get_all_creds = AsyncMock(return_value=[cred])

        saved = dict(_PROVIDERS)
        _PROVIDERS.clear()
        _PROVIDERS["agent_mail"] = provider
        try:
            await cleanup_managed_credentials("user-1", store)
        finally:
            _PROVIDERS.clear()
            _PROVIDERS.update(saved)

        # No exception raised — cleanup failure is swallowed.


class TestGetPickerToken:
    """POST /{provider}/credentials/{cred_id}/picker-token must:
    1. Return the access token for OAuth2 creds the caller owns.
    2. 404 for non-owned, non-existent, or wrong-provider creds.
    3. 400 for non-OAuth2 creds (API key, host-scoped, user/password).
    4. 404 for SDK default creds (same hardening as get_credential).
    5. Preserve the `TestGetCredentialReturnsMetaOnly` contract — the
       existing meta-only endpoint must still strip secrets even after
       this picker-token endpoint exists."""

    def test_oauth2_owner_gets_access_token(self):
        # Use a Google cred with a drive.file scope — only picker-eligible
        # (provider, scope) pairs can mint a token. GitHub-style creds are
        # explicitly rejected; see `test_non_picker_provider_rejected_as_400`.
        cred = _make_oauth2_cred(
            cred_id="cred-gdrive",
            provider="google",
        )
        cred.scopes = ["https://www.googleapis.com/auth/drive.file"]
        with patch(
            "backend.api.features.integrations.router.creds_manager"
        ) as mock_mgr:
            mock_mgr.get = AsyncMock(return_value=cred)
            resp = client.post("/google/credentials/cred-gdrive/picker-token")

        assert resp.status_code == 200
        data = resp.json()
        # The whole point of this endpoint: the access token IS returned here.
        assert data["access_token"] == "ghp_secret_token"
        # Only the two declared fields come back — nothing else leaks.
        assert set(data.keys()) <= {"access_token", "access_token_expires_at"}

    def test_non_picker_provider_rejected_as_400(self):
        """Provider allowlist: even with a valid OAuth2 credential, a
        non-picker provider (GitHub, etc.) cannot mint a picker token.
        Stops this endpoint from being used as a generic bearer-token
        extraction path for any stored OAuth cred under the same user."""
        cred = _make_oauth2_cred(provider="github")
        with patch(
            "backend.api.features.integrations.router.creds_manager"
        ) as mock_mgr:
            mock_mgr.get = AsyncMock(return_value=cred)
            resp = client.post("/github/credentials/cred-456/picker-token")

        assert resp.status_code == 400
        assert "not available for provider" in resp.json()["detail"]
        assert "ghp_secret_token" not in str(resp.json())

    def test_google_oauth_without_drive_scope_rejected(self):
        """Scope allowlist: a Google OAuth2 cred that only carries non-picker
        scopes (e.g. gmail.readonly, calendar) cannot mint a picker token.
        Forces the frontend to reconnect with a Drive scope before the
        picker is available."""
        cred = _make_oauth2_cred(provider="google")
        cred.scopes = [
            "https://www.googleapis.com/auth/gmail.readonly",
            "https://www.googleapis.com/auth/calendar",
        ]
        with patch(
            "backend.api.features.integrations.router.creds_manager"
        ) as mock_mgr:
            mock_mgr.get = AsyncMock(return_value=cred)
            resp = client.post("/google/credentials/cred-456/picker-token")

        assert resp.status_code == 400
        assert "picker" in resp.json()["detail"].lower()

    def test_api_key_credential_rejected_as_400(self):
        cred = _make_api_key_cred()
        with patch(
            "backend.api.features.integrations.router.creds_manager"
        ) as mock_mgr:
            mock_mgr.get = AsyncMock(return_value=cred)
            resp = client.post("/openai/credentials/cred-123/picker-token")

        assert resp.status_code == 400
        # API keys must not silently fall through to a 200 response of some
        # other shape — the client should see a clear shape rejection.
        body = str(resp.json())
        assert "sk-secret-key-value" not in body

    def test_user_password_credential_rejected_as_400(self):
        cred = _make_user_password_cred()
        with patch(
            "backend.api.features.integrations.router.creds_manager"
        ) as mock_mgr:
            mock_mgr.get = AsyncMock(return_value=cred)
            resp = client.post("/openai/credentials/cred-789/picker-token")

        assert resp.status_code == 400
        body = str(resp.json())
        assert "s3cret-pass" not in body
        assert "admin" not in body

    def test_host_scoped_credential_rejected_as_400(self):
        cred = _make_host_scoped_cred()
        with patch(
            "backend.api.features.integrations.router.creds_manager"
        ) as mock_mgr:
            mock_mgr.get = AsyncMock(return_value=cred)
            resp = client.post("/openai/credentials/cred-host/picker-token")

        assert resp.status_code == 400
        assert "top-secret" not in str(resp.json())

    def test_missing_credential_returns_404(self):
        with patch(
            "backend.api.features.integrations.router.creds_manager"
        ) as mock_mgr:
            mock_mgr.get = AsyncMock(return_value=None)
            resp = client.post("/github/credentials/nonexistent/picker-token")

        assert resp.status_code == 404
        assert resp.json()["detail"] == "Credentials not found"

    def test_wrong_provider_returns_404(self):
        """Symmetric with get_credential: provider mismatch is a generic
        404, not a 400, so we don't leak existence of a credential the
        caller doesn't own on that provider."""
        cred = _make_oauth2_cred(provider="github")
        with patch(
            "backend.api.features.integrations.router.creds_manager"
        ) as mock_mgr:
            mock_mgr.get = AsyncMock(return_value=cred)
            resp = client.post("/google/credentials/cred-456/picker-token")

        assert resp.status_code == 404
        assert resp.json()["detail"] == "Credentials not found"

    def test_sdk_default_returns_404(self):
        """SDK defaults are invisible to the user-facing API — picker-token
        must not mint a token for them either."""
        with patch(
            "backend.api.features.integrations.router.creds_manager"
        ) as mock_mgr:
            mock_mgr.get = AsyncMock()
            resp = client.post("/openai/credentials/openai-default/picker-token")

        assert resp.status_code == 404
        mock_mgr.get.assert_not_called()

    def test_oauth2_without_access_token_returns_400(self):
        """A stored OAuth2 cred whose access_token is missing can't satisfy
        a picker init. Surface a clear reconnect instruction rather than
        returning an empty string."""
        cred = _make_oauth2_cred()
        # Simulate a cred that lost its access token
        object.__setattr__(cred, "access_token", None)

        with patch(
            "backend.api.features.integrations.router.creds_manager"
        ) as mock_mgr:
            mock_mgr.get = AsyncMock(return_value=cred)
            resp = client.post("/github/credentials/cred-456/picker-token")

        assert resp.status_code == 400
        assert "reconnect" in resp.json()["detail"].lower()

    def test_meta_only_endpoint_still_strips_access_token(self):
        """Regression guard for the coexistence contract: the new
        picker-token endpoint must NOT accidentally leak the token through
        the meta-only GET endpoint. TestGetCredentialReturnsMetaOnly
        covers this more broadly; this is a fast sanity check co-located
        with the new endpoint's tests."""
        cred = _make_oauth2_cred()
        with patch(
            "backend.api.features.integrations.router.creds_manager"
        ) as mock_mgr:
            mock_mgr.get = AsyncMock(return_value=cred)
            resp = client.get("/github/credentials/cred-456")

        assert resp.status_code == 200
        body = resp.json()
        assert "access_token" not in body
        assert "refresh_token" not in body
        assert "ghp_secret_token" not in str(body)


class TestWebhookPingOwnership:
    """POST /webhooks/{webhook_id}/ping must verify the caller owns the webhook.

    Regression guard for SECRT-2434: the endpoint fetched the webhook by
    primary key alone (`get_webhook` is documented unsafe for user-facing
    use) and never compared `webhook.user_id` to the caller. That let any
    authenticated user enumerate webhook IDs and trigger pings on other
    users' webhooks (IDOR / broken object-level authorization).

    A foreign webhook must be indistinguishable from a non-existent one
    (both 404), so the endpoint can't be used as an existence oracle.
    """

    def test_foreign_webhook_returns_404_without_pinging(self):
        webhook = _make_webhook(user_id="someone-else")
        with (
            patch(
                "backend.api.features.integrations.router.get_webhook",
                AsyncMock(return_value=webhook),
            ),
            patch(
                "backend.api.features.integrations.router.get_webhook_manager"
            ) as mock_get_mgr,
            patch(
                "backend.api.features.integrations.router.creds_manager"
            ) as mock_creds,
        ):
            resp = client.post("/webhooks/wh-123/ping")

        assert resp.status_code == 404
        # Ownership check must short-circuit before any side effects.
        mock_get_mgr.assert_not_called()
        mock_creds.get.assert_not_called()

    def test_nonexistent_webhook_returns_404(self):
        with patch(
            "backend.api.features.integrations.router.get_webhook",
            AsyncMock(side_effect=NotFoundError("Webhook #wh-x not found")),
        ):
            resp = client.post("/webhooks/wh-x/ping")

        assert resp.status_code == 404

    def test_owned_webhook_pings(self, test_user_id):
        webhook = _make_webhook(user_id=test_user_id)
        webhook_manager = MagicMock()
        webhook_manager.trigger_ping = AsyncMock()
        with (
            patch(
                "backend.api.features.integrations.router.get_webhook",
                AsyncMock(return_value=webhook),
            ),
            patch(
                "backend.api.features.integrations.router.get_webhook_manager",
                return_value=webhook_manager,
            ),
            patch(
                "backend.api.features.integrations.router.creds_manager"
            ) as mock_creds,
            patch(
                "backend.api.features.integrations.router.wait_for_webhook_event",
                AsyncMock(return_value=True),
            ),
        ):
            mock_creds.get = AsyncMock(return_value=None)
            resp = client.post("/webhooks/wh-123/ping")

        assert resp.status_code == 200
        assert resp.json() is True
        webhook_manager.trigger_ping.assert_awaited_once()


class TestDeviceAuthEndpoints:
    """POST /{provider}/device-auth/{initiate,poll}.

    The concurrency contract here is load-bearing: `peek` must survive many
    polls while `consume` must admit exactly one terminal handler, or an
    approval produces duplicate credentials.
    """

    PROVIDER = "stripe_link"

    @staticmethod
    def _initiation():
        from backend.integrations.oauth.device_base import DeviceAuthInitiation

        return DeviceAuthInitiation(
            device_code="dev-code-secret",
            user_code="GLOW-RELISH",
            verification_url="https://login.link.com/device",
            verification_url_complete="https://login.link.com/device?code=GLOW-RELISH",
            expires_in=600,
            interval=5,
        )

    @staticmethod
    def _state(metadata: dict | None = None):
        from backend.data.model import OAuthState

        return OAuthState(
            token="state-token",
            provider="stripe_link",
            expires_at=9999999999,
            scopes=["userinfo:read"],
            state_metadata=(
                metadata
                if metadata is not None
                else {"flow_type": "device_code", "device_code": "dev-code-secret"}
            ),
        )

    def _patched(self, handler, mock_mgr):
        return (
            patch.object(router, "dependencies", []),
            patch(
                "backend.api.features.integrations.router._get_device_auth_handler",
                return_value=handler,
            ),
            patch("backend.api.features.integrations.router.creds_manager", mock_mgr),
            # Off by default. Every test in this class drives the same user and
            # provider within the same second, so with a live Redis the
            # throttle would answer for whichever endpoint ran second. It has
            # its own tests below.
            patch(
                "backend.api.features.integrations.router._throttle_upstream",
                new=AsyncMock(return_value=False),
            ),
        )

    def test_initiate_does_not_return_the_device_code(self):
        """The device code is the bearer secret of the flow.

        It is stored encrypted server-side and the browser never needs it, so
        it must not appear anywhere in the response.
        """
        handler = MagicMock()
        handler.handle_default_scopes.return_value = ["userinfo:read"]
        handler.initiate_device_auth = AsyncMock(return_value=self._initiation())

        mock_mgr = MagicMock()
        mock_mgr.store.store_state_token = AsyncMock(
            return_value=("state-token", "verifier")
        )

        p1, p2, p3, p4 = self._patched(handler, mock_mgr)
        with p1, p2, p3, p4:
            response = client.post(
                f"/{self.PROVIDER}/device-auth/initiate?scopes=userinfo:read"
            )

        assert response.status_code == 200
        body = response.json()
        assert body["user_code"] == "GLOW-RELISH"
        assert body["state_token"] == "state-token"
        assert "device_code" not in body
        assert "dev-code-secret" not in response.text

    def test_initiate_outlives_the_provider_code(self):
        """State must expire after the device code, not before.

        Otherwise the poll loop dies with "invalid state token" while the user
        can still legitimately approve on their phone.
        """
        handler = MagicMock()
        handler.handle_default_scopes.return_value = ["userinfo:read"]
        handler.initiate_device_auth = AsyncMock(return_value=self._initiation())

        mock_mgr = MagicMock()
        mock_mgr.store.store_state_token = AsyncMock(
            return_value=("state-token", "verifier")
        )

        p1, p2, p3, p4 = self._patched(handler, mock_mgr)
        with p1, p2, p3, p4:
            client.post(f"/{self.PROVIDER}/device-auth/initiate")

        kwargs = mock_mgr.store.store_state_token.await_args.kwargs
        assert kwargs["expires_in_seconds"] > 600
        assert kwargs["state_metadata"]["device_code"] == "dev-code-secret"

    def test_initiate_does_not_leak_upstream_error_text(self):
        handler = MagicMock()
        handler.handle_default_scopes.return_value = []
        handler.initiate_device_auth = AsyncMock(
            side_effect=RuntimeError("401 {'secret_hint': 'client_id lwlpk_xyz'}")
        )

        p1, p2, p3, p4 = self._patched(handler, MagicMock())
        with p1, p2, p3, p4:
            response = client.post(f"/{self.PROVIDER}/device-auth/initiate")

        assert response.status_code == 502
        assert "lwlpk_xyz" not in response.text

    def test_poll_rejects_an_unknown_state_token(self):
        handler = MagicMock()
        mock_mgr = MagicMock()
        mock_mgr.store.peek_state_token = AsyncMock(return_value=None)

        p1, p2, p3, p4 = self._patched(handler, mock_mgr)
        with p1, p2, p3, p4:
            response = client.post(
                f"/{self.PROVIDER}/device-auth/poll",
                json={"state_token": "not-a-real-token"},
            )

        assert response.status_code == 400

    def test_poll_rejects_a_non_device_flow_state_token(self):
        """A plain OAuth state token has no device code and must not be usable."""
        handler = MagicMock()
        mock_mgr = MagicMock()
        mock_mgr.store.peek_state_token = AsyncMock(return_value=self._state({}))

        p1, p2, p3, p4 = self._patched(handler, mock_mgr)
        with p1, p2, p3, p4:
            response = client.post(
                f"/{self.PROVIDER}/device-auth/poll",
                json={"state_token": "state-token"},
            )

        assert response.status_code == 400

    def test_pending_poll_does_not_consume_the_state_token(self):
        """`peek` is non-consuming so the loop can poll for the full 10 minutes."""
        from backend.integrations.oauth.device_base import DeviceAuthPollResult

        handler = MagicMock()
        handler.poll_for_tokens = AsyncMock(
            return_value=DeviceAuthPollResult(status="pending")
        )

        mock_mgr = MagicMock()
        mock_mgr.store.peek_state_token = AsyncMock(return_value=self._state())
        mock_mgr.store.consume_state_token = AsyncMock()

        p1, p2, p3, p4 = self._patched(handler, mock_mgr)
        with p1, p2, p3, p4:
            for _ in range(3):
                response = client.post(
                    f"/{self.PROVIDER}/device-auth/poll",
                    json={"state_token": "state-token"},
                )
                assert response.status_code == 200
                assert response.json()["status"] == "pending"

        mock_mgr.store.consume_state_token.assert_not_awaited()

    def test_poll_race_loser_does_not_create_a_second_credential(self):
        """Two polls can both see an approval; only the one that consumes wins.

        `consume_state_token` returning None means another poll already handled
        this terminal state -- storing credentials anyway would duplicate them
        for a single authorization.
        """
        from backend.data.model import OAuth2Credentials
        from backend.integrations.oauth.device_base import DeviceAuthPollResult

        handler = MagicMock()
        handler.handle_default_scopes.side_effect = lambda scopes: scopes
        handler.poll_for_tokens = AsyncMock(
            return_value=DeviceAuthPollResult(
                status="approved",
                credentials=OAuth2Credentials(
                    provider="stripe_link",
                    access_token=SecretStr("at"),
                    refresh_token=SecretStr("rt"),
                    scopes=["userinfo:read"],
                    title="Stripe Link",
                ),
            )
        )

        mock_mgr = MagicMock()
        mock_mgr.store.peek_state_token = AsyncMock(return_value=self._state())
        mock_mgr.store.consume_state_token = AsyncMock(return_value=None)

        with (
            patch.object(router, "dependencies", []),
            patch(
                "backend.api.features.integrations.router._get_device_auth_handler",
                return_value=handler,
            ),
            patch("backend.api.features.integrations.router.creds_manager", mock_mgr),
            patch(
                "backend.api.features.integrations.router._throttle_upstream",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "backend.api.features.integrations.router._merge_or_create_credential",
                new=AsyncMock(),
            ) as mock_merge,
        ):
            response = client.post(
                f"/{self.PROVIDER}/device-auth/poll",
                json={"state_token": "state-token"},
            )

        assert response.status_code == 200
        # The winner handled this grant, so the loser must not store a second
        # credential. It reports the approval without one — a first-time grant
        # has no credential id to look the stored one up by, and answering
        # `pending` would send the client back to a state token the winner has
        # already consumed, where the next poll 400s.
        assert response.json()["status"] == "approved"
        assert response.json()["credentials"] is None
        mock_merge.assert_not_awaited()

    def test_poll_does_not_leak_upstream_error_text(self):
        handler = MagicMock()
        handler.poll_for_tokens = AsyncMock(
            side_effect=RuntimeError("500 {'trace': 'internal-host-1.link.com'}")
        )

        mock_mgr = MagicMock()
        mock_mgr.store.peek_state_token = AsyncMock(return_value=self._state())

        p1, p2, p3, p4 = self._patched(handler, mock_mgr)
        with p1, p2, p3, p4:
            response = client.post(
                f"/{self.PROVIDER}/device-auth/poll",
                json={"state_token": "state-token"},
            )

        assert response.status_code == 502
        assert "internal-host-1" not in response.text

    def test_poll_stores_the_credential_on_the_winning_approval(self):
        """The primary success path: consume wins, credential is stored.

        Every other approved-status test here makes `consume_state_token`
        return None to exercise the race loser, so the path that actually
        stores a credential -- and the scope un-flatten below it -- had no
        coverage at all.
        """
        from backend.data.model import OAuth2Credentials
        from backend.integrations.oauth.device_base import DeviceAuthPollResult

        stored = OAuth2Credentials(
            provider="stripe_link",
            access_token=SecretStr("tok-access-do-not-leak"),
            refresh_token=SecretStr("tok-refresh-do-not-leak"),
            # Link returns RFC 6749 space-delimited scopes; the router
            # un-flattens them so scope checks compare like with like.
            scopes=["userinfo:read payment_methods.agentic"],
            title="Stripe Link",
        )
        handler = MagicMock()
        handler.handle_default_scopes.side_effect = lambda scopes: scopes
        handler.poll_for_tokens = AsyncMock(
            return_value=DeviceAuthPollResult(status="approved", credentials=stored)
        )

        mock_mgr = MagicMock()
        mock_mgr.store.peek_state_token = AsyncMock(return_value=self._state())
        mock_mgr.store.consume_state_token = AsyncMock(return_value=self._state())

        with (
            patch.object(router, "dependencies", []),
            patch(
                "backend.api.features.integrations.router._get_device_auth_handler",
                return_value=handler,
            ),
            patch("backend.api.features.integrations.router.creds_manager", mock_mgr),
            patch(
                "backend.api.features.integrations.router._throttle_upstream",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "backend.api.features.integrations.router._merge_or_create_credential",
                new=AsyncMock(side_effect=lambda *a, **kw: a[2]),
            ) as mock_merge,
        ):
            response = client.post(
                f"/{self.PROVIDER}/device-auth/poll",
                json={"state_token": "state-token"},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "approved"
        assert body["credentials"] is not None
        mock_merge.assert_awaited()
        # The space-delimited scope string is split before storage.
        assert stored.scopes == ["userinfo:read", "payment_methods.agentic"]
        # And no secret rides along in the response.
        assert "do-not-leak" not in response.text

    def test_poll_maps_a_missing_user_row_to_an_invalid_token(self):
        """A caller with no backend User row raises a Prisma RecordNotFound in
        `peek_state_token`. That used to surface as a 500 carrying raw DB text,
        inconsistent with `initiate`, which already returns a clean message."""
        handler = MagicMock()
        mock_mgr = MagicMock()
        mock_mgr.store.peek_state_token = AsyncMock(
            side_effect=RuntimeError(
                "An operation failed because it depends on one or more records "
                "that were required but not found. Expected a record, found none."
            )
        )

        p1, p2, p3, p4 = self._patched(handler, mock_mgr)
        with p1, p2, p3, p4:
            response = client.post(
                f"/{self.PROVIDER}/device-auth/poll",
                json={"state_token": "state-token"},
            )

        assert response.status_code == 400
        assert "Expected a record" not in response.text

    def test_every_state_operation_is_scoped_to_the_authenticated_user(self):
        """User scoping is the whole authorization control on these endpoints.

        Every other test here hands `creds_manager` a bare MagicMock and never
        checks which user id reached the store, so a regression that passed a
        constant — or dropped the argument — would leave the suite green.
        """
        from backend.integrations.oauth.device_base import DeviceAuthPollResult

        handler = MagicMock()
        handler.handle_default_scopes.return_value = ["userinfo:read"]
        handler.initiate_device_auth = AsyncMock(return_value=self._initiation())
        handler.poll_for_tokens = AsyncMock(
            return_value=DeviceAuthPollResult(status="pending")
        )

        mock_mgr = MagicMock()
        mock_mgr.store.store_state_token = AsyncMock(return_value=("tok", "v"))
        mock_mgr.store.peek_state_token = AsyncMock(return_value=self._state())

        p1, p2, p3, p4 = self._patched(handler, mock_mgr)
        with p1, p2, p3, p4:
            client.post(f"/{self.PROVIDER}/device-auth/initiate")
            client.post(
                f"/{self.PROVIDER}/device-auth/poll",
                json={"state_token": "state-token"},
            )

        authenticated = mock_mgr.store.store_state_token.await_args.kwargs["user_id"]
        assert authenticated == JWT_USER_ID
        # peek is positional: (user_id, token, provider)
        assert mock_mgr.store.peek_state_token.await_args.args[0] == JWT_USER_ID

    def test_one_user_cannot_poll_another_users_state_token(self):
        """The store scopes by user, so a token belonging to someone else reads
        as absent rather than as someone else's flow."""
        handler = MagicMock()
        mock_mgr = MagicMock()
        # What the store returns for a token that is not this user's.
        mock_mgr.store.peek_state_token = AsyncMock(return_value=None)

        p1, p2, p3, p4 = self._patched(handler, mock_mgr)
        with p1, p2, p3, p4:
            response = client.post(
                f"/{self.PROVIDER}/device-auth/poll",
                json={"state_token": "another-users-token"},
            )

        assert response.status_code == 400
        assert mock_mgr.store.peek_state_token.await_args.args[0] == JWT_USER_ID
        # And the provider is never contacted for a token we could not place.
        handler.poll_for_tokens.assert_not_called()

    def test_the_throttle_holds_callers_to_the_provider_interval(self):
        """Every call drives the provider under one public client id shared by
        the whole platform, so a single looping account can get Stripe Link
        connect throttled for everyone. Covered here rather than in the other
        tests, which switch it off so they can drive the same user twice."""
        from backend.integrations.oauth.device_base import DeviceAuthPollResult

        handler = MagicMock()
        handler.poll_for_tokens = AsyncMock(
            return_value=DeviceAuthPollResult(status="pending")
        )
        mock_mgr = MagicMock()
        mock_mgr.store.peek_state_token = AsyncMock(return_value=self._state())

        with (
            patch.object(router, "dependencies", []),
            patch(
                "backend.api.features.integrations.router._get_device_auth_handler",
                return_value=handler,
            ),
            patch("backend.api.features.integrations.router.creds_manager", mock_mgr),
            patch(
                "backend.api.features.integrations.router._throttle_upstream",
                new=AsyncMock(return_value=True),
            ),
        ):
            poll = client.post(
                f"/{self.PROVIDER}/device-auth/poll",
                json={"state_token": "state-token"},
            )
            initiate = client.post(f"/{self.PROVIDER}/device-auth/initiate")

        # RFC 8628's own vocabulary, and the provider is never contacted.
        assert poll.status_code == 200
        assert poll.json()["status"] == "slow_down"
        handler.poll_for_tokens.assert_not_called()

        assert initiate.status_code == 429

    async def test_the_throttle_fails_open_when_redis_is_unreachable(self):
        """It protects an upstream client id; it must never be the reason a
        request fails or hangs."""
        import backend.api.features.integrations.router as router_module

        with patch(
            "backend.data.redis_client.get_redis_async",
            new=AsyncMock(side_effect=ConnectionError("redis down")),
        ):
            throttled = await router_module._throttle_upstream(
                TEST_USER_ID, ProviderName.STRIPE_LINK, 5, scope="poll", flow="f"
            )

        assert throttled is False

    async def test_initiate_and_poll_do_not_share_a_throttle_key(self):
        """A live poll loop must not lock out starting a new flow.

        Both endpoints claimed one key, and poll re-claims its key every
        `interval` seconds with the same TTL — so the key never expired and
        initiate returned 429 for as long as any flow was open. Leaving a
        dialog open while approving on a phone is the intended behaviour, and
        it blocked every other surface, including cancel-and-retry.
        """
        import backend.api.features.integrations.router as router_module

        claimed: dict[str, int] = {}

        class _Redis:
            async def set(self, key, value, ex=None, nx=False):
                if nx and key in claimed:
                    return None
                claimed[key] = ex
                return True

        with patch(
            "backend.data.redis_client.get_redis_async",
            new=AsyncMock(return_value=_Redis()),
        ):
            first_poll = await router_module._throttle_upstream(
                TEST_USER_ID, ProviderName.STRIPE_LINK, 5, scope="poll", flow="flow-a"
            )
            # A poll loop is live; starting a flow must still be allowed.
            initiate = await router_module._throttle_upstream(
                TEST_USER_ID, ProviderName.STRIPE_LINK, 3, scope="initiate"
            )
            # And each scope still throttles itself.
            second_poll = await router_module._throttle_upstream(
                TEST_USER_ID, ProviderName.STRIPE_LINK, 5, scope="poll", flow="flow-a"
            )
            # A *different* flow must not be starved by the first one's window.
            other_flow = await router_module._throttle_upstream(
                TEST_USER_ID, ProviderName.STRIPE_LINK, 5, scope="poll", flow="flow-b"
            )

        assert first_poll is False
        assert initiate is False, "an active poll loop blocked initiate"
        assert second_poll is True
        assert other_flow is False, "a second concurrent flow was starved"
        assert len(claimed) == 3, claimed

    async def test_an_unstorable_grant_is_revoked_not_left_live(self):
        """The device code is spent by the time the store runs, so the state
        token cannot be replayed to retry. Handing the authorization back is
        the only way to avoid a live grant with no local record of it."""
        from backend.data.model import OAuth2Credentials
        from backend.integrations.oauth.device_base import DeviceAuthPollResult

        issued = OAuth2Credentials(
            provider="stripe_link",
            access_token=SecretStr("at"),
            refresh_token=SecretStr("rt"),
            scopes=["userinfo:read"],
            title="Stripe Link",
        )
        handler = MagicMock()
        handler.handle_default_scopes.side_effect = lambda scopes: scopes
        handler.poll_for_tokens = AsyncMock(
            return_value=DeviceAuthPollResult(status="approved", credentials=issued)
        )
        handler.revoke_tokens = AsyncMock(return_value=True)

        mock_mgr = MagicMock()
        mock_mgr.store.peek_state_token = AsyncMock(return_value=self._state())
        mock_mgr.store.consume_state_token = AsyncMock(return_value=self._state())

        with (
            patch.object(router, "dependencies", []),
            patch(
                "backend.api.features.integrations.router._get_device_auth_handler",
                return_value=handler,
            ),
            patch("backend.api.features.integrations.router.creds_manager", mock_mgr),
            patch(
                "backend.api.features.integrations.router._throttle_upstream",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "backend.api.features.integrations.router._merge_or_create_credential",
                new=AsyncMock(side_effect=RuntimeError("db down")),
            ),
        ):
            with pytest.raises(Exception):
                client.post(
                    f"/{self.PROVIDER}/device-auth/poll",
                    json={"state_token": "state-token"},
                )

        handler.revoke_tokens.assert_awaited_once_with(issued)

    def test_the_race_loser_reports_this_grants_credential_not_the_newest(self):
        """`get_creds_by_provider` has no ordering contract, and a merged
        re-auth keeps its position — so picking the last entry could hand the
        connect modal a different wallet than the user approved."""
        from backend.integrations.oauth.device_base import DeviceAuthPollResult

        handler = MagicMock()
        handler.poll_for_tokens = AsyncMock(
            return_value=DeviceAuthPollResult(status="approved")
        )

        state = self._state()
        state.credential_id = "the-right-wallet"

        mock_mgr = MagicMock()
        mock_mgr.store.peek_state_token = AsyncMock(return_value=state)
        mock_mgr.store.consume_state_token = AsyncMock(return_value=None)
        mock_mgr.store.get_creds_by_id = AsyncMock(
            return_value=_make_oauth2_cred("the-right-wallet", "stripe_link")
        )

        p1, p2, p3, p4 = self._patched(handler, mock_mgr)
        with p1, p2, p3, p4:
            response = client.post(
                f"/{self.PROVIDER}/device-auth/poll",
                json={"state_token": "state-token"},
            )

        assert response.status_code == 200
        mock_mgr.store.get_creds_by_id.assert_awaited_once_with(
            JWT_USER_ID, "the-right-wallet"
        )
        # And never the "newest for this provider" shortcut.
        mock_mgr.store.get_creds_by_provider.assert_not_called()
