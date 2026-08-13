"""Tests for incremental OAuth authorization (scope upgrade)."""

from unittest.mock import AsyncMock, MagicMock, patch

import fastapi
import fastapi.testclient
import pytest
from pydantic import SecretStr

from backend.api.features.integrations.router import router
from backend.data.model import APIKeyCredentials, OAuth2Credentials, OAuthState

app = fastapi.FastAPI()
app.include_router(router)
client = fastapi.testclient.TestClient(app)

TEST_USER_ID = "test-user-id"


def _make_google_oauth2_cred(
    cred_id: str = "google-cred-1",
    scopes: list[str] | None = None,
    username: str = "alice@gmail.com",
    title: str = "My Google",
) -> OAuth2Credentials:
    return OAuth2Credentials(
        id=cred_id,
        provider="google",
        title=title,
        access_token=SecretStr("ya29.access-token"),
        refresh_token=SecretStr("1//refresh-token"),
        scopes=(
            scopes
            if scopes is not None
            else ["https://www.googleapis.com/auth/gmail.readonly"]
        ),
        username=username,
        access_token_expires_at=9999999999,
    )


def _make_github_oauth2_cred(
    cred_id: str = "github-cred-1",
    scopes: list[str] | None = None,
    username: str = "alice",
    title: str = "My GitHub",
) -> OAuth2Credentials:
    return OAuth2Credentials(
        id=cred_id,
        provider="github",
        title=title,
        access_token=SecretStr("ghp_access_token"),
        refresh_token=SecretStr("ghp_refresh_token"),
        scopes=scopes if scopes is not None else ["repo"],
        username=username,
    )


@pytest.fixture(autouse=True)
def setup_auth(mock_jwt_user):
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


# ==================== OAuthState model tests ==================== #


class TestOAuthStateCredentialId:
    """OAuthState model should support a credential_id field for upgrades."""

    def test_oauth_state_accepts_credential_id(self):
        state = OAuthState(
            token="abc",
            provider="google",
            expires_at=9999999999,
            scopes=["openid"],
            credential_id="existing-cred-id",
        )
        assert state.credential_id == "existing-cred-id"

    def test_oauth_state_defaults_credential_id_none(self):
        state = OAuthState(
            token="abc",
            provider="google",
            expires_at=9999999999,
            scopes=["openid"],
        )
        assert state.credential_id is None


# ==================== Login endpoint tests ==================== #


class TestIncrementalOAuthLogin:
    """Tests for the login endpoint with credential_id parameter."""

    def test_login_with_credential_id_stores_in_state(self):
        """Login with credential_id should pass it through to store_state_token."""
        existing = _make_google_oauth2_cred()
        handler = MagicMock()
        handler.get_login_url.return_value = "https://accounts.google.com/auth"

        with (
            patch(
                "backend.api.features.integrations.router._get_provider_oauth_handler",
                return_value=handler,
            ),
            patch("backend.api.features.integrations.router.creds_manager") as mock_mgr,
        ):
            mock_mgr.store.get_creds_by_id = AsyncMock(return_value=existing)
            mock_mgr.store.store_state_token = AsyncMock(
                return_value=("state-token", "code-challenge")
            )

            resp = client.get(
                "/google/login",
                params={
                    "scopes": "https://www.googleapis.com/auth/calendar.readonly",
                    "credential_id": "google-cred-1",
                },
            )

        assert resp.status_code == 200
        # Verify store_state_token was called with credential_id
        call_kwargs = mock_mgr.store.store_state_token.call_args
        assert call_kwargs.kwargs.get("credential_id") == "google-cred-1" or (
            len(call_kwargs.args) > 3 and call_kwargs.args[3] == "google-cred-1"
        )

    def test_login_github_unions_scopes_for_upgrade(self):
        """For GitHub, login should request union of existing + new scopes."""
        existing = _make_github_oauth2_cred(scopes=["repo"])
        handler = MagicMock()
        handler.get_login_url.return_value = "https://github.com/login/oauth/authorize"

        with (
            patch(
                "backend.api.features.integrations.router._get_provider_oauth_handler",
                return_value=handler,
            ),
            patch("backend.api.features.integrations.router.creds_manager") as mock_mgr,
        ):
            mock_mgr.store.get_creds_by_id = AsyncMock(return_value=existing)
            mock_mgr.store.store_state_token = AsyncMock(
                return_value=("state-token", "code-challenge")
            )

            resp = client.get(
                "/github/login",
                params={
                    "scopes": "read:org",
                    "credential_id": "github-cred-1",
                },
            )

        assert resp.status_code == 200
        # The scopes passed to get_login_url should be the union
        login_scopes = handler.get_login_url.call_args[0][0]
        assert set(login_scopes) == {"repo", "read:org"}

    def test_login_google_keeps_requested_scopes_only(self):
        """For Google, login should use only the new scopes (include_granted_scopes handles merging)."""
        existing = _make_google_oauth2_cred(
            scopes=["https://www.googleapis.com/auth/gmail.readonly"]
        )
        handler = MagicMock()
        handler.get_login_url.return_value = "https://accounts.google.com/auth"

        with (
            patch(
                "backend.api.features.integrations.router._get_provider_oauth_handler",
                return_value=handler,
            ),
            patch("backend.api.features.integrations.router.creds_manager") as mock_mgr,
        ):
            mock_mgr.store.get_creds_by_id = AsyncMock(return_value=existing)
            mock_mgr.store.store_state_token = AsyncMock(
                return_value=("state-token", "code-challenge")
            )

            resp = client.get(
                "/google/login",
                params={
                    "scopes": "https://www.googleapis.com/auth/calendar.readonly",
                    "credential_id": "google-cred-1",
                },
            )

        assert resp.status_code == 200
        login_scopes = handler.get_login_url.call_args[0][0]
        # Google should NOT union scopes in the login URL
        assert "https://www.googleapis.com/auth/calendar.readonly" in login_scopes
        assert "https://www.googleapis.com/auth/gmail.readonly" not in login_scopes
        # Verify credential_id was passed through to store_state_token
        call_kwargs = mock_mgr.store.store_state_token.call_args
        assert call_kwargs.kwargs.get("credential_id") == "google-cred-1"

    def test_login_credential_not_found_returns_404(self):
        handler = MagicMock()
        with (
            patch(
                "backend.api.features.integrations.router._get_provider_oauth_handler",
                return_value=handler,
            ),
            patch("backend.api.features.integrations.router.creds_manager") as mock_mgr,
        ):
            mock_mgr.store.get_creds_by_id = AsyncMock(return_value=None)

            resp = client.get(
                "/google/login",
                params={
                    "scopes": "openid",
                    "credential_id": "nonexistent",
                },
            )

        assert resp.status_code == 404

    def test_login_credential_provider_mismatch_returns_400(self):
        """credential_id pointing to a Google cred when URL says github -> 400."""
        google_cred = _make_google_oauth2_cred()
        handler = MagicMock()

        with (
            patch(
                "backend.api.features.integrations.router._get_provider_oauth_handler",
                return_value=handler,
            ),
            patch("backend.api.features.integrations.router.creds_manager") as mock_mgr,
        ):
            mock_mgr.store.get_creds_by_id = AsyncMock(return_value=google_cred)

            resp = client.get(
                "/github/login",
                params={
                    "scopes": "repo",
                    "credential_id": "google-cred-1",
                },
            )

        assert resp.status_code == 400

    def test_login_non_oauth2_credential_returns_400(self):
        """credential_id pointing to an API key credential -> 400."""
        api_key_cred = APIKeyCredentials(
            id="apikey-1",
            provider="github",
            title="API Key",
            api_key=SecretStr("ghp_key"),
        )
        handler = MagicMock()

        with (
            patch(
                "backend.api.features.integrations.router._get_provider_oauth_handler",
                return_value=handler,
            ),
            patch("backend.api.features.integrations.router.creds_manager") as mock_mgr,
        ):
            mock_mgr.store.get_creds_by_id = AsyncMock(return_value=api_key_cred)

            resp = client.get(
                "/github/login",
                params={
                    "scopes": "repo",
                    "credential_id": "apikey-1",
                },
            )

        assert resp.status_code == 400


# ==================== Callback endpoint tests ==================== #


class TestIncrementalOAuthCallback:
    """Tests for the callback endpoint when upgrading credentials."""

    def _make_state_with_credential_id(
        self,
        credential_id: str,
        scopes: list[str] | None = None,
        provider: str = "google",
    ) -> OAuthState:
        return OAuthState(
            token="state-token",
            provider=provider,
            expires_at=9999999999,
            scopes=(
                scopes
                if scopes is not None
                else ["https://www.googleapis.com/auth/calendar.readonly"]
            ),
            credential_id=credential_id,
        )

    def test_callback_upgrades_existing_credential(self):
        """When state has credential_id, should update existing credential."""
        existing = _make_google_oauth2_cred(
            scopes=["https://www.googleapis.com/auth/gmail.readonly"]
        )
        new_cred = _make_google_oauth2_cred(
            scopes=[
                "https://www.googleapis.com/auth/gmail.readonly",
                "https://www.googleapis.com/auth/calendar.readonly",
            ]
        )
        state = self._make_state_with_credential_id("google-cred-1")
        handler = MagicMock()
        handler.exchange_code_for_tokens = AsyncMock(return_value=new_cred)
        handler.handle_default_scopes.return_value = state.scopes

        with (
            patch(
                "backend.api.features.integrations.router._get_provider_oauth_handler",
                return_value=handler,
            ),
            patch("backend.api.features.integrations.router.creds_manager") as mock_mgr,
        ):
            mock_mgr.store.verify_state_token = AsyncMock(return_value=state)
            mock_mgr.store.get_creds_by_id = AsyncMock(return_value=existing)
            mock_mgr.update = AsyncMock()
            mock_mgr.create = AsyncMock()

            resp = client.post(
                "/google/callback",
                json={"code": "auth-code", "state_token": "state-token"},
            )

        assert resp.status_code == 200
        # Should call update, not create
        mock_mgr.update.assert_called_once()
        mock_mgr.create.assert_not_called()

    def test_callback_upgrade_merges_scopes(self):
        """Upgraded credential should have union of old + new scopes."""
        existing = _make_google_oauth2_cred(
            scopes=["https://www.googleapis.com/auth/gmail.readonly"]
        )
        new_cred = _make_google_oauth2_cred(
            scopes=[
                "https://www.googleapis.com/auth/gmail.readonly",
                "https://www.googleapis.com/auth/calendar.readonly",
            ]
        )
        state = self._make_state_with_credential_id("google-cred-1")
        handler = MagicMock()
        handler.exchange_code_for_tokens = AsyncMock(return_value=new_cred)
        handler.handle_default_scopes.return_value = state.scopes

        with (
            patch(
                "backend.api.features.integrations.router._get_provider_oauth_handler",
                return_value=handler,
            ),
            patch("backend.api.features.integrations.router.creds_manager") as mock_mgr,
        ):
            mock_mgr.store.verify_state_token = AsyncMock(return_value=state)
            mock_mgr.store.get_creds_by_id = AsyncMock(return_value=existing)
            mock_mgr.update = AsyncMock()

            resp = client.post(
                "/google/callback",
                json={"code": "auth-code", "state_token": "state-token"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert set(data["scopes"]) == {
            "https://www.googleapis.com/auth/gmail.readonly",
            "https://www.googleapis.com/auth/calendar.readonly",
        }

    def test_callback_upgrade_preserves_id_and_title(self):
        """Upgraded credential should keep its original ID and title."""
        existing = _make_google_oauth2_cred(
            cred_id="original-id", title="My Work Google"
        )
        new_cred = _make_google_oauth2_cred(cred_id="new-id-from-exchange")
        state = self._make_state_with_credential_id("original-id")
        handler = MagicMock()
        handler.exchange_code_for_tokens = AsyncMock(return_value=new_cred)
        handler.handle_default_scopes.return_value = state.scopes

        with (
            patch(
                "backend.api.features.integrations.router._get_provider_oauth_handler",
                return_value=handler,
            ),
            patch("backend.api.features.integrations.router.creds_manager") as mock_mgr,
        ):
            mock_mgr.store.verify_state_token = AsyncMock(return_value=state)
            mock_mgr.store.get_creds_by_id = AsyncMock(return_value=existing)
            mock_mgr.update = AsyncMock()

            resp = client.post(
                "/google/callback",
                json={"code": "auth-code", "state_token": "state-token"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "original-id"
        assert data["title"] == "My Work Google"

    def test_callback_upgrade_rejects_username_mismatch(self):
        """Should reject if the new auth returns a different username."""
        existing = _make_google_oauth2_cred(username="alice@gmail.com")
        new_cred = _make_google_oauth2_cred(username="bob@gmail.com")
        state = self._make_state_with_credential_id("google-cred-1")
        handler = MagicMock()
        handler.exchange_code_for_tokens = AsyncMock(return_value=new_cred)
        handler.handle_default_scopes.return_value = state.scopes

        with (
            patch(
                "backend.api.features.integrations.router._get_provider_oauth_handler",
                return_value=handler,
            ),
            patch("backend.api.features.integrations.router.creds_manager") as mock_mgr,
        ):
            mock_mgr.store.verify_state_token = AsyncMock(return_value=state)
            mock_mgr.store.get_creds_by_id = AsyncMock(return_value=existing)

            resp = client.post(
                "/google/callback",
                json={"code": "auth-code", "state_token": "state-token"},
            )

        assert resp.status_code == 400
        assert "username" in resp.json()["detail"].lower()

    def test_callback_implicit_merge_same_provider_username(self):
        """Without credential_id, should auto-merge when same provider+username exists."""
        existing = _make_google_oauth2_cred(
            scopes=["https://www.googleapis.com/auth/gmail.readonly"]
        )
        new_cred = _make_google_oauth2_cred(
            cred_id="new-cred-id",
            scopes=[
                "https://www.googleapis.com/auth/gmail.readonly",
                "https://www.googleapis.com/auth/calendar.readonly",
            ],
            username="alice@gmail.com",
        )
        # State WITHOUT credential_id
        state = OAuthState(
            token="state-token",
            provider="google",
            expires_at=9999999999,
            scopes=["https://www.googleapis.com/auth/calendar.readonly"],
        )
        handler = MagicMock()
        handler.exchange_code_for_tokens = AsyncMock(return_value=new_cred)
        handler.handle_default_scopes.return_value = state.scopes

        with (
            patch(
                "backend.api.features.integrations.router._get_provider_oauth_handler",
                return_value=handler,
            ),
            patch("backend.api.features.integrations.router.creds_manager") as mock_mgr,
        ):
            mock_mgr.store.verify_state_token = AsyncMock(return_value=state)
            mock_mgr.store.get_creds_by_provider = AsyncMock(return_value=[existing])
            mock_mgr.store.get_creds_by_id = AsyncMock(return_value=existing)
            mock_mgr.update = AsyncMock()
            mock_mgr.create = AsyncMock()

            resp = client.post(
                "/google/callback",
                json={"code": "auth-code", "state_token": "state-token"},
            )

        assert resp.status_code == 200
        # Should update the existing credential, not create a new one
        mock_mgr.update.assert_called_once()
        mock_mgr.create.assert_not_called()
        # The returned ID should be the existing credential's ID
        data = resp.json()
        assert data["id"] == "google-cred-1"

    def test_callback_no_implicit_merge_different_username(self):
        """Without credential_id, different username should create new credential."""
        existing = _make_google_oauth2_cred(username="alice@gmail.com")
        new_cred = _make_google_oauth2_cred(
            cred_id="new-cred-id",
            username="bob@gmail.com",
        )
        state = OAuthState(
            token="state-token",
            provider="google",
            expires_at=9999999999,
            scopes=["https://www.googleapis.com/auth/gmail.readonly"],
        )
        handler = MagicMock()
        handler.exchange_code_for_tokens = AsyncMock(return_value=new_cred)
        handler.handle_default_scopes.return_value = state.scopes

        with (
            patch(
                "backend.api.features.integrations.router._get_provider_oauth_handler",
                return_value=handler,
            ),
            patch("backend.api.features.integrations.router.creds_manager") as mock_mgr,
        ):
            mock_mgr.store.verify_state_token = AsyncMock(return_value=state)
            mock_mgr.store.get_creds_by_provider = AsyncMock(return_value=[existing])
            mock_mgr.create = AsyncMock()
            mock_mgr.update = AsyncMock()

            resp = client.post(
                "/google/callback",
                json={"code": "auth-code", "state_token": "state-token"},
            )

        assert resp.status_code == 200
        mock_mgr.create.assert_called_once()
        mock_mgr.update.assert_not_called()
        # Verify the implicit merge lookup was attempted
        mock_mgr.store.get_creds_by_provider.assert_called_once()

    def test_callback_creates_new_when_no_existing(self):
        """Without credential_id and no matching credential, creates new."""
        new_cred = _make_google_oauth2_cred()
        state = OAuthState(
            token="state-token",
            provider="google",
            expires_at=9999999999,
            scopes=["https://www.googleapis.com/auth/gmail.readonly"],
        )
        handler = MagicMock()
        handler.exchange_code_for_tokens = AsyncMock(return_value=new_cred)
        handler.handle_default_scopes.return_value = state.scopes

        with (
            patch(
                "backend.api.features.integrations.router._get_provider_oauth_handler",
                return_value=handler,
            ),
            patch("backend.api.features.integrations.router.creds_manager") as mock_mgr,
        ):
            mock_mgr.store.verify_state_token = AsyncMock(return_value=state)
            mock_mgr.store.get_creds_by_provider = AsyncMock(return_value=[])
            mock_mgr.create = AsyncMock()
            mock_mgr.update = AsyncMock()

            resp = client.post(
                "/google/callback",
                json={"code": "auth-code", "state_token": "state-token"},
            )

        assert resp.status_code == 200
        mock_mgr.create.assert_called_once()
        mock_mgr.update.assert_not_called()
        # Verify the implicit merge lookup was attempted
        mock_mgr.store.get_creds_by_provider.assert_called_once()


# ==================== Round 2: Review feedback tests ==================== #


class TestManagedCredentialProtection:
    """Managed/system credentials must not be upgradeable."""

    def test_login_rejects_managed_credential_id(self):
        """Explicit credential_id pointing to a managed credential -> 400."""
        managed = _make_google_oauth2_cred(cred_id="managed-1")
        managed.is_managed = True
        handler = MagicMock()

        with (
            patch(
                "backend.api.features.integrations.router._get_provider_oauth_handler",
                return_value=handler,
            ),
            patch("backend.api.features.integrations.router.creds_manager") as mock_mgr,
        ):
            mock_mgr.store.get_creds_by_id = AsyncMock(return_value=managed)

            resp = client.get(
                "/google/login",
                params={
                    "scopes": "https://www.googleapis.com/auth/calendar.readonly",
                    "credential_id": "managed-1",
                },
            )

        assert resp.status_code == 400

    def test_callback_rejects_upgrade_of_managed_credential(self):
        """Callback with credential_id for a managed credential -> 400."""
        managed = _make_google_oauth2_cred(cred_id="managed-1")
        managed.is_managed = True
        new_cred = _make_google_oauth2_cred()
        state = OAuthState(
            token="state-token",
            provider="google",
            expires_at=9999999999,
            scopes=["https://www.googleapis.com/auth/calendar.readonly"],
            credential_id="managed-1",
        )
        handler = MagicMock()
        handler.exchange_code_for_tokens = AsyncMock(return_value=new_cred)
        handler.handle_default_scopes.return_value = state.scopes

        with (
            patch(
                "backend.api.features.integrations.router._get_provider_oauth_handler",
                return_value=handler,
            ),
            patch("backend.api.features.integrations.router.creds_manager") as mock_mgr,
        ):
            mock_mgr.store.verify_state_token = AsyncMock(return_value=state)
            mock_mgr.store.get_creds_by_id = AsyncMock(return_value=managed)

            resp = client.post(
                "/google/callback",
                json={"code": "auth-code", "state_token": "state-token"},
            )

        assert resp.status_code == 400


class TestMetadataNoneGuard:
    """Metadata merge must handle None values."""

    def test_callback_upgrade_handles_none_metadata(self):
        """Upgrading credential with metadata=None should not crash."""
        existing = _make_google_oauth2_cred(
            scopes=["https://www.googleapis.com/auth/gmail.readonly"]
        )
        existing.metadata = None  # type: ignore[assignment]
        new_cred = _make_google_oauth2_cred(
            scopes=[
                "https://www.googleapis.com/auth/gmail.readonly",
                "https://www.googleapis.com/auth/calendar.readonly",
            ]
        )
        new_cred.metadata = None  # type: ignore[assignment]
        state = OAuthState(
            token="state-token",
            provider="google",
            expires_at=9999999999,
            scopes=["https://www.googleapis.com/auth/calendar.readonly"],
            credential_id="google-cred-1",
        )
        handler = MagicMock()
        handler.exchange_code_for_tokens = AsyncMock(return_value=new_cred)
        handler.handle_default_scopes.return_value = state.scopes

        with (
            patch(
                "backend.api.features.integrations.router._get_provider_oauth_handler",
                return_value=handler,
            ),
            patch("backend.api.features.integrations.router.creds_manager") as mock_mgr,
        ):
            mock_mgr.store.verify_state_token = AsyncMock(return_value=state)
            mock_mgr.store.get_creds_by_id = AsyncMock(return_value=existing)
            mock_mgr.update = AsyncMock()

            resp = client.post(
                "/google/callback",
                json={"code": "auth-code", "state_token": "state-token"},
            )

        assert resp.status_code == 200


class TestStateHelperScopesPattern:
    """Test helper should handle empty scopes correctly."""

    def test_make_state_preserves_empty_scopes(self):
        """_make_state_with_credential_id([]) should keep empty list."""
        state_maker = TestIncrementalOAuthCallback()
        state = state_maker._make_state_with_credential_id("cred-1", scopes=[])
        assert state.scopes == []


class TestSystemCredentialProtection:
    """Platform-owned system credentials must never be upgraded."""

    def test_login_rejects_system_credential_id(self):
        """Explicit credential_id pointing to a system credential -> 400."""
        handler = MagicMock()

        with (
            patch(
                "backend.api.features.integrations.router._get_provider_oauth_handler",
                return_value=handler,
            ),
            patch(
                "backend.api.features.integrations.router.is_system_credential",
                return_value=True,
            ),
            patch("backend.api.features.integrations.router.creds_manager") as mock_mgr,
        ):
            mock_mgr.store.get_creds_by_id = AsyncMock()

            resp = client.get(
                "/google/login",
                params={
                    "scopes": "https://www.googleapis.com/auth/calendar.readonly",
                    "credential_id": "system-cred-id",
                },
            )

        assert resp.status_code == 400
        assert "system credentials" in resp.json()["detail"].lower()
        # The store lookup must never happen for system credentials.
        mock_mgr.store.get_creds_by_id.assert_not_called()

    def test_callback_rejects_upgrade_of_system_credential(self):
        """Defense-in-depth: even if a stale login state points at a system
        credential, the callback-time `_upgrade_existing_credential` must
        reject it before persisting anything."""
        existing = _make_google_oauth2_cred(cred_id="sys-cred-id")
        new_cred = _make_google_oauth2_cred(
            scopes=[
                "https://www.googleapis.com/auth/gmail.readonly",
                "https://www.googleapis.com/auth/calendar.readonly",
            ]
        )
        state = OAuthState(
            token="state-token",
            provider="google",
            expires_at=9999999999,
            scopes=["https://www.googleapis.com/auth/calendar.readonly"],
            credential_id="sys-cred-id",
        )
        handler = MagicMock()
        handler.exchange_code_for_tokens = AsyncMock(return_value=new_cred)
        handler.handle_default_scopes.return_value = state.scopes

        # is_system_credential returns True only when asked about "sys-cred-id"
        # — emulating the real predicate that recognises platform-reserved IDs.
        def _is_system(cred_id):
            return cred_id == "sys-cred-id"

        with (
            patch(
                "backend.api.features.integrations.router._get_provider_oauth_handler",
                return_value=handler,
            ),
            patch(
                "backend.api.features.integrations.router.is_system_credential",
                side_effect=_is_system,
            ),
            patch("backend.api.features.integrations.router.creds_manager") as mock_mgr,
        ):
            mock_mgr.store.verify_state_token = AsyncMock(return_value=state)
            mock_mgr.store.get_creds_by_id = AsyncMock(return_value=existing)
            mock_mgr.update = AsyncMock()

            resp = client.post(
                "/google/callback",
                json={"code": "auth-code", "state_token": "state-token"},
            )

        assert resp.status_code == 400
        assert "system credentials" in resp.json()["detail"].lower()
        # No write must have happened for the system credential.
        mock_mgr.update.assert_not_called()

    def test_implicit_merge_skips_system_credentials(self):
        """The implicit (provider+username) merge filter must exclude system
        credentials so a user login cannot accidentally overwrite one."""
        system_match = _make_google_oauth2_cred(
            cred_id="sys-cred-id", username="alice@gmail.com"
        )
        new_cred = _make_google_oauth2_cred(
            cred_id="new-cred-id",
            scopes=system_match.scopes,
            username="alice@gmail.com",
        )
        state = OAuthState(
            token="state-token",
            provider="google",
            expires_at=9999999999,
            scopes=system_match.scopes,
        )
        handler = MagicMock()
        handler.exchange_code_for_tokens = AsyncMock(return_value=new_cred)
        handler.handle_default_scopes.return_value = state.scopes

        def _is_system(cred_id):
            return cred_id == "sys-cred-id"

        with (
            patch(
                "backend.api.features.integrations.router._get_provider_oauth_handler",
                return_value=handler,
            ),
            patch(
                "backend.api.features.integrations.router.is_system_credential",
                side_effect=_is_system,
            ),
            patch("backend.api.features.integrations.router.creds_manager") as mock_mgr,
        ):
            mock_mgr.store.verify_state_token = AsyncMock(return_value=state)
            mock_mgr.store.get_creds_by_provider = AsyncMock(
                return_value=[system_match]
            )
            mock_mgr.create = AsyncMock()
            mock_mgr.update = AsyncMock()

            resp = client.post(
                "/google/callback",
                json={"code": "auth-code", "state_token": "state-token"},
            )

        assert resp.status_code == 200
        # Since the only provider+username match is a system credential, the
        # callback must create a new credential rather than overwriting it.
        mock_mgr.create.assert_called_once()
        mock_mgr.update.assert_not_called()

    def test_upgrade_rejects_provider_mismatch(self):
        """Defense-in-depth: if a stale login somehow passed validation but the
        stored credential's provider no longer matches the new token's
        provider, the write-path must refuse to overwrite it."""
        existing = _make_google_oauth2_cred(cred_id="mixed-up-cred")
        # Simulate a provider drift: the new credential exchange returned a
        # different provider than what's stored on disk.
        new_cred = _make_github_oauth2_cred(cred_id="mixed-up-cred")
        state = OAuthState(
            token="state-token",
            provider="google",
            expires_at=9999999999,
            scopes=["https://www.googleapis.com/auth/gmail.readonly"],
            credential_id="mixed-up-cred",
        )
        handler = MagicMock()
        handler.exchange_code_for_tokens = AsyncMock(return_value=new_cred)
        handler.handle_default_scopes.return_value = state.scopes

        with (
            patch(
                "backend.api.features.integrations.router._get_provider_oauth_handler",
                return_value=handler,
            ),
            patch("backend.api.features.integrations.router.creds_manager") as mock_mgr,
        ):
            mock_mgr.store.verify_state_token = AsyncMock(return_value=state)
            mock_mgr.store.get_creds_by_id = AsyncMock(return_value=existing)
            mock_mgr.update = AsyncMock()

            resp = client.post(
                "/google/callback",
                json={"code": "auth-code", "state_token": "state-token"},
            )

        assert resp.status_code == 400
        assert "provider" in resp.json()["detail"].lower()
        mock_mgr.update.assert_not_called()


class TestPreserveRefreshTokenAndUsername:
    """Incremental callbacks must not silently drop refresh_token/username."""

    def test_upgrade_preserves_existing_refresh_token_when_new_is_empty(self):
        """If the new token response omits refresh_token, keep the existing one."""
        existing = _make_google_oauth2_cred(
            scopes=["https://www.googleapis.com/auth/gmail.readonly"],
        )
        existing.refresh_token = SecretStr("original-refresh")
        # Google may omit refresh_token on incremental re-authorization.
        new_cred = _make_google_oauth2_cred(
            scopes=[
                "https://www.googleapis.com/auth/gmail.readonly",
                "https://www.googleapis.com/auth/calendar.readonly",
            ],
        )
        new_cred.refresh_token = None  # type: ignore[assignment]

        state = OAuthState(
            token="state-token",
            provider="google",
            expires_at=9999999999,
            scopes=["https://www.googleapis.com/auth/calendar.readonly"],
            credential_id="google-cred-1",
        )
        handler = MagicMock()
        handler.exchange_code_for_tokens = AsyncMock(return_value=new_cred)
        handler.handle_default_scopes.return_value = state.scopes

        captured: dict[str, OAuth2Credentials] = {}

        async def _capture_update(_user_id, creds):
            captured["creds"] = creds

        with (
            patch(
                "backend.api.features.integrations.router._get_provider_oauth_handler",
                return_value=handler,
            ),
            patch("backend.api.features.integrations.router.creds_manager") as mock_mgr,
        ):
            mock_mgr.store.verify_state_token = AsyncMock(return_value=state)
            mock_mgr.store.get_creds_by_id = AsyncMock(return_value=existing)
            mock_mgr.update = AsyncMock(side_effect=_capture_update)

            resp = client.post(
                "/google/callback",
                json={"code": "auth-code", "state_token": "state-token"},
            )

        assert resp.status_code == 200
        updated = captured["creds"]
        assert updated.refresh_token is not None
        assert updated.refresh_token.get_secret_value() == "original-refresh"

    def test_upgrade_preserves_existing_username_when_new_is_empty(self):
        """If the new response lacks username, keep the existing one."""
        existing = _make_google_oauth2_cred(username="alice@gmail.com")
        new_cred = _make_google_oauth2_cred(scopes=existing.scopes)
        new_cred.username = None

        state = OAuthState(
            token="state-token",
            provider="google",
            expires_at=9999999999,
            scopes=existing.scopes,
            credential_id="google-cred-1",
        )
        handler = MagicMock()
        handler.exchange_code_for_tokens = AsyncMock(return_value=new_cred)
        handler.handle_default_scopes.return_value = state.scopes

        captured: dict[str, OAuth2Credentials] = {}

        async def _capture_update(_user_id, creds):
            captured["creds"] = creds

        with (
            patch(
                "backend.api.features.integrations.router._get_provider_oauth_handler",
                return_value=handler,
            ),
            patch("backend.api.features.integrations.router.creds_manager") as mock_mgr,
        ):
            mock_mgr.store.verify_state_token = AsyncMock(return_value=state)
            mock_mgr.store.get_creds_by_id = AsyncMock(return_value=existing)
            mock_mgr.update = AsyncMock(side_effect=_capture_update)

            resp = client.post(
                "/google/callback",
                json={"code": "auth-code", "state_token": "state-token"},
            )

        assert resp.status_code == 200
        assert captured["creds"].username == "alice@gmail.com"


class TestImplicitMergeScopeGuard:
    """Implicit (provider+username) merge must not advertise scopes wider than
    the freshly-minted token actually grants."""

    def _build_state(self, scopes: list[str]) -> OAuthState:
        return OAuthState(
            token="state-token",
            provider="google",
            expires_at=9999999999,
            scopes=scopes,
        )

    def test_implicit_merge_skipped_when_new_scopes_narrower(self):
        """If the new token doesn't cover all existing scopes, create a
        fresh credential instead of overwriting the existing one."""
        existing = _make_google_oauth2_cred(
            scopes=[
                "https://www.googleapis.com/auth/gmail.readonly",
                "https://www.googleapis.com/auth/calendar.readonly",
            ],
        )
        # New login only requested gmail — narrower than existing.
        new_cred = _make_google_oauth2_cred(
            cred_id="new-cred-id",
            scopes=["https://www.googleapis.com/auth/gmail.readonly"],
        )
        state = self._build_state(["https://www.googleapis.com/auth/gmail.readonly"])
        handler = MagicMock()
        handler.exchange_code_for_tokens = AsyncMock(return_value=new_cred)
        handler.handle_default_scopes.return_value = state.scopes

        with (
            patch(
                "backend.api.features.integrations.router._get_provider_oauth_handler",
                return_value=handler,
            ),
            patch("backend.api.features.integrations.router.creds_manager") as mock_mgr,
        ):
            mock_mgr.store.verify_state_token = AsyncMock(return_value=state)
            mock_mgr.store.get_creds_by_provider = AsyncMock(return_value=[existing])
            mock_mgr.store.get_creds_by_id = AsyncMock(return_value=existing)
            mock_mgr.create = AsyncMock()
            mock_mgr.update = AsyncMock()

            resp = client.post(
                "/google/callback",
                json={"code": "auth-code", "state_token": "state-token"},
            )

        assert resp.status_code == 200
        mock_mgr.create.assert_called_once()
        mock_mgr.update.assert_not_called()

    def test_implicit_merge_allowed_when_new_scopes_are_superset(self):
        """If the new token covers every existing scope, the implicit merge
        path can proceed as before."""
        existing = _make_google_oauth2_cred(
            scopes=["https://www.googleapis.com/auth/gmail.readonly"],
        )
        new_cred = _make_google_oauth2_cred(
            cred_id="new-cred-id",
            scopes=[
                "https://www.googleapis.com/auth/gmail.readonly",
                "https://www.googleapis.com/auth/calendar.readonly",
            ],
        )
        state = self._build_state(
            [
                "https://www.googleapis.com/auth/gmail.readonly",
                "https://www.googleapis.com/auth/calendar.readonly",
            ]
        )
        handler = MagicMock()
        handler.exchange_code_for_tokens = AsyncMock(return_value=new_cred)
        handler.handle_default_scopes.return_value = state.scopes

        with (
            patch(
                "backend.api.features.integrations.router._get_provider_oauth_handler",
                return_value=handler,
            ),
            patch("backend.api.features.integrations.router.creds_manager") as mock_mgr,
        ):
            mock_mgr.store.verify_state_token = AsyncMock(return_value=state)
            mock_mgr.store.get_creds_by_provider = AsyncMock(return_value=[existing])
            mock_mgr.store.get_creds_by_id = AsyncMock(return_value=existing)
            mock_mgr.create = AsyncMock()
            mock_mgr.update = AsyncMock()

            resp = client.post(
                "/google/callback",
                json={"code": "auth-code", "state_token": "state-token"},
            )

        assert resp.status_code == 200
        mock_mgr.update.assert_called_once()
        mock_mgr.create.assert_not_called()


class TestUpgradeExistingCredentialDoesNotMutateCaller:
    """Cursor Low (thread PRRT_kwDOJKSTjM58rern): ``_upgrade_existing_credential``
    used to mutate the caller's ``new_credentials`` object in-place
    (overwriting id/title/scopes/metadata/refresh_token/username). Safe
    today because all callers immediately replace their reference, but
    fragile — a future reader of ``credentials`` after the call would
    silently see overwritten values. Pin the contract so the caller's
    object stays intact."""

    @pytest.mark.asyncio
    async def test_caller_credentials_object_is_unchanged_after_upgrade(self):
        from backend.api.features.integrations.router import (
            _upgrade_existing_credential,
        )

        existing = _make_google_oauth2_cred(
            cred_id="existing-cred-id",
            scopes=["https://www.googleapis.com/auth/gmail.readonly"],
            username="alice@gmail.com",
            title="Existing title",
        )
        new_credentials = _make_google_oauth2_cred(
            cred_id="new-cred-id-from-exchange",
            scopes=[
                "https://www.googleapis.com/auth/gmail.readonly",
                "https://www.googleapis.com/auth/calendar.readonly",
            ],
            username="alice@gmail.com",
            title="New title from exchange",
        )

        # Snapshot the caller's object BEFORE the call so we can detect
        # any in-place mutation by comparing afterwards.
        snapshot = new_credentials.model_copy(deep=True)

        with (
            patch(
                "backend.api.features.integrations.router.is_system_credential",
                return_value=False,
            ),
            patch("backend.api.features.integrations.router.creds_manager") as mock_mgr,
        ):
            mock_mgr.store.get_creds_by_id = AsyncMock(return_value=existing)
            mock_mgr.update = AsyncMock()

            returned = await _upgrade_existing_credential(
                TEST_USER_ID, existing.id, new_credentials
            )

        # Caller's object must not have been touched — no id/title/scopes
        # rewrite, no refresh_token/username/metadata mutation.
        assert new_credentials.id == snapshot.id
        assert new_credentials.title == snapshot.title
        assert new_credentials.scopes == snapshot.scopes
        assert new_credentials.metadata == snapshot.metadata
        assert new_credentials.username == snapshot.username
        assert (
            new_credentials.refresh_token.get_secret_value()
            if new_credentials.refresh_token
            else None
        ) == (
            snapshot.refresh_token.get_secret_value()
            if snapshot.refresh_token
            else None
        )

        # The returned object carries the merged state, and is persisted.
        assert returned.id == existing.id
        assert set(returned.scopes) == {
            "https://www.googleapis.com/auth/gmail.readonly",
            "https://www.googleapis.com/auth/calendar.readonly",
        }
        mock_mgr.update.assert_called_once()


class TestExplicitUpgradeScopeGuard:
    """Explicit scope upgrade (`credential_id` set on the OAuth state) must
    enforce the same scope-coverage guard as the implicit-merge path.

    Without the guard a narrowed re-auth — the user lands on the provider's
    consent screen and only grants some of the requested scopes — would
    overwrite the existing credential's ``access_token`` with a narrower
    token while merging the wider scope set onto the record.  The
    credential matcher then routes AutoPilot tools to that record believing
    it covers scopes its token does not actually grant, and the tool fails
    with opaque 401/403s on the missing scopes.  Users perceive this as
    "AutoPilot keeps picking the old creds" because the loop never breaks.
    """

    def _make_state_with_credential_id(
        self,
        credential_id: str,
        scopes: list[str] | None = None,
        provider: str = "google",
    ) -> OAuthState:
        return OAuthState(
            token="state-token",
            provider=provider,
            expires_at=9999999999,
            scopes=(
                scopes
                if scopes is not None
                else [
                    "https://www.googleapis.com/auth/gmail.readonly",
                    "https://www.googleapis.com/auth/calendar.readonly",
                ]
            ),
            credential_id=credential_id,
        )

    def test_explicit_upgrade_skipped_when_new_scopes_narrower(self):
        """The user re-auths to upgrade scopes but the provider returns a
        narrower token than what the existing record advertises.  The
        existing credential must stay intact (its old token still grants
        the lost scopes) and the new (narrower) credential is persisted
        alongside it."""
        existing = _make_google_oauth2_cred(
            scopes=[
                "https://www.googleapis.com/auth/gmail.readonly",
                "https://www.googleapis.com/auth/calendar.readonly",
            ],
        )
        # The user only granted Gmail on the consent screen — narrower
        # than the existing record's claimed Gmail + Calendar.
        new_cred = _make_google_oauth2_cred(
            cred_id="new-cred-id",
            scopes=["https://www.googleapis.com/auth/gmail.readonly"],
        )
        state = self._make_state_with_credential_id(
            "google-cred-1",
            scopes=["https://www.googleapis.com/auth/gmail.readonly"],
        )
        handler = MagicMock()
        handler.exchange_code_for_tokens = AsyncMock(return_value=new_cred)
        handler.handle_default_scopes.return_value = state.scopes

        with (
            patch(
                "backend.api.features.integrations.router._get_provider_oauth_handler",
                return_value=handler,
            ),
            patch("backend.api.features.integrations.router.creds_manager") as mock_mgr,
        ):
            mock_mgr.store.verify_state_token = AsyncMock(return_value=state)
            mock_mgr.store.get_creds_by_id = AsyncMock(return_value=existing)
            mock_mgr.create = AsyncMock()
            mock_mgr.update = AsyncMock()

            resp = client.post(
                "/google/callback",
                json={"code": "auth-code", "state_token": "state-token"},
            )

        assert resp.status_code == 200
        # No mutation of the existing record — keep its wider token intact.
        mock_mgr.update.assert_not_called()
        # The narrower credential is persisted as a new record so the user
        # ends up with both: the AutoPilot matcher will now pick the one
        # that actually grants the requested scopes for each tool call.
        mock_mgr.create.assert_called_once()

    def test_explicit_upgrade_proceeds_when_new_scopes_are_superset(self):
        """Typical happy path: the OAuth flow asked for existing ∪ new and
        the user authorised everything.  The new token covers the wider
        set, so merging into the existing record is safe."""
        existing = _make_google_oauth2_cred(
            scopes=["https://www.googleapis.com/auth/gmail.readonly"],
        )
        new_cred = _make_google_oauth2_cred(
            cred_id="new-cred-id",
            scopes=[
                "https://www.googleapis.com/auth/gmail.readonly",
                "https://www.googleapis.com/auth/calendar.readonly",
            ],
        )
        state = self._make_state_with_credential_id("google-cred-1")
        handler = MagicMock()
        handler.exchange_code_for_tokens = AsyncMock(return_value=new_cred)
        handler.handle_default_scopes.return_value = state.scopes

        with (
            patch(
                "backend.api.features.integrations.router._get_provider_oauth_handler",
                return_value=handler,
            ),
            patch("backend.api.features.integrations.router.creds_manager") as mock_mgr,
        ):
            mock_mgr.store.verify_state_token = AsyncMock(return_value=state)
            mock_mgr.store.get_creds_by_id = AsyncMock(return_value=existing)
            mock_mgr.create = AsyncMock()
            mock_mgr.update = AsyncMock()

            resp = client.post(
                "/google/callback",
                json={"code": "auth-code", "state_token": "state-token"},
            )

        assert resp.status_code == 200
        mock_mgr.update.assert_called_once()
        mock_mgr.create.assert_not_called()

    def test_explicit_upgrade_with_equal_scopes_still_merges(self):
        """Re-auth with the same scope set (e.g. user refreshed an expired
        token by going through the OAuth flow again) is the same as
        ``new ⊇ existing`` — merging into the existing record updates
        the access_token without scope drift."""
        scopes = ["https://www.googleapis.com/auth/gmail.readonly"]
        existing = _make_google_oauth2_cred(scopes=scopes)
        new_cred = _make_google_oauth2_cred(cred_id="new-cred-id", scopes=scopes)
        state = self._make_state_with_credential_id("google-cred-1", scopes=scopes)
        handler = MagicMock()
        handler.exchange_code_for_tokens = AsyncMock(return_value=new_cred)
        handler.handle_default_scopes.return_value = state.scopes

        with (
            patch(
                "backend.api.features.integrations.router._get_provider_oauth_handler",
                return_value=handler,
            ),
            patch("backend.api.features.integrations.router.creds_manager") as mock_mgr,
        ):
            mock_mgr.store.verify_state_token = AsyncMock(return_value=state)
            mock_mgr.store.get_creds_by_id = AsyncMock(return_value=existing)
            mock_mgr.create = AsyncMock()
            mock_mgr.update = AsyncMock()

            resp = client.post(
                "/google/callback",
                json={"code": "auth-code", "state_token": "state-token"},
            )

        assert resp.status_code == 200
        mock_mgr.update.assert_called_once()
        mock_mgr.create.assert_not_called()
