"""Tests for AyrshareManagedProvider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.data.model import APIKeyCredentials
from backend.integrations.managed_providers.ayrshare import (
    AyrshareManagedProvider,
    _get_or_create_profile_key,
    _settings_available,
)

_USER_ID = "user-ayrshare-test"


class TestIsAvailable:
    """Ayrshare opts out of the automatic managed-credentials sweep."""

    @pytest.mark.asyncio(loop_scope="session")
    async def test_is_available_always_false(self):
        """Returning False keeps ensure_managed_credentials from auto-provisioning.

        Profile quota is a real per-user subscription cost, so provisioning
        must only happen when the user explicitly opens the social-connect
        flow via /api/integrations/ayrshare/sso_url.
        """
        # Even with both org secrets populated, is_available stays False —
        # callers who actually want to provision use
        # ensure_managed_credential() directly, bypassing this gate.
        with patch(
            "backend.integrations.managed_providers.ayrshare.Settings"
        ) as mock_settings:
            mock_settings.return_value.secrets.ayrshare_api_key = "api-key"
            mock_settings.return_value.secrets.ayrshare_jwt_key = "jwt-key"
            assert await AyrshareManagedProvider().is_available() is False


class TestSettingsAvailable:
    """Pre-flight check used by the SSO-URL route before provisioning."""

    def test_returns_true_when_both_secrets_set(self):
        with patch(
            "backend.integrations.managed_providers.ayrshare.Settings"
        ) as mock_settings:
            mock_settings.return_value.secrets.ayrshare_api_key = "api-key"
            mock_settings.return_value.secrets.ayrshare_jwt_key = "jwt-key"
            assert _settings_available() is True

    def test_returns_false_when_api_key_missing(self):
        with patch(
            "backend.integrations.managed_providers.ayrshare.Settings"
        ) as mock_settings:
            mock_settings.return_value.secrets.ayrshare_api_key = ""
            mock_settings.return_value.secrets.ayrshare_jwt_key = "jwt-key"
            assert _settings_available() is False

    def test_returns_false_when_jwt_key_missing(self):
        with patch(
            "backend.integrations.managed_providers.ayrshare.Settings"
        ) as mock_settings:
            mock_settings.return_value.secrets.ayrshare_api_key = "api-key"
            mock_settings.return_value.secrets.ayrshare_jwt_key = ""
            assert _settings_available() is False


class TestGetOrCreateProfileKey:
    """Legacy migration + fresh-profile paths."""

    def _mock_store(self, legacy_key: SecretStr | None):
        """Build a minimal store stub that yields a mutable user_integrations."""
        managed = MagicMock()
        managed.ayrshare_profile_key = legacy_key

        user_integrations = MagicMock()
        user_integrations.managed_credentials = managed

        # edit_user_integrations is used as `async with store.edit(...) as ui:`.
        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=user_integrations)
        cm.__aexit__ = AsyncMock(return_value=None)

        store = MagicMock()
        store.edit_user_integrations = MagicMock(return_value=cm)
        return store, user_integrations

    @pytest.mark.asyncio(loop_scope="session")
    async def test_reuses_legacy_key_and_clears_field(self):
        """Pre-migration users keep their linked socials — the legacy profile
        key is migrated into the returned credential and the legacy field is
        cleared in the same write."""
        legacy = SecretStr("legacy-profile-key")
        store, ui = self._mock_store(legacy_key=legacy)

        with patch(
            "backend.integrations.managed_providers.ayrshare.AyrshareClient"
        ) as mock_client:
            result = await _get_or_create_profile_key(_USER_ID, store)

        assert result == "legacy-profile-key"
        # create_profile must NOT be called — we reuse the existing one.
        mock_client.assert_not_called()
        # Legacy field must be cleared under the same lock as the read.
        assert ui.managed_credentials.ayrshare_profile_key is None

    @pytest.mark.asyncio(loop_scope="session")
    async def test_creates_new_profile_when_no_legacy_key(self):
        """Fresh users get a new Ayrshare profile created; the managed
        credential wraps the returned profile key."""
        store, _ = self._mock_store(legacy_key=None)

        fake_profile = MagicMock(profileKey="fresh-profile-key")
        client_instance = MagicMock()
        client_instance.create_profile = AsyncMock(return_value=fake_profile)

        with patch(
            "backend.integrations.managed_providers.ayrshare.AyrshareClient",
            return_value=client_instance,
        ):
            result = await _get_or_create_profile_key(_USER_ID, store)

        assert result == "fresh-profile-key"
        client_instance.create_profile.assert_awaited_once()


class TestProvision:
    """provision() returns an is_managed=True APIKeyCredentials."""

    @pytest.mark.asyncio(loop_scope="session")
    async def test_provision_returns_managed_api_key_credential(self):
        with patch(
            "backend.integrations.managed_providers.ayrshare."
            "_get_or_create_profile_key",
            new=AsyncMock(return_value="profile-key-xyz"),
        ):
            with patch(
                "backend.integrations.managed_providers.ayrshare."
                "IntegrationCredentialsStore"
            ):
                creds = await AyrshareManagedProvider().provision(_USER_ID)

        assert isinstance(creds, APIKeyCredentials)
        assert creds.provider == "ayrshare"
        assert creds.is_managed is True
        assert creds.api_key.get_secret_value() == "profile-key-xyz"
