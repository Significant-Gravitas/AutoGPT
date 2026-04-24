"""Tests for AyrshareManagedProvider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.data.model import APIKeyCredentials
from backend.integrations.managed_providers.ayrshare import (
    AyrshareManagedProvider,
    _migrate_legacy_or_create_profile_key,
    settings_available,
)

_USER_ID = "user-ayrshare-test"


class TestIsAvailable:
    """`is_available` is truthful; opt-out lives on `auto_provision`."""

    @pytest.mark.asyncio(loop_scope="session")
    async def test_is_available_true_when_secrets_set(self):
        """Truthful: returns True when org-level secrets are configured.

        The sweep skip is driven by `auto_provision = False`, not by
        lying about availability.  Callers like `ensure_managed_credential`
        do not gate on `is_available`, so this remains truthful without
        triggering auto-provisioning.
        """
        with patch(
            "backend.integrations.managed_providers.ayrshare.Settings"
        ) as mock_settings:
            mock_settings.return_value.secrets.ayrshare_api_key = "api-key"
            mock_settings.return_value.secrets.ayrshare_jwt_key = "jwt-key"
            assert await AyrshareManagedProvider().is_available() is True

    @pytest.mark.asyncio(loop_scope="session")
    async def test_is_available_false_when_secrets_missing(self):
        with patch(
            "backend.integrations.managed_providers.ayrshare.Settings"
        ) as mock_settings:
            mock_settings.return_value.secrets.ayrshare_api_key = ""
            mock_settings.return_value.secrets.ayrshare_jwt_key = "jwt-key"
            assert await AyrshareManagedProvider().is_available() is False

    def test_auto_provision_opt_out(self):
        """Ayrshare opts out of the credentials sweep — per-user Ayrshare profiles
        count against our subscription quota, so we only provision when the
        user explicitly clicks the builder's SSO flow."""
        assert AyrshareManagedProvider.auto_provision is False


class TestSettingsAvailable:
    """Pre-flight check used by the SSO-URL route before provisioning."""

    def test_returns_true_when_both_secrets_set(self):
        with patch(
            "backend.integrations.managed_providers.ayrshare.Settings"
        ) as mock_settings:
            mock_settings.return_value.secrets.ayrshare_api_key = "api-key"
            mock_settings.return_value.secrets.ayrshare_jwt_key = "jwt-key"
            assert settings_available() is True

    def test_returns_false_when_api_key_missing(self):
        with patch(
            "backend.integrations.managed_providers.ayrshare.Settings"
        ) as mock_settings:
            mock_settings.return_value.secrets.ayrshare_api_key = ""
            mock_settings.return_value.secrets.ayrshare_jwt_key = "jwt-key"
            assert settings_available() is False

    def test_returns_false_when_jwt_key_missing(self):
        with patch(
            "backend.integrations.managed_providers.ayrshare.Settings"
        ) as mock_settings:
            mock_settings.return_value.secrets.ayrshare_api_key = "api-key"
            mock_settings.return_value.secrets.ayrshare_jwt_key = ""
            assert settings_available() is False


class TestReadOrCreateProfileKey:
    """Legacy migration + fresh-profile paths.

    The read path is read-only w.r.t. the legacy field — clearing the legacy
    ``managed_credentials.ayrshare_profile_key`` happens in
    :meth:`AyrshareManagedProvider.post_provision`, only after the managed
    credential is durably stored.
    """

    def _mock_store(self, legacy_key: SecretStr | None):
        """Build a minimal store stub that exposes the new public
        read accessor; read-only callers don't go through
        ``edit_user_integrations``."""
        managed = MagicMock()
        managed.ayrshare_profile_key = legacy_key

        user_integrations = MagicMock()
        user_integrations.managed_credentials = managed

        store = MagicMock()
        store.get_user_integrations = AsyncMock(return_value=user_integrations)
        return store, user_integrations

    @pytest.mark.asyncio(loop_scope="session")
    async def test_reuses_legacy_key_without_clearing(self):
        """Legacy field is NOT cleared here — that happens in post_provision.

        If `_migrate_legacy_or_create_profile_key` cleared eagerly and the subsequent
        `add_managed_credential` failed, a retry would see an empty legacy
        field and create a fresh Ayrshare profile, orphaning the user's
        linked social accounts.
        """
        legacy = SecretStr("legacy-profile-key")
        store, ui = self._mock_store(legacy_key=legacy)

        with patch(
            "backend.integrations.managed_providers.ayrshare.AyrshareClient"
        ) as mock_client:
            result = await _migrate_legacy_or_create_profile_key(_USER_ID, store)

        assert result == "legacy-profile-key"
        # create_profile must NOT be called — we reuse the existing one.
        mock_client.assert_not_called()
        # Legacy field must NOT be cleared by the read path.
        assert ui.managed_credentials.ayrshare_profile_key is legacy

    @pytest.mark.asyncio(loop_scope="session")
    async def test_creates_new_profile_when_no_legacy(self):
        """Without a legacy key, we create a fresh profile with a unique title."""
        store, _ = self._mock_store(legacy_key=None)

        fake_profile = MagicMock(profileKey="fresh-profile-key")
        client_instance = MagicMock()
        client_instance.create_profile = AsyncMock(return_value=fake_profile)

        with patch(
            "backend.integrations.managed_providers.ayrshare.AyrshareClient",
            return_value=client_instance,
        ):
            result = await _migrate_legacy_or_create_profile_key(_USER_ID, store)

        assert result == "fresh-profile-key"
        client_instance.create_profile.assert_awaited_once()
        # The title must include the user_id AND a suffix — unique-per-call
        # avoids collisions with orphaned upstream profiles (Ayrshare has
        # no API to retrieve an existing profile's key).
        call_kwargs = client_instance.create_profile.call_args.kwargs
        assert call_kwargs["title"].startswith(f"User {_USER_ID}-")
        assert call_kwargs["title"] != f"User {_USER_ID}"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_profile_title_suffix_is_unique_across_calls(self):
        """Two separate provision attempts produce different titles so an
        orphaned profile from a prior attempt never causes a duplicate-
        title collision on create."""
        from backend.integrations.managed_providers.ayrshare import _profile_title

        t1 = _profile_title(_USER_ID)
        t2 = _profile_title(_USER_ID)
        assert t1 != t2
        assert t1.startswith(f"User {_USER_ID}-")
        assert t2.startswith(f"User {_USER_ID}-")


class TestPostProvisionClearsLegacy:
    """post_provision runs only after the managed credential is durable.

    Verifies the migration-ordering fix for the data-loss race:
    - provision() reads the legacy key without clearing it.
    - add_managed_credential persists the new credential.
    - post_provision then clears the legacy field.
    Failure between provision and add_managed_credential leaves the legacy
    key intact, so a retry reuses it and keeps the user's linked socials.
    """

    def _mock_store_for_clear(self, legacy_key: SecretStr | None):
        managed = MagicMock()
        managed.ayrshare_profile_key = legacy_key

        user_integrations = MagicMock()
        user_integrations.managed_credentials = managed

        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=user_integrations)
        cm.__aexit__ = AsyncMock(return_value=None)

        store = MagicMock()
        store.edit_user_integrations = MagicMock(return_value=cm)
        return store, user_integrations

    @pytest.mark.asyncio(loop_scope="session")
    async def test_post_provision_clears_populated_legacy_field(self):
        store, ui = self._mock_store_for_clear(SecretStr("legacy"))
        fake_cred = APIKeyCredentials(
            provider="ayrshare",
            title="t",
            api_key=SecretStr("k"),
            expires_at=None,
            is_managed=True,
        )
        await AyrshareManagedProvider().post_provision(_USER_ID, store, fake_cred)
        assert ui.managed_credentials.ayrshare_profile_key is None

    @pytest.mark.asyncio(loop_scope="session")
    async def test_post_provision_skips_when_legacy_already_clear(self):
        """Idempotent — a second call on a fresh user touches nothing."""
        store, ui = self._mock_store_for_clear(legacy_key=None)
        fake_cred = APIKeyCredentials(
            provider="ayrshare",
            title="t",
            api_key=SecretStr("k"),
            expires_at=None,
            is_managed=True,
        )
        await AyrshareManagedProvider().post_provision(_USER_ID, store, fake_cred)
        # Stayed None; no write attempted.
        assert ui.managed_credentials.ayrshare_profile_key is None


class TestProvision:
    """provision() returns an is_managed=True APIKeyCredentials."""

    @pytest.mark.asyncio(loop_scope="session")
    async def test_provision_returns_managed_api_key_credential(self):
        """provision now receives the caller-supplied store (framework-injected)."""
        store = MagicMock()
        with patch(
            "backend.integrations.managed_providers.ayrshare."
            "_migrate_legacy_or_create_profile_key",
            new=AsyncMock(return_value="profile-key-xyz"),
        ) as read_mock:
            creds = await AyrshareManagedProvider().provision(_USER_ID, store)

        assert isinstance(creds, APIKeyCredentials)
        assert creds.provider == "ayrshare"
        assert creds.is_managed is True
        assert creds.api_key.get_secret_value() == "profile-key-xyz"
        # The store passed into provision must be the one forwarded to the
        # internal read helper — no hidden construction of a fresh store.
        read_mock.assert_awaited_once_with(_USER_ID, store)


class TestMigrationOrderingSafety:
    """Regression test for the migration ordering fix.

    Verifies that if ``add_managed_credential`` raises after
    ``provision()`` succeeds, the legacy ``ayrshare_profile_key`` survives —
    a retry can then reuse it and keep the user's linked socials.
    """

    @pytest.mark.asyncio(loop_scope="session")
    async def test_add_managed_credential_failure_retains_legacy_key(self):
        from backend.integrations.managed_credentials import _provision_under_lock

        # Mock store: has_managed_credential=False, edit_user_integrations
        # yields a UserIntegrations with the legacy key populated, and
        # add_managed_credential raises to simulate a DB blip.  Because
        # _provision_under_lock fails before post_provision runs, the
        # legacy key must remain untouched — a retry will reuse it.
        legacy = SecretStr("legacy-profile-key")

        managed = MagicMock()
        managed.ayrshare_profile_key = legacy
        user_integrations = MagicMock()
        user_integrations.managed_credentials = managed

        lock_cm = AsyncMock()
        lock_cm.__aenter__ = AsyncMock(return_value=None)
        lock_cm.__aexit__ = AsyncMock(return_value=None)
        locks = MagicMock()
        locks.locked = MagicMock(return_value=lock_cm)

        store = MagicMock()
        store.locks = AsyncMock(return_value=locks)
        store.has_managed_credential = AsyncMock(return_value=False)
        # The provisioning read path uses the new public accessor; the
        # post_provision clear path still uses edit_user_integrations.
        # Here we assert the read works and the clear never runs because
        # add_managed_credential fails first.
        store.get_user_integrations = AsyncMock(return_value=user_integrations)
        store.add_managed_credential = AsyncMock(side_effect=RuntimeError("DB blip"))

        # provision now receives the caller-supplied store directly, so no
        # constructor patch is needed — the framework threads the same mock
        # from _provision_under_lock into AyrshareManagedProvider.provision.
        provider = AyrshareManagedProvider()
        with pytest.raises(RuntimeError, match="DB blip"):
            await _provision_under_lock(_USER_ID, store, "ayrshare", provider)

        # The legacy key MUST still be populated — otherwise a retry would
        # create a fresh Ayrshare profile and orphan the user's socials.
        assert user_integrations.managed_credentials.ayrshare_profile_key is legacy
