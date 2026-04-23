"""Ayrshare managed credential provider.

Provisions a per-user Ayrshare profile key and stores it as a standard
``APIKeyCredentials(provider="ayrshare", is_managed=True)`` in the user's
credentials list.  This lets every Ayrshare block declare a normal
``credentials`` field and go through the same schema-driven resolution as
any other provider — no bespoke ``managed_credentials.ayrshare_profile_key``
side channel required.

Auto-provisioned by the credential-list sweep the same way AgentMail is,
so every Ayrshare block just declares a standard ``credentials`` field
and the managed entry appears in the builder dropdown automatically.
Profile creation counts against the org Ayrshare subscription quota, so
that cost is accepted at the plan level rather than gated per-user.

Legacy compatibility: on first provision we migrate
``UserIntegrations.managed_credentials.ayrshare_profile_key`` (pre-migration
data) into the new managed credential.  The legacy field is then cleared
in :meth:`AyrshareManagedProvider.post_provision`, which runs only after
``add_managed_credential`` has durably stored the managed credential —
this ordering ensures that a failure between ``provision`` and the
persist step leaves the legacy key intact so a retry can still reuse it
(covered by ``TestMigrationOrderingSafety``).

User-visible caveat: provisioning the profile creates the Ayrshare profile
but does not link any social accounts.  The user still needs to open the
Ayrshare SSO popup to OAuth each social network; the block will return
the Ayrshare API's "not linked" error until they do.  That part remains
platform UX, not a credential concern.
"""

from __future__ import annotations

import logging

from pydantic import SecretStr

from backend.data.model import APIKeyCredentials, Credentials
from backend.integrations.ayrshare import AyrshareClient
from backend.integrations.credentials_store import IntegrationCredentialsStore
from backend.integrations.managed_credentials import ManagedCredentialProvider
from backend.util.exceptions import MissingConfigError
from backend.util.settings import Settings

logger = logging.getLogger(__name__)


class AyrshareManagedProvider(ManagedCredentialProvider):
    provider_name = "ayrshare"

    # Opt out of the startup sweep — each Ayrshare profile counts against
    # our subscription quota, so we only provision when the user actually
    # opens a block that needs it (triggered by the builder's per-provider
    # ``GET /{provider}/credentials`` call).
    auto_provision = False

    async def is_available(self) -> bool:
        """Both Ayrshare org-level secrets must be configured."""
        return settings_available()

    async def provision(
        self, user_id: str, store: IntegrationCredentialsStore
    ) -> Credentials:
        profile_key = await _read_or_create_profile_key(user_id, store)
        return APIKeyCredentials(
            provider=self.provider_name,
            title="Ayrshare (managed by AutoGPT)",
            api_key=SecretStr(profile_key),
            expires_at=None,
            is_managed=True,
        )

    async def deprovision(self, user_id: str, credential: Credentials) -> None:
        # Ayrshare's public API does not expose a programmatic profile-delete
        # endpoint today.  Orphaned profiles incur no runtime cost on our
        # side (billing is per-post, not per-profile) and can be cleaned up
        # manually from the Ayrshare dashboard if ever needed.
        logger.info(
            "[ayrshare] No programmatic deprovisioning; leaving profile "
            "for user %s intact.",
            user_id,
        )

    async def post_provision(
        self,
        user_id: str,
        store: IntegrationCredentialsStore,
        credential: Credentials,
    ) -> None:
        """Clear the legacy ``ayrshare_profile_key`` side-channel after migration.

        See :meth:`ManagedCredentialProvider.post_provision` for retry
        semantics (idempotent, failures are swallowed and logged).
        """
        _ = credential  # unused; the side channel is provider-specific
        async with store.edit_user_integrations(user_id) as user_integrations:
            if user_integrations.managed_credentials.ayrshare_profile_key is not None:
                logger.debug(
                    "[ayrshare] Clearing legacy profile_key for user %s", user_id
                )
                user_integrations.managed_credentials.ayrshare_profile_key = None


def _profile_title(user_id: str) -> str:
    """The deterministic Ayrshare profile title for a user.

    Used both to create the profile and to find it again on retry — a
    collision-free lookup key so we never double-create for the same user.
    """
    return f"User {user_id}"


async def _read_or_create_profile_key(
    user_id: str, store: IntegrationCredentialsStore
) -> str:
    """Return the Ayrshare profile key for *user_id*, creating one if needed.

    **Resolution order — idempotent, retry-safe:**

    1. **Legacy side channel.** If
       ``UserIntegrations.managed_credentials.ayrshare_profile_key`` is set
       (pre-migration data), reuse it verbatim so existing linked socials
       keep working.  Read-only here — clearing moves to
       :meth:`AyrshareManagedProvider.post_provision` (runs after the
       managed credential is durably stored; if persistence fails, legacy
       stays intact so a retry reuses it).

    2. **Existing Ayrshare profile by title.** Before calling
       ``create_profile``, list Ayrshare's profiles under our account and
       check for one titled :func:`_profile_title`.  This covers the
       "previous attempt created the profile upstream but failed to persist
       our managed credential" path — without it, every such retry would
       leak another profile against the subscription's quota.

    3. **Create a fresh profile.**  Only reached when the user has never
       had a profile upstream.  The deterministic title keeps ``(2)`` a
       reliable recovery for any future retry.

    The outer :func:`~backend.integrations.managed_credentials._provision_under_lock`
    holds a distributed Redis lock on ``(user, provider)`` across this whole
    function *and* the subsequent ``add_managed_credential`` call, so
    concurrent workers cannot race through steps 2/3 and create duplicates.
    """
    user_integrations = await store.get_user_integrations(user_id)
    legacy_key = user_integrations.managed_credentials.ayrshare_profile_key
    if legacy_key:
        logger.debug("[ayrshare] Reusing legacy profile key for user %s", user_id)
        return (
            legacy_key.get_secret_value()
            if isinstance(legacy_key, SecretStr)
            else str(legacy_key)
        )

    try:
        client = AyrshareClient()
    except MissingConfigError as exc:
        raise RuntimeError("Ayrshare integration is not configured") from exc

    title = _profile_title(user_id)
    # Ayrshare's GET /profiles never returns ``profileKey`` — there's no
    # supported endpoint to recover an existing profile's key.  If a
    # profile with our title exists upstream it's orphaned (we lost our
    # side of the secret), so delete it and create fresh.  This is
    # destructive — any social accounts linked to the orphan are lost —
    # but the orphan is already unusable without the key, so the only
    # alternatives would be a stuck user or a shadow profile forever.
    existing = await client.list_profiles()
    orphan = next((p for p in existing if p.title == title), None)
    if orphan is not None:
        logger.warning(
            "[ayrshare] Orphaned upstream profile detected for user %s "
            "(refId=%s); deleting so we can create a fresh one with a "
            "retrievable profileKey.  Any previously-linked social "
            "accounts on the orphan will be lost.",
            user_id,
            orphan.refId,
        )
        await client.delete_profile(title=title)

    logger.debug("[ayrshare] Creating profile for user %s", user_id)
    profile = await client.create_profile(title=title, messaging_active=True)
    return profile.profileKey


def settings_available() -> bool:
    """True when Ayrshare org-level secrets are configured.

    Exposed so on-demand callers (e.g. the SSO-URL route) can pre-flight the
    config before calling :func:`~backend.integrations.managed_credentials.ensure_managed_credential`.
    """
    settings = Settings()
    return bool(settings.secrets.ayrshare_api_key and settings.secrets.ayrshare_jwt_key)
