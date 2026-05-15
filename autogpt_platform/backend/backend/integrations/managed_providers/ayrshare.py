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
import secrets

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

    # Opt out of the credentials sweep — each Ayrshare profile counts
    # against our subscription quota, so we only provision when the user
    # explicitly triggers the SSO flow from a block.  See
    # ``ManagedCredentialProvider.auto_provision`` for the gate semantics.
    auto_provision = False

    async def is_available(self) -> bool:
        """True when both ``AYRSHARE_API_KEY`` and ``AYRSHARE_JWT_KEY`` are
        configured.  Pure env-var check — does NOT ping Ayrshare."""
        return settings_available()

    async def provision(
        self, user_id: str, store: IntegrationCredentialsStore
    ) -> Credentials:
        profile_key = await _migrate_legacy_or_create_profile_key(user_id, store)
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
    """A unique Ayrshare profile title for this provision attempt.

    Appends a short random suffix so we never collide with an orphan
    upstream profile (same user, lost managed credential from a prior
    session).  Ayrshare's ``DELETE /profiles`` requires the ``profileKey``
    we no longer have, so avoiding the collision in the first place is
    the only reliable recovery path.  The suffix is cosmetic — Ayrshare
    profiles are keyed by ``profileKey``, not title.
    """
    return f"User {user_id}-{secrets.token_hex(3)}"


async def _migrate_legacy_or_create_profile_key(
    user_id: str, store: IntegrationCredentialsStore
) -> str:
    """Return an Ayrshare profile key for *user_id*.

    Only called from :meth:`AyrshareManagedProvider.provision`, which is
    itself only reached when the outer
    :func:`~backend.integrations.managed_credentials._provision_under_lock`
    has already confirmed via ``has_managed_credential`` that this user has
    no managed Ayrshare credential yet.  So this function does NOT re-check
    the managed credential — it only has to decide between:

    1. **Legacy side channel.** If
       ``UserIntegrations.managed_credentials.ayrshare_profile_key`` is set
       (pre-migration data), reuse it verbatim so existing linked socials
       keep working.  Read-only here — clearing moves to
       :meth:`AyrshareManagedProvider.post_provision` (runs after the
       managed credential is durably stored; if persistence fails, legacy
       stays intact so a retry reuses it).

    2. **Create a fresh profile with a unique title.** The title carries
       a random suffix so we never collide with orphaned upstream profiles
       (Ayrshare doesn't expose an API to retrieve an existing profile's
       ``profileKey``, so collision-avoidance is the only reliable recovery
       path).  Orphans stick around in Ayrshare's dashboard until cleaned
       up manually — acceptable cost for unblocking the user.

    ``_provision_under_lock`` also holds the distributed Redis lock across
    this whole function *and* the subsequent ``add_managed_credential``
    call, so concurrent workers cannot race and create duplicates.
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
    logger.debug("[ayrshare] Creating profile for user %s (title=%s)", user_id, title)
    profile = await client.create_profile(title=title, messaging_active=True)
    return profile.profileKey


def settings_available() -> bool:
    """True when Ayrshare org-level secrets are configured.

    Exposed so on-demand callers (e.g. the SSO-URL route) can pre-flight the
    config before calling :func:`~backend.integrations.managed_credentials.ensure_managed_credential`.
    """
    settings = Settings()
    return bool(settings.secrets.ayrshare_api_key and settings.secrets.ayrshare_jwt_key)
