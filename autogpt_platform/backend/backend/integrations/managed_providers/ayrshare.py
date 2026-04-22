"""Ayrshare managed credential provider.

Provisions a per-user Ayrshare profile key and stores it as a standard
``APIKeyCredentials(provider="ayrshare", is_managed=True)`` in the user's
credentials list.  This lets every Ayrshare block declare a normal
``credentials`` field and go through the same schema-driven resolution as
any other provider — no bespoke ``managed_credentials.ayrshare_profile_key``
side channel required.

**Opt-out of auto-provisioning.**  Unlike AgentMail, :meth:`is_available`
returns ``False`` so the ``ensure_managed_credentials`` sweep *never*
provisions Ayrshare automatically on credential-list loads — Ayrshare
profiles count against the org subscription quota and must not be created
for users who never touch a social-media block.  Callers who genuinely
need to provision (currently only ``/api/integrations/ayrshare/sso_url``)
invoke :func:`~backend.integrations.managed_credentials.ensure_managed_credential`
with an instance of this provider, which re-uses the same distributed
Redis lock + upsert path as AgentMail.

Legacy compatibility: on first provision we migrate
``UserIntegrations.managed_credentials.ayrshare_profile_key`` (pre-migration
data) into the new managed credential and clear the legacy field in the
same write — so the eventual removal of the legacy schema field has
nothing to chase.

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

    async def is_available(self) -> bool:
        # Opt out of the `ensure_managed_credentials` startup sweep.  Profile
        # creation has real per-user quota cost — we only want to create one
        # when the user explicitly requests a social-media connection via
        # `/api/integrations/ayrshare/sso_url`.  Callers that *do* want to
        # provision on demand go through
        # `ensure_managed_credential(user_id, store, AyrshareManagedProvider())`
        # which bypasses this gate.
        return False

    async def provision(self, user_id: str) -> Credentials:
        # Lazy import: avoids a managed_providers → credentials_store cycle
        # and keeps the module importable at startup from
        # managed_providers/__init__.py.
        store = IntegrationCredentialsStore()
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

        Runs only after the managed credential is durably stored.  If the
        legacy field is still populated, the profile key now lives in two
        places — clear the old one so the eventual schema removal has
        nothing to chase.  A failure here is harmless: the managed
        credential is already persisted, subsequent provision calls
        short-circuit, and the legacy field stays until the next successful
        migration attempt.
        """
        _ = credential  # unused; the side channel is provider-specific
        async with store.edit_user_integrations(user_id) as user_integrations:
            if user_integrations.managed_credentials.ayrshare_profile_key is not None:
                logger.debug(
                    "[ayrshare] Clearing legacy profile_key for user %s", user_id
                )
                user_integrations.managed_credentials.ayrshare_profile_key = None


async def _read_or_create_profile_key(
    user_id: str, store: IntegrationCredentialsStore
) -> str:
    """Return the Ayrshare profile key for *user_id*, creating one if needed.

    **Read-only for the legacy field.**  When
    ``managed_credentials.ayrshare_profile_key`` is populated (pre-migration
    data), it is reused verbatim so existing linked socials keep working.
    The legacy field is *not* cleared here — that happens in
    :meth:`AyrshareManagedProvider.post_provision`, which runs only after
    the managed credential is durably stored.  If this function cleared
    eagerly and the subsequent ``add_managed_credential`` failed, a retry
    would see an empty legacy field and create a *fresh* Ayrshare profile,
    orphaning the user's linked social accounts.
    """
    user_integrations = await store._get_user_integrations(user_id)
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

    logger.debug("[ayrshare] Creating profile for user %s", user_id)
    profile = await client.create_profile(
        title=f"User {user_id}", messaging_active=True
    )
    return profile.profileKey


def _settings_available() -> bool:
    """True when Ayrshare org-level secrets are configured.

    Exposed so on-demand callers (e.g. the SSO-URL route) can pre-flight the
    config before calling :func:`~backend.integrations.managed_credentials.ensure_managed_credential`.
    """
    settings = Settings()
    return bool(settings.secrets.ayrshare_api_key and settings.secrets.ayrshare_jwt_key)
