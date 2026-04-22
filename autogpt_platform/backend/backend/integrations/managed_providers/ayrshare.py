"""Ayrshare managed credential provider.

Provisions a per-user Ayrshare profile key and stores it as a standard
``APIKeyCredentials(provider="ayrshare", is_managed=True)`` in the user's
credentials list.  This lets every Ayrshare block declare a normal
``credentials`` field and go through the same schema-driven resolution as
any other provider — no bespoke ``managed_credentials.ayrshare_profile_key``
side channel required.

Legacy compatibility: if the user already has a profile key in
``UserIntegrations.managed_credentials.ayrshare_profile_key`` (pre-migration
data), :meth:`AyrshareManagedProvider.provision` reuses it instead of
creating a new profile with Ayrshare.  The caller can then clear the legacy
field once the migrated credential is in place.

User-visible caveat: provisioning the profile creates the Ayrshare profile
but does not link any social accounts.  The user still needs to open the
Ayrshare SSO popup (exposed via ``/api/integrations/ayrshare/sso_url``) to
OAuth each social network; the block will return the Ayrshare API's
"not linked" error until they do.  That part remains platform UX, not a
credential concern.
"""

from __future__ import annotations

import logging

from pydantic import SecretStr

from backend.data.model import APIKeyCredentials, Credentials
from backend.data.user import get_user_integrations
from backend.integrations.ayrshare import AyrshareClient
from backend.integrations.managed_credentials import ManagedCredentialProvider
from backend.util.exceptions import MissingConfigError
from backend.util.settings import Settings

logger = logging.getLogger(__name__)


class AyrshareManagedProvider(ManagedCredentialProvider):
    provider_name = "ayrshare"

    async def is_available(self) -> bool:
        settings = Settings()
        return bool(
            settings.secrets.ayrshare_api_key and settings.secrets.ayrshare_jwt_key
        )

    async def provision(self, user_id: str) -> Credentials:
        profile_key = await _get_or_create_profile_key(user_id)
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


async def _get_or_create_profile_key(user_id: str) -> str:
    """Return the Ayrshare profile key for *user_id*, creating one if needed.

    Reuses a legacy ``managed_credentials.ayrshare_profile_key`` entry when
    present (pre-migration data), otherwise creates a fresh profile via the
    Ayrshare API.
    """
    user_integrations = await get_user_integrations(user_id)
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
