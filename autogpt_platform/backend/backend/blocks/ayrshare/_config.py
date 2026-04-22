"""Shared provider config for Ayrshare social-media blocks.

The "credential" exposed to blocks is the **per-user Ayrshare profile key**,
not the org-level ``AYRSHARE_API_KEY``.  Profile keys are provisioned per
user by :class:`~backend.integrations.managed_providers.ayrshare.AyrshareManagedProvider`
and stored in the normal credentials list with ``is_managed=True``, so every
Ayrshare block fits the standard credential flow:

    credentials: CredentialsMetaInput = ayrshare.credentials_field(...)

and ``run_block`` / ``resolve_block_credentials`` take care of the rest.  No
more side-channel ``managed_credentials.ayrshare_profile_key`` lookups.

We intentionally do NOT call ``with_api_key("AYRSHARE_API_KEY", ...)`` — that
would create an org-level default credential whose ``api_key`` is the admin
key, which would be wrong to hand to a block as a "profile key".  Passing an
unused env-var name keeps the auth-type registered (``api_key``) without
materialising a default credential.
"""

from backend.sdk import ProviderBuilder

ayrshare = (
    ProviderBuilder("ayrshare")
    .with_api_key("_AYRSHARE_PROFILE_KEY_UNUSED", "Ayrshare Profile Key")
    .build()
)
