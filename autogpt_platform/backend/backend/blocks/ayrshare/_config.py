"""Shared provider config for Ayrshare social-media blocks.

The "credential" exposed to blocks is the **per-user Ayrshare profile key**,
not the org-level ``AYRSHARE_API_KEY``.  Profile keys are provisioned per
user by :class:`~backend.integrations.managed_providers.ayrshare.AyrshareManagedProvider`
and stored in the normal credentials list with ``is_managed=True``, so every
Ayrshare block fits the standard credential flow:

    credentials: CredentialsMetaInput = ayrshare.credentials_field(...)

``run_block`` / ``resolve_block_credentials`` take care of the rest.

``with_managed_api_key()`` registers ``api_key`` as a supported auth type
without the env-var-backed default credential that ``with_api_key()`` would
create — the org-level ``AYRSHARE_API_KEY`` is the admin key and must never
reach a block as a "profile key".
"""

from backend.sdk import ProviderBuilder

ayrshare = (
    ProviderBuilder("ayrshare")
    .with_description("Post to every social network")
    .with_managed_api_key()
    .build()
)
