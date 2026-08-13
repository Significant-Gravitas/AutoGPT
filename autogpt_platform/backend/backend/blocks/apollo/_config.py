"""Provider registration for Apollo.

Registers the provider description shown in the settings integrations UI.
Apollo doesn't use a full :class:`ProviderBuilder` chain (auth is set up in
``_auth.py``), so this file only declares metadata.
"""

from backend.sdk import ProviderBuilder

apollo = (
    ProviderBuilder("apollo")
    .with_description("Sales intelligence and prospecting")
    .with_supported_auth_types("api_key")
    .build()
)
