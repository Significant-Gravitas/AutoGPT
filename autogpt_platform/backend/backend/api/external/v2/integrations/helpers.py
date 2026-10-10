"""
V2 External API - Integration Helpers

Shared logic for credential-related operations.
"""

from typing import Any

from backend.integrations.creds_manager import IntegrationCredentialsManager

from ..models import CredentialInfo, CredentialRequirement

creds_manager = IntegrationCredentialsManager()


async def get_credential_requirements(
    creds_schema: dict[str, Any],
    user_id: str,
) -> list[CredentialRequirement]:
    """
    Extract credential requirements from a graph's credentials input schema
    and match them against the user's existing credentials.
    """
    all_credentials = await creds_manager.store.get_all_creds(user_id)

    return [
        CredentialRequirement(
            field_name=field_name,
            provider=provider,
            supported_types=list(field_schema.get("credentials_types") or []),
            required_scopes=list(field_schema.get("credentials_scopes") or []),
            matching_credentials=[
                CredentialInfo.from_internal(c)
                for c in all_credentials
                if c.provider.lower() == provider.lower()
            ],
        )
        for field_name, field_schema in (creds_schema.get("properties") or {}).items()
        for provider in field_schema.get("credentials_provider") or []
    ]
