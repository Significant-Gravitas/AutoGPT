"""
V2 External API - Integration Helpers

Shared logic for credential-related operations.
"""

from backend.integrations.creds_manager import IntegrationCredentialsManager

from ..models import CredentialInfo, CredentialRequirement

creds_manager = IntegrationCredentialsManager()


async def get_credential_requirements(
    creds_schema: dict,
    user_id: str,
) -> list[CredentialRequirement]:
    """
    Extract credential requirements from a graph's credentials input schema
    and match them against the user's existing credentials.
    """
    all_credentials = await creds_manager.store.get_all_creds(user_id)

    requirements = []
    for field_name, field_schema in creds_schema.get("properties", {}).items():
        providers: list[str] = []
        if "anyOf" in field_schema:
            for option in field_schema["anyOf"]:
                if "provider" in option:
                    providers.append(option["provider"])
        elif "provider" in field_schema:
            providers.append(field_schema["provider"])

        for provider in providers:
            matching = [
                CredentialInfo.from_internal(c)
                for c in all_credentials
                if c.provider.lower() == provider.lower()
            ]

            requirements.append(
                CredentialRequirement(
                    provider=provider,
                    required_scopes=[],
                    matching_credentials=matching,
                )
            )

    return requirements
