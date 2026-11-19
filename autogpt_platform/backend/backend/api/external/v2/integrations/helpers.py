"""
V2 External API - Integration Helpers

Shared logic for credential-related operations.
"""

from collections import defaultdict
from typing import Any

from backend.integrations.creds_manager import IntegrationCredentialsManager

from ..models import CredentialInfo, CredentialRequirement
from ..pagination import Page, PageRequest

creds_manager = IntegrationCredentialsManager()


async def get_credential_requirements(
    creds_schema: dict[str, Any],
    user_id: str,
    page: PageRequest,
) -> Page[CredentialRequirement]:
    """
    Extract the requested page of credential requirements from a graph's
    credentials input schema and match them against the user's credentials.
    """
    required = [
        (field_name, field_schema, provider)
        for field_name, field_schema in (creds_schema.get("properties") or {}).items()
        for provider in field_schema.get("credentials_provider") or []
    ]
    start = (page.page - 1) * page.limit
    on_page = required[start : start + page.limit]

    # One CredentialInfo per credential, shared by every requirement it matches.
    providers = {provider.lower() for _, _, provider in on_page}
    matching: defaultdict[str, list[CredentialInfo]] = defaultdict(list)
    if providers:
        for c in await creds_manager.store.get_all_creds(user_id):
            if c.provider.lower() in providers:
                matching[c.provider.lower()].append(CredentialInfo.from_internal(c))

    return page.paged(
        [
            CredentialRequirement(
                field_name=field_name,
                provider=provider,
                supported_types=list(field_schema.get("credentials_types") or []),
                required_scopes=list(field_schema.get("credentials_scopes") or []),
                matching_credentials=matching[provider.lower()],
            )
            for field_name, field_schema, provider in on_page
        ],
        total_count=len(required),
    )
