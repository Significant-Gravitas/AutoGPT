"""
V2 External API - Integrations Endpoints

Provides access to user's integration credentials.
"""

import logging

from fastapi import APIRouter, HTTPException, Path, Security
from prisma.enums import APIKeyPermission

from backend.api.external.middleware import require_permission
from backend.api.features.library import db as library_db
from backend.data import graph as graph_db
from backend.data.auth.base import APIAuthorizationInfo
from backend.data.model import Credentials, OAuth2Credentials
from backend.integrations.creds_manager import IntegrationCredentialsManager

from .models import (
    Credential,
    CredentialRequirement,
    CredentialRequirementsResponse,
    CredentialsListResponse,
)

logger = logging.getLogger(__name__)

integrations_router = APIRouter()
creds_manager = IntegrationCredentialsManager()


# ============================================================================
# Conversion Functions
# ============================================================================


def _convert_credential(cred: Credentials) -> Credential:
    """Convert internal credential to v2 API model."""
    scopes: list[str] = []
    if isinstance(cred, OAuth2Credentials):
        scopes = cred.scopes or []

    return Credential(
        id=cred.id,
        provider=cred.provider,
        title=cred.title,
        scopes=scopes,
    )


# ============================================================================
# Endpoints
# ============================================================================


@integrations_router.get(
    path="/credentials",
    summary="List all credentials",
    response_model=CredentialsListResponse,
)
async def list_credentials(
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_INTEGRATIONS)
    ),
) -> CredentialsListResponse:
    """
    List all integration credentials for the authenticated user.

    This returns all OAuth credentials the user has connected, across
    all integration providers.
    """
    credentials = await creds_manager.store.get_all_creds(auth.user_id)

    return CredentialsListResponse(
        credentials=[_convert_credential(c) for c in credentials]
    )


@integrations_router.get(
    path="/credentials/{provider}",
    summary="List credentials by provider",
    response_model=CredentialsListResponse,
)
async def list_credentials_by_provider(
    provider: str = Path(description="Provider name (e.g., 'github', 'google')"),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_INTEGRATIONS)
    ),
) -> CredentialsListResponse:
    """
    List integration credentials for a specific provider.
    """
    all_credentials = await creds_manager.store.get_all_creds(auth.user_id)

    # Filter by provider
    filtered = [c for c in all_credentials if c.provider.lower() == provider.lower()]

    return CredentialsListResponse(
        credentials=[_convert_credential(c) for c in filtered]
    )


@integrations_router.get(
    path="/graphs/{graph_id}/credentials",
    summary="List credentials matching graph requirements",
    response_model=CredentialRequirementsResponse,
)
async def list_graph_credential_requirements(
    graph_id: str = Path(description="Graph ID"),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_INTEGRATIONS)
    ),
) -> CredentialRequirementsResponse:
    """
    List credential requirements for a graph and matching user credentials.

    This helps identify which credentials the user needs to provide
    when executing a graph.
    """
    # Get the graph
    graph = await graph_db.get_graph(
        graph_id=graph_id,
        version=None,  # Active version
        user_id=auth.user_id,
        include_subgraphs=True,
    )
    if not graph:
        raise HTTPException(status_code=404, detail=f"Graph #{graph_id} not found")

    # Get the credentials input schema which contains provider requirements
    creds_schema = graph.credentials_input_schema
    all_credentials = await creds_manager.store.get_all_creds(auth.user_id)

    requirements = []
    for field_name, field_schema in creds_schema.get("properties", {}).items():
        # Extract provider from schema
        # The schema structure varies, but typically has provider info
        providers = []
        if "anyOf" in field_schema:
            for option in field_schema["anyOf"]:
                if "provider" in option:
                    providers.append(option["provider"])
        elif "provider" in field_schema:
            providers.append(field_schema["provider"])

        for provider in providers:
            # Find matching credentials
            matching = [
                _convert_credential(c)
                for c in all_credentials
                if c.provider.lower() == provider.lower()
            ]

            requirements.append(
                CredentialRequirement(
                    provider=provider,
                    required_scopes=[],  # Would need to extract from schema
                    matching_credentials=matching,
                )
            )

    return CredentialRequirementsResponse(requirements=requirements)


@integrations_router.get(
    path="/library/{agent_id}/credentials",
    summary="List credentials matching library agent requirements",
    response_model=CredentialRequirementsResponse,
)
async def list_library_agent_credential_requirements(
    agent_id: str = Path(description="Library agent ID"),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_INTEGRATIONS)
    ),
) -> CredentialRequirementsResponse:
    """
    List credential requirements for a library agent and matching user credentials.

    This helps identify which credentials the user needs to provide
    when executing an agent from their library.
    """
    # Get the library agent
    try:
        library_agent = await library_db.get_library_agent(
            id=agent_id,
            user_id=auth.user_id,
        )
    except Exception:
        raise HTTPException(status_code=404, detail=f"Agent #{agent_id} not found")

    # Get the underlying graph
    graph = await graph_db.get_graph(
        graph_id=library_agent.graph_id,
        version=library_agent.graph_version,
        user_id=auth.user_id,
        include_subgraphs=True,
    )
    if not graph:
        raise HTTPException(
            status_code=404,
            detail=f"Graph for agent #{agent_id} not found",
        )

    # Get the credentials input schema
    creds_schema = graph.credentials_input_schema
    all_credentials = await creds_manager.store.get_all_creds(auth.user_id)

    requirements = []
    for field_name, field_schema in creds_schema.get("properties", {}).items():
        # Extract provider from schema
        providers = []
        if "anyOf" in field_schema:
            for option in field_schema["anyOf"]:
                if "provider" in option:
                    providers.append(option["provider"])
        elif "provider" in field_schema:
            providers.append(field_schema["provider"])

        for provider in providers:
            # Find matching credentials
            matching = [
                _convert_credential(c)
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

    return CredentialRequirementsResponse(requirements=requirements)
