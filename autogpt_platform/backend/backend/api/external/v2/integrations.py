"""
V2 External API - Integrations Endpoints

Provides access to user's integration credentials.
"""

import logging
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Path, Security
from prisma.enums import APIKeyPermission
from pydantic import SecretStr
from starlette.status import HTTP_201_CREATED

from backend.api.external.middleware import require_permission
from backend.api.features.library import db as library_db
from backend.data import graph as graph_db
from backend.data.auth.base import APIAuthorizationInfo
from backend.data.model import APIKeyCredentials
from backend.integrations.creds_manager import IntegrationCredentialsManager

from .models import (
    CredentialCreateRequest,
    CredentialDeleteResponse,
    CredentialInfo,
    CredentialListResponse,
    CredentialRequirement,
    CredentialRequirementsResponse,
)

logger = logging.getLogger(__name__)

integrations_router = APIRouter()
creds_manager = IntegrationCredentialsManager()


# ============================================================================
# Endpoints
# ============================================================================


@integrations_router.get(
    path="/credentials",
    summary="List all credentials",
    response_model=CredentialListResponse,
)
async def list_credentials(
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_INTEGRATIONS)
    ),
) -> CredentialListResponse:
    """
    List all integration credentials for the authenticated user.

    This returns all OAuth credentials the user has connected, across
    all integration providers.
    """
    credentials = await creds_manager.store.get_all_creds(auth.user_id)

    return CredentialListResponse(
        credentials=[CredentialInfo.from_internal(c) for c in credentials]
    )


@integrations_router.get(
    path="/credentials/{provider}",
    summary="List credentials by provider",
    response_model=CredentialListResponse,
)
async def list_credentials_by_provider(
    provider: str = Path(description="Provider name (e.g., 'github', 'google')"),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_INTEGRATIONS)
    ),
) -> CredentialListResponse:
    """
    List integration credentials for a specific provider.
    """
    all_credentials = await creds_manager.store.get_all_creds(auth.user_id)

    # Filter by provider
    filtered = [c for c in all_credentials if c.provider.lower() == provider.lower()]

    return CredentialListResponse(
        credentials=[CredentialInfo.from_internal(c) for c in filtered]
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
                CredentialInfo.from_internal(c)
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

    return CredentialRequirementsResponse(requirements=requirements)


# ============================================================================
# Endpoints - Credential Management
# ============================================================================


@integrations_router.post(
    path="/credentials",
    summary="Create an API key credential",
    status_code=HTTP_201_CREATED,
)
async def create_credential(
    request: CredentialCreateRequest,
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.MANAGE_INTEGRATIONS)
    ),
) -> CredentialInfo:
    """
    Create a new API key credential.

    Only API key credentials can be created via the external API.
    OAuth credentials must be created via the OAuth flow in the web UI.
    """
    credentials = APIKeyCredentials(
        id=str(uuid4()),
        provider=request.provider,
        title=request.title,
        api_key=SecretStr(request.api_key),
    )

    await creds_manager.create(auth.user_id, credentials)
    return CredentialInfo.from_internal(credentials)


@integrations_router.delete(
    path="/credentials/{credential_id}",
    summary="Delete a credential",
)
async def delete_credential(
    credential_id: str = Path(description="Credential ID"),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.DELETE_INTEGRATIONS)
    ),
) -> CredentialDeleteResponse:
    """
    Delete an integration credential.

    This permanently removes the credential. Any agents using this
    credential will fail on their next execution.
    """
    # Verify the credential exists
    existing = await creds_manager.store.get_creds_by_id(
        user_id=auth.user_id, credentials_id=credential_id
    )
    if not existing:
        raise HTTPException(
            status_code=404, detail=f"Credential #{credential_id} not found"
        )

    await creds_manager.delete(auth.user_id, credential_id)
    return CredentialDeleteResponse()
