"""
V2 External API - Credential CRUD Endpoints

Provides endpoints for managing integration credentials.
"""

import logging
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Path, Query, Security
from prisma.enums import APIKeyPermission
from pydantic import SecretStr
from starlette.status import HTTP_201_CREATED

from backend.api.external.middleware import require_permission
from backend.data.auth.base import APIAuthorizationInfo
from backend.data.model import APIKeyCredentials

from ..models import (
    CredentialCreateRequest,
    CredentialDeleteResponse,
    CredentialInfo,
    CredentialListResponse,
)
from .helpers import creds_manager

logger = logging.getLogger(__name__)

credentials_router = APIRouter()


@credentials_router.get(
    path="/credentials",
    summary="List credentials",
)
async def list_credentials(
    provider: Optional[str] = Query(
        default=None,
        description="Filter by provider name (e.g., 'github', 'google')",
    ),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_INTEGRATIONS)
    ),
) -> CredentialListResponse:
    """
    List integration credentials for the authenticated user.

    Optionally filter by provider.
    """
    credentials = await creds_manager.store.get_all_creds(auth.user_id)

    if provider:
        credentials = [c for c in credentials if c.provider.lower() == provider.lower()]

    return CredentialListResponse(
        credentials=[CredentialInfo.from_internal(c) for c in credentials]
    )


@credentials_router.post(
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


@credentials_router.delete(
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
    existing = await creds_manager.store.get_creds_by_id(
        user_id=auth.user_id, credentials_id=credential_id
    )
    if not existing:
        raise HTTPException(
            status_code=404, detail=f"Credential #{credential_id} not found"
        )

    await creds_manager.delete(auth.user_id, credential_id)
    return CredentialDeleteResponse()
