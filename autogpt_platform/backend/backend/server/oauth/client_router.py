"""
OAuth Client Management endpoints.

Implements self-service client registration and management:
- POST /oauth/clients - Register a new client
- GET /oauth/clients - List owned clients
- GET /oauth/clients/{client_id} - Get client details
- PATCH /oauth/clients/{client_id} - Update client
- DELETE /oauth/clients/{client_id} - Delete client
- POST /oauth/clients/{client_id}/rotate-secret - Rotate client secret
"""

import hashlib
import secrets

from autogpt_libs.auth import get_user_id
from fastapi import APIRouter, HTTPException, Security
from prisma.enums import OAuthClientStatus

from backend.data.db import prisma
from backend.server.oauth.models import (
    ClientResponse,
    ClientSecretResponse,
    OAuthScope,
    RegisterClientRequest,
    UpdateClientRequest,
)

client_router = APIRouter(prefix="/oauth/clients", tags=["oauth-clients"])


def _generate_client_id() -> str:
    """Generate a unique client ID."""
    return f"app_{secrets.token_urlsafe(16)}"


def _generate_client_secret() -> str:
    """Generate a secure client secret."""
    return secrets.token_urlsafe(32)


def _hash_secret(secret: str, salt: str) -> str:
    """Hash a client secret with salt."""
    return hashlib.sha256(f"{salt}{secret}".encode()).hexdigest()


def _client_to_response(client) -> ClientResponse:
    """Convert Prisma client to response model."""
    return ClientResponse(
        id=client.id,
        client_id=client.clientId,
        client_type=client.clientType,
        name=client.name,
        description=client.description,
        logo_url=client.logoUrl,
        homepage_url=client.homepageUrl,
        privacy_policy_url=client.privacyPolicyUrl,
        terms_of_service_url=client.termsOfServiceUrl,
        redirect_uris=client.redirectUris,
        allowed_scopes=client.allowedScopes,
        webhook_domains=client.webhookDomains,
        status=client.status.value,
        created_at=client.createdAt,
        updated_at=client.updatedAt,
    )


# Default allowed scopes for new clients
DEFAULT_ALLOWED_SCOPES = [
    OAuthScope.OPENID.value,
    OAuthScope.PROFILE.value,
    OAuthScope.EMAIL.value,
    OAuthScope.INTEGRATIONS_LIST.value,
    OAuthScope.INTEGRATIONS_CONNECT.value,
    OAuthScope.INTEGRATIONS_DELETE.value,
    OAuthScope.AGENTS_EXECUTE.value,
]


@client_router.post("/", response_model=ClientSecretResponse)
async def register_client(
    request: RegisterClientRequest,
    user_id: str = Security(get_user_id),
) -> ClientSecretResponse:
    """
    Register a new OAuth client.

    The client is immediately active (no admin approval required).
    For confidential clients, the client_secret is returned only once.
    """
    # Generate client credentials
    client_id = _generate_client_id()
    client_secret = None
    client_secret_hash = None
    client_secret_salt = None

    if request.client_type == "confidential":
        client_secret = _generate_client_secret()
        client_secret_salt = secrets.token_urlsafe(16)
        client_secret_hash = _hash_secret(client_secret, client_secret_salt)

    # Create client
    await prisma.oauthclient.create(
        data={  # type: ignore[typeddict-item]
            "clientId": client_id,
            "clientSecretHash": client_secret_hash,
            "clientSecretSalt": client_secret_salt,
            "clientType": request.client_type,
            "name": request.name,
            "description": request.description,
            "logoUrl": str(request.logo_url) if request.logo_url else None,
            "homepageUrl": str(request.homepage_url) if request.homepage_url else None,
            "privacyPolicyUrl": (
                str(request.privacy_policy_url) if request.privacy_policy_url else None
            ),
            "termsOfServiceUrl": (
                str(request.terms_of_service_url)
                if request.terms_of_service_url
                else None
            ),
            "redirectUris": request.redirect_uris,
            "allowedScopes": DEFAULT_ALLOWED_SCOPES,
            "webhookDomains": request.webhook_domains,
            "status": OAuthClientStatus.ACTIVE,
            "ownerId": user_id,
        }
    )

    return ClientSecretResponse(
        client_id=client_id,
        client_secret=client_secret or "",
    )


@client_router.get("/", response_model=list[ClientResponse])
async def list_clients(
    user_id: str = Security(get_user_id),
) -> list[ClientResponse]:
    """List all OAuth clients owned by the current user."""
    clients = await prisma.oauthclient.find_many(
        where={"ownerId": user_id},
        order={"createdAt": "desc"},
    )
    return [_client_to_response(c) for c in clients]


@client_router.get("/{client_id}", response_model=ClientResponse)
async def get_client(
    client_id: str,
    user_id: str = Security(get_user_id),
) -> ClientResponse:
    """Get details of a specific OAuth client."""
    client = await prisma.oauthclient.find_first(
        where={"clientId": client_id, "ownerId": user_id}
    )

    if not client:
        raise HTTPException(status_code=404, detail="Client not found")

    return _client_to_response(client)


@client_router.patch("/{client_id}", response_model=ClientResponse)
async def update_client(
    client_id: str,
    request: UpdateClientRequest,
    user_id: str = Security(get_user_id),
) -> ClientResponse:
    """Update an OAuth client."""
    client = await prisma.oauthclient.find_first(
        where={"clientId": client_id, "ownerId": user_id}
    )

    if not client:
        raise HTTPException(status_code=404, detail="Client not found")

    # Build update data
    update_data: dict[str, str | list[str] | None] = {}
    if request.name is not None:
        update_data["name"] = request.name
    if request.description is not None:
        update_data["description"] = request.description
    if request.logo_url is not None:
        update_data["logoUrl"] = str(request.logo_url)
    if request.homepage_url is not None:
        update_data["homepageUrl"] = str(request.homepage_url)
    if request.privacy_policy_url is not None:
        update_data["privacyPolicyUrl"] = str(request.privacy_policy_url)
    if request.terms_of_service_url is not None:
        update_data["termsOfServiceUrl"] = str(request.terms_of_service_url)
    if request.redirect_uris is not None:
        update_data["redirectUris"] = request.redirect_uris
    if request.webhook_domains is not None:
        update_data["webhookDomains"] = request.webhook_domains

    if not update_data:
        return _client_to_response(client)

    updated = await prisma.oauthclient.update(
        where={"id": client.id},
        data=update_data,  # type: ignore[arg-type]
    )

    return _client_to_response(updated)


@client_router.delete("/{client_id}")
async def delete_client(
    client_id: str,
    user_id: str = Security(get_user_id),
) -> dict:
    """
    Delete an OAuth client.

    This will also revoke all tokens and authorizations for this client.
    """
    client = await prisma.oauthclient.find_first(
        where={"clientId": client_id, "ownerId": user_id}
    )

    if not client:
        raise HTTPException(status_code=404, detail="Client not found")

    # Delete cascades will handle tokens, codes, and authorizations
    await prisma.oauthclient.delete(where={"id": client.id})

    return {"status": "deleted", "client_id": client_id}


@client_router.post("/{client_id}/rotate-secret", response_model=ClientSecretResponse)
async def rotate_client_secret(
    client_id: str,
    user_id: str = Security(get_user_id),
) -> ClientSecretResponse:
    """
    Rotate the client secret for a confidential client.

    The new secret is returned only once. All existing tokens remain valid.
    """
    client = await prisma.oauthclient.find_first(
        where={"clientId": client_id, "ownerId": user_id}
    )

    if not client:
        raise HTTPException(status_code=404, detail="Client not found")

    if client.clientType != "confidential":
        raise HTTPException(
            status_code=400,
            detail="Cannot rotate secret for public clients",
        )

    # Generate new secret
    new_secret = _generate_client_secret()
    new_salt = secrets.token_urlsafe(16)
    new_hash = _hash_secret(new_secret, new_salt)

    await prisma.oauthclient.update(
        where={"id": client.id},
        data={
            "clientSecretHash": new_hash,
            "clientSecretSalt": new_salt,
        },
    )

    return ClientSecretResponse(
        client_id=client_id,
        client_secret=new_secret,
    )


@client_router.post("/{client_id}/suspend")
async def suspend_client(
    client_id: str,
    user_id: str = Security(get_user_id),
) -> ClientResponse:
    """Suspend an OAuth client (prevents new authorizations)."""
    client = await prisma.oauthclient.find_first(
        where={"clientId": client_id, "ownerId": user_id}
    )

    if not client:
        raise HTTPException(status_code=404, detail="Client not found")

    updated = await prisma.oauthclient.update(
        where={"id": client.id},
        data={"status": OAuthClientStatus.SUSPENDED},
    )

    return _client_to_response(updated)


@client_router.post("/{client_id}/activate")
async def activate_client(
    client_id: str,
    user_id: str = Security(get_user_id),
) -> ClientResponse:
    """Reactivate a suspended OAuth client."""
    client = await prisma.oauthclient.find_first(
        where={"clientId": client_id, "ownerId": user_id}
    )

    if not client:
        raise HTTPException(status_code=404, detail="Client not found")

    updated = await prisma.oauthclient.update(
        where={"id": client.id},
        data={"status": OAuthClientStatus.ACTIVE},
    )

    return _client_to_response(updated)
