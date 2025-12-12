"""
Credential Grants endpoints for external OAuth clients.

Allows external applications to:
- List their credential grants (metadata only, NOT credential values)
- Get grant details
- Delete credentials via grants (if permitted)

Credentials are NEVER returned to external applications.
"""

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Security
from pydantic import BaseModel

from backend.data import credential_grants as grants_db
from backend.data.db import prisma
from backend.integrations.credentials_store import IntegrationCredentialsStore
from backend.server.external.oauth_middleware import OAuthTokenInfo, require_scope

grants_router = APIRouter(prefix="/grants", tags=["grants"])


# ================================================================
# Response Models
# ================================================================


class GrantSummary(BaseModel):
    """Summary of a credential grant (returned in list endpoints)."""

    id: str
    provider: str
    granted_scopes: list[str]
    permissions: list[str]
    created_at: datetime
    last_used_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None


class GrantDetail(BaseModel):
    """Detailed grant information."""

    id: str
    provider: str
    credential_id: str
    granted_scopes: list[str]
    permissions: list[str]
    created_at: datetime
    updated_at: datetime
    last_used_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None


# ================================================================
# Endpoints
# ================================================================


@grants_router.get("/", response_model=list[GrantSummary])
async def list_grants(
    token: OAuthTokenInfo = Security(require_scope("integrations:list")),
) -> list[GrantSummary]:
    """
    List all active credential grants for this client and user.

    Returns grant metadata but NOT credential values.
    Credentials are never exposed to external applications.
    """
    # Get the OAuth client's database ID from the public client_id
    client = await prisma.oauthclient.find_unique(where={"clientId": token.client_id})
    if not client:
        raise HTTPException(status_code=400, detail="Invalid client")

    grants = await grants_db.get_grants_for_user_client(
        user_id=token.user_id,
        client_id=client.id,
        include_revoked=False,
        include_expired=False,
    )

    return [
        GrantSummary(
            id=grant.id,
            provider=grant.provider,
            granted_scopes=grant.grantedScopes,
            permissions=[p.value for p in grant.permissions],
            created_at=grant.createdAt,
            last_used_at=grant.lastUsedAt,
            expires_at=grant.expiresAt,
        )
        for grant in grants
    ]


@grants_router.get("/{grant_id}", response_model=GrantDetail)
async def get_grant(
    grant_id: str,
    token: OAuthTokenInfo = Security(require_scope("integrations:list")),
) -> GrantDetail:
    """
    Get detailed information about a specific grant.

    Returns grant metadata including scopes and permissions.
    Does NOT return the credential value.
    """
    # Get the OAuth client's database ID
    client = await prisma.oauthclient.find_unique(where={"clientId": token.client_id})
    if not client:
        raise HTTPException(status_code=400, detail="Invalid client")

    grant = await grants_db.get_credential_grant(
        grant_id=grant_id,
        user_id=token.user_id,
        client_id=client.id,
    )

    if not grant:
        raise HTTPException(status_code=404, detail="Grant not found")

    # Check if expired
    if grant.expiresAt and grant.expiresAt < datetime.now(timezone.utc):
        raise HTTPException(status_code=404, detail="Grant has expired")

    # Check if revoked
    if grant.revokedAt:
        raise HTTPException(status_code=404, detail="Grant has been revoked")

    return GrantDetail(
        id=grant.id,
        provider=grant.provider,
        credential_id=grant.credentialId,
        granted_scopes=grant.grantedScopes,
        permissions=[p.value for p in grant.permissions],
        created_at=grant.createdAt,
        updated_at=grant.updatedAt,
        last_used_at=grant.lastUsedAt,
        expires_at=grant.expiresAt,
        revoked_at=grant.revokedAt,
    )


@grants_router.delete("/{grant_id}/credential")
async def delete_credential_via_grant(
    grant_id: str,
    token: OAuthTokenInfo = Security(require_scope("integrations:delete")),
) -> dict:
    """
    Delete the underlying credential associated with a grant.

    This requires the grant to have the DELETE permission.
    Deleting the credential also invalidates all grants for that credential.
    """
    from prisma.enums import CredentialGrantPermission

    # Get the OAuth client's database ID
    client = await prisma.oauthclient.find_unique(where={"clientId": token.client_id})
    if not client:
        raise HTTPException(status_code=400, detail="Invalid client")

    # Get the grant
    grant = await grants_db.get_credential_grant(
        grant_id=grant_id,
        user_id=token.user_id,
        client_id=client.id,
    )

    if not grant:
        raise HTTPException(status_code=404, detail="Grant not found")

    # Check if grant is valid
    if grant.revokedAt:
        raise HTTPException(status_code=400, detail="Grant has been revoked")

    if grant.expiresAt and grant.expiresAt < datetime.now(timezone.utc):
        raise HTTPException(status_code=400, detail="Grant has expired")

    # Check DELETE permission
    if CredentialGrantPermission.DELETE not in grant.permissions:
        raise HTTPException(
            status_code=403,
            detail="Grant does not have DELETE permission for this credential",
        )

    # Delete the credential using the credentials store
    try:
        creds_store = IntegrationCredentialsStore()
        await creds_store.delete_creds_by_id(
            user_id=token.user_id,
            credentials_id=grant.credentialId,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete credential: {str(e)}",
        )

    # Revoke all grants for this credential
    await grants_db.revoke_grants_for_credential(
        user_id=token.user_id,
        credential_id=grant.credentialId,
    )

    return {"message": "Credential deleted successfully"}
