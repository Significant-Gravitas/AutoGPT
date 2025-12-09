"""
Credential Grant data layer.

Handles database operations for credential grants which allow OAuth clients
to use credentials on behalf of users.
"""

from datetime import datetime, timezone
from typing import Optional

from prisma.enums import CredentialGrantPermission
from prisma.models import CredentialGrant

from backend.data.db import prisma


async def create_credential_grant(
    user_id: str,
    client_id: str,
    credential_id: str,
    provider: str,
    granted_scopes: list[str],
    permissions: list[CredentialGrantPermission],
    expires_at: Optional[datetime] = None,
) -> CredentialGrant:
    """
    Create a new credential grant.

    Args:
        user_id: ID of the user granting access
        client_id: Database ID of the OAuth client
        credential_id: ID of the credential being granted
        provider: Provider name (e.g., "google", "github")
        granted_scopes: List of integration scopes granted
        permissions: List of permissions (USE, DELETE)
        expires_at: Optional expiration datetime

    Returns:
        Created CredentialGrant
    """
    return await prisma.credentialgrant.create(
        data={
            "userId": user_id,
            "clientId": client_id,
            "credentialId": credential_id,
            "provider": provider,
            "grantedScopes": granted_scopes,
            "permissions": permissions,
            "expiresAt": expires_at,
        }
    )


async def get_credential_grant(
    grant_id: str,
    user_id: Optional[str] = None,
    client_id: Optional[str] = None,
) -> Optional[CredentialGrant]:
    """
    Get a credential grant by ID.

    Args:
        grant_id: Grant ID
        user_id: Optional user ID filter
        client_id: Optional client database ID filter

    Returns:
        CredentialGrant or None
    """
    where: dict[str, str] = {"id": grant_id}
    if user_id:
        where["userId"] = user_id
    if client_id:
        where["clientId"] = client_id

    return await prisma.credentialgrant.find_first(where=where)  # type: ignore[arg-type]


async def get_grants_for_user_client(
    user_id: str,
    client_id: str,
    include_revoked: bool = False,
    include_expired: bool = False,
) -> list[CredentialGrant]:
    """
    Get all credential grants for a user-client pair.

    Args:
        user_id: User ID
        client_id: Client database ID
        include_revoked: Include revoked grants
        include_expired: Include expired grants

    Returns:
        List of CredentialGrant objects
    """
    where: dict[str, str | None] = {
        "userId": user_id,
        "clientId": client_id,
    }

    if not include_revoked:
        where["revokedAt"] = None

    grants = await prisma.credentialgrant.find_many(
        where=where,  # type: ignore[arg-type]
        order={"createdAt": "desc"},
    )

    # Filter expired if needed
    if not include_expired:
        now = datetime.now(timezone.utc)
        grants = [g for g in grants if g.expiresAt is None or g.expiresAt > now]

    return grants


async def get_grants_for_credential(
    user_id: str,
    credential_id: str,
) -> list[CredentialGrant]:
    """
    Get all active grants for a specific credential.

    Args:
        user_id: User ID
        credential_id: Credential ID

    Returns:
        List of active CredentialGrant objects
    """
    now = datetime.now(timezone.utc)

    grants = await prisma.credentialgrant.find_many(
        where={
            "userId": user_id,
            "credentialId": credential_id,
            "revokedAt": None,
        },
        include={"Client": True},
    )

    # Filter expired
    return [g for g in grants if g.expiresAt is None or g.expiresAt > now]


async def get_grant_by_credential_and_client(
    user_id: str,
    credential_id: str,
    client_id: str,
) -> Optional[CredentialGrant]:
    """
    Get the grant for a specific credential and client.

    Args:
        user_id: User ID
        credential_id: Credential ID
        client_id: Client database ID

    Returns:
        CredentialGrant or None
    """
    return await prisma.credentialgrant.find_first(
        where={
            "userId": user_id,
            "credentialId": credential_id,
            "clientId": client_id,
            "revokedAt": None,
        }
    )


async def update_grant_scopes(
    grant_id: str,
    granted_scopes: list[str],
) -> CredentialGrant:
    """
    Update the granted scopes for a credential grant.

    Args:
        grant_id: Grant ID
        granted_scopes: New list of granted scopes

    Returns:
        Updated CredentialGrant
    """
    result = await prisma.credentialgrant.update(
        where={"id": grant_id},
        data={"grantedScopes": granted_scopes},
    )
    if result is None:
        raise ValueError(f"Grant {grant_id} not found")
    return result


async def update_grant_last_used(grant_id: str) -> None:
    """
    Update the lastUsedAt timestamp for a grant.

    Args:
        grant_id: Grant ID
    """
    await prisma.credentialgrant.update(
        where={"id": grant_id},
        data={"lastUsedAt": datetime.now(timezone.utc)},
    )


async def revoke_grant(grant_id: str) -> CredentialGrant:
    """
    Revoke a credential grant.

    Args:
        grant_id: Grant ID

    Returns:
        Revoked CredentialGrant
    """
    result = await prisma.credentialgrant.update(
        where={"id": grant_id},
        data={"revokedAt": datetime.now(timezone.utc)},
    )
    if result is None:
        raise ValueError(f"Grant {grant_id} not found")
    return result


async def revoke_grants_for_credential(
    user_id: str,
    credential_id: str,
) -> int:
    """
    Revoke all grants for a specific credential.

    Args:
        user_id: User ID
        credential_id: Credential ID

    Returns:
        Number of grants revoked
    """
    return await prisma.credentialgrant.update_many(
        where={
            "userId": user_id,
            "credentialId": credential_id,
            "revokedAt": None,
        },
        data={"revokedAt": datetime.now(timezone.utc)},
    )


async def revoke_grants_for_client(
    user_id: str,
    client_id: str,
) -> int:
    """
    Revoke all grants for a specific client.

    Args:
        user_id: User ID
        client_id: Client database ID

    Returns:
        Number of grants revoked
    """
    return await prisma.credentialgrant.update_many(
        where={
            "userId": user_id,
            "clientId": client_id,
            "revokedAt": None,
        },
        data={"revokedAt": datetime.now(timezone.utc)},
    )


async def delete_grant(grant_id: str) -> None:
    """
    Permanently delete a credential grant.

    Args:
        grant_id: Grant ID
    """
    await prisma.credentialgrant.delete(where={"id": grant_id})


async def check_grant_permission(
    grant_id: str,
    required_permission: CredentialGrantPermission,
) -> bool:
    """
    Check if a grant has a specific permission.

    Args:
        grant_id: Grant ID
        required_permission: Permission to check

    Returns:
        True if grant has the permission
    """
    grant = await prisma.credentialgrant.find_unique(where={"id": grant_id})
    if not grant:
        return False

    return required_permission in grant.permissions


async def is_grant_valid(grant_id: str) -> bool:
    """
    Check if a grant is valid (not revoked and not expired).

    Args:
        grant_id: Grant ID

    Returns:
        True if grant is valid
    """
    grant = await prisma.credentialgrant.find_unique(where={"id": grant_id})
    if not grant:
        return False

    if grant.revokedAt:
        return False

    if grant.expiresAt and grant.expiresAt < datetime.now(timezone.utc):
        return False

    return True
