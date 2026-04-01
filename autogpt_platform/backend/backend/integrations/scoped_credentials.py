"""Scoped credential store using the IntegrationCredential table.

Provides the new credential resolution path (USER → WORKSPACE → ORG)
using the IntegrationCredential table introduced in PR1. During the
dual-read transition period, callers should try this store first and
fall back to the legacy IntegrationCredentialsStore.

This store is used alongside the existing credentials_store.py which
reads from the User.integrations encrypted blob.
"""

import logging
from typing import Optional

from backend.data.db import prisma
from backend.util.encryption import JSONCryptor

logger = logging.getLogger(__name__)

_cryptor = JSONCryptor()


async def get_scoped_credentials(
    user_id: str,
    organization_id: str,
    workspace_id: str | None = None,
    provider: str | None = None,
) -> list[dict]:
    """Get credentials visible to the user in the current org/workspace context.

    Resolution order (per plan 3D):
    1. USER credentials created by this user in this org
    2. WORKSPACE credentials for the active workspace (if workspace is set)
    3. ORG credentials for the active org

    Returns a list of credential metadata dicts (not decrypted payloads).
    """
    results: list[dict] = []

    # 1. User-scoped credentials
    user_where: dict = {
        "organizationId": organization_id,
        "ownerType": "USER",
        "ownerId": user_id,
        "status": "active",
    }
    if provider:
        user_where["provider"] = provider

    user_creds = await prisma.integrationcredential.find_many(where=user_where)
    for c in user_creds:
        results.append(_cred_to_metadata(c, scope="USER"))

    # 2. Workspace-scoped credentials (only if workspace is active)
    if workspace_id:
        ws_where: dict = {
            "organizationId": organization_id,
            "ownerType": "WORKSPACE",
            "ownerId": workspace_id,
            "status": "active",
        }
        if provider:
            ws_where["provider"] = provider

        ws_creds = await prisma.integrationcredential.find_many(where=ws_where)
        for c in ws_creds:
            results.append(_cred_to_metadata(c, scope="WORKSPACE"))

    # 3. Org-scoped credentials
    org_where: dict = {
        "organizationId": organization_id,
        "ownerType": "ORG",
        "ownerId": organization_id,
        "status": "active",
    }
    if provider:
        org_where["provider"] = provider

    org_creds = await prisma.integrationcredential.find_many(where=org_where)
    for c in org_creds:
        results.append(_cred_to_metadata(c, scope="ORG"))

    return results


async def get_credential_by_id(
    credential_id: str,
    user_id: str,
    organization_id: str,
    workspace_id: str | None = None,
    decrypt: bool = False,
) -> Optional[dict]:
    """Get a specific credential by ID if the user has access.

    Access rules:
    - USER creds: only the creating user can access
    - WORKSPACE creds: any workspace member can access (verified by caller)
    - ORG creds: any org member can access (verified by caller)
    """
    cred = await prisma.integrationcredential.find_unique(where={"id": credential_id})
    if cred is None or cred.organizationId != organization_id:
        return None

    # Access check
    if cred.ownerType == "USER" and cred.createdByUserId != user_id:
        return None

    result = _cred_to_metadata(cred, scope=cred.ownerType)
    if decrypt:
        result["payload"] = _cryptor.decrypt(cred.encryptedPayload)

    return result


async def create_credential(
    organization_id: str,
    owner_type: str,  # USER, WORKSPACE, ORG
    owner_id: str,  # userId, workspaceId, or orgId
    provider: str,
    credential_type: str,
    display_name: str,
    payload: dict,
    user_id: str,
    expires_at=None,
    metadata: dict | None = None,
) -> dict:
    """Create a new scoped credential."""
    encrypted = _cryptor.encrypt(payload)

    cred = await prisma.integrationcredential.create(
        data={
            "organizationId": organization_id,
            "ownerType": owner_type,
            "ownerId": owner_id,
            "provider": provider,
            "credentialType": credential_type,
            "displayName": display_name,
            "encryptedPayload": encrypted,
            "createdByUserId": user_id,
            "expiresAt": expires_at,
            "metadata": metadata,
        }
    )

    return _cred_to_metadata(cred, scope=owner_type)


async def delete_credential(
    credential_id: str, user_id: str, organization_id: str
) -> None:
    """Soft-delete a credential by setting status to 'revoked'."""
    cred = await prisma.integrationcredential.find_unique(where={"id": credential_id})
    if cred is None or cred.organizationId != organization_id:
        raise ValueError(f"Credential {credential_id} not found")

    # Only the creator or an admin can delete (admin check done at route level)
    await prisma.integrationcredential.update(
        where={"id": credential_id},
        data={"status": "revoked"},
    )


def _cred_to_metadata(cred, scope: str) -> dict:
    """Convert a Prisma IntegrationCredential to a metadata dict."""
    return {
        "id": cred.id,
        "provider": cred.provider,
        "credentialType": cred.credentialType,
        "displayName": cred.displayName,
        "scope": scope,
        "createdByUserId": cred.createdByUserId,
        "lastUsedAt": cred.lastUsedAt,
        "expiresAt": cred.expiresAt,
        "createdAt": cred.createdAt,
    }
