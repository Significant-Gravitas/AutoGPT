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
from uuid import uuid4

from backend.data.db import prisma
from backend.util.encryption import JSONCryptor

logger = logging.getLogger(__name__)

_cryptor = JSONCryptor()


async def get_scoped_credentials(
    user_id: str,
    organization_id: str,
    team_id: str | None = None,
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
    if team_id:
        ws_where: dict = {
            "organizationId": organization_id,
            "ownerType": "TEAM",
            "ownerId": team_id,
            "status": "active",
        }
        if provider:
            ws_where["provider"] = provider

        ws_creds = await prisma.integrationcredential.find_many(where=ws_where)
        for c in ws_creds:
            results.append(_cred_to_metadata(c, scope="TEAM"))

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
    team_id: str | None = None,
    decrypt: bool = False,
) -> Optional[dict]:
    """Get a specific credential by ID if the user has access.

    Access rules (enforced HERE, not trusted to callers):
    - USER creds: only the creating user can access
    - TEAM creds: only via the matching active team, or verified team
      membership when the caller's active context is a different team
    - ORG creds: any org member (``organization_id`` comes from a
      membership-verified RequestContext)
    """
    cred = await prisma.integrationcredential.find_unique(where={"id": credential_id})
    if cred is None or cred.organizationId != organization_id:
        return None

    if cred.ownerType == "USER" and cred.createdByUserId != user_id:
        return None

    if cred.ownerType == "TEAM" and cred.ownerId != team_id:
        # Not the active team — allow only if the user is actually a
        # member of the owning team. Without this check any org member
        # could fetch (and with decrypt=True, exfiltrate) another
        # team's secrets by ID.
        membership = await prisma.teammember.find_unique(
            where={"teamId_userId": {"teamId": cred.ownerId, "userId": user_id}}
        )
        if membership is None or membership.status != "ACTIVE":
            return None

    result = _cred_to_metadata(cred, scope=cred.ownerType)
    if decrypt:
        result["payload"] = _cryptor.decrypt(cred.encryptedPayload)

    return result


async def create_credential(
    organization_id: str,
    owner_type: str,  # USER, TEAM, ORG
    owner_id: str,  # userId, teamId, or orgId
    provider: str,
    credential_type: str,
    display_name: str,
    payload: dict,
    user_id: str,
    team_id: str | None = None,
    expires_at=None,
    metadata: dict | None = None,
) -> dict:
    """Create a new scoped credential.

    For TEAM-owned credentials, pass ``owner_type="TEAM"``, ``owner_id=<teamId>``
    and ``team_id=<teamId>`` so the dedicated ``teamId`` FK is populated (this
    gives ``onDelete: Cascade`` cleanup when the team is deleted). The read
    path (:func:`get_scoped_credentials` / :func:`get_credential_by_id`)
    resolves TEAM rows by ``ownerType="TEAM"`` + ``ownerId=<teamId>`` +
    ``organizationId``, so those three fields are the load-bearing shape.
    """
    if owner_type == "TEAM" and team_id != owner_id:
        # Enforce the invariant the docstring promises: without the matching
        # teamId FK, the row loses cascade cleanup and diverges from what the
        # read path resolves on.
        raise ValueError("team_id must equal owner_id for TEAM-owned credentials")

    # Generate the id up front so the encrypted payload's id matches the row's
    # primary key. Otherwise a decrypted read (via CREDENTIALS_ADAPTER) would
    # surface the client-supplied id from the blob instead of the authoritative
    # row id, breaking id-based resolution.
    credential_id = str(uuid4())
    encrypted = _cryptor.encrypt({**payload, "id": credential_id})

    cred = await prisma.integrationcredential.create(
        data={
            "id": credential_id,
            "organizationId": organization_id,
            "ownerType": owner_type,
            "ownerId": owner_id,
            "teamId": team_id,
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


async def list_team_credentials(
    organization_id: str,
    team_id: str,
    provider: str | None = None,
) -> list[dict]:
    """List active TEAM-owned credentials for a team (metadata only, no secrets).

    Matches the TEAM branch of :func:`get_scoped_credentials`. Callers MUST
    have verified the requester's team membership before exposing the result.
    """
    where: dict = {
        "organizationId": organization_id,
        "ownerType": "TEAM",
        "ownerId": team_id,
        "status": "active",
    }
    if provider:
        where["provider"] = provider

    creds = await prisma.integrationcredential.find_many(where=where)
    return [_cred_to_metadata(c, scope="TEAM") for c in creds]


async def delete_team_credential(
    credential_id: str,
    team_id: str,
    organization_id: str,
) -> None:
    """Soft-delete a TEAM-owned credential (``status`` -> ``'revoked'``).

    Scoped to a single team: the credential must be TEAM-owned by exactly
    ``team_id`` within ``organization_id``. This prevents a team admin from
    revoking a *different* team's credential by ID (cross-team escalation
    inside a shared org). The caller MUST have verified team-admin
    authorization for ``team_id`` first.

    Raises:
        ValueError: If no matching TEAM credential exists for this team/org.
    """
    cred = await prisma.integrationcredential.find_unique(where={"id": credential_id})
    if (
        cred is None
        or cred.organizationId != organization_id
        or cred.ownerType != "TEAM"
        or cred.ownerId != team_id
    ):
        raise ValueError(f"Credential {credential_id} not found")

    await prisma.integrationcredential.update(
        where={"id": credential_id},
        data={"status": "revoked"},
    )


async def delete_credential(
    credential_id: str,
    user_id: str,
    organization_id: str,
    is_org_admin: bool = False,
) -> None:
    """Soft-delete a credential by setting status to 'revoked'.

    Only the creator may revoke, unless ``is_org_admin`` is set — which
    callers must derive from a verified RequestContext (owner/admin),
    never from request input.
    """
    cred = await prisma.integrationcredential.find_unique(where={"id": credential_id})
    if cred is None or cred.organizationId != organization_id:
        raise ValueError(f"Credential {credential_id} not found")

    if not is_org_admin and cred.createdByUserId != user_id:
        raise ValueError(f"Credential {credential_id} not found")

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
