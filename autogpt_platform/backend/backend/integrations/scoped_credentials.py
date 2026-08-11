"""Scoped credential store using the IntegrationCredential table.

Provides the new credential resolution path (USER → TEAM → ORG)
using the IntegrationCredential table introduced in PR1. During the
dual-read transition period, callers should try this store first and
fall back to the legacy IntegrationCredentialsStore.

This store is used alongside the existing credentials_store.py which
reads from the User.integrations encrypted blob.
"""

import logging
from datetime import datetime
from enum import StrEnum
from typing import Any, NotRequired, Optional, TypedDict
from uuid import uuid4

from prisma.enums import CredentialOwnerType
from prisma.types import IntegrationCredentialCreateInput

from backend.data.db import prisma
from backend.data.model import CredentialsType
from backend.util.encryption import JSONCryptor
from backend.util.json import SafeJson

logger = logging.getLogger(__name__)


class CredentialStatus(StrEnum):
    """Lifecycle status of an ``IntegrationCredential`` row.

    Lowercase because the column is a plain ``String @default("active")``.
    Deliberately *not* the casing of ``TeamMember.status`` (``ACTIVE``), which
    is a real Prisma enum on an unrelated model — don't normalize them.
    """

    ACTIVE = "active"
    REVOKED = "revoked"


class CredentialMetadata(TypedDict):
    """Non-secret view of a credential row — the store↔caller boundary shape.

    The integrations router maps this straight onto ``CredentialsMetaResponse``,
    so the keys here are load-bearing API surface.
    """

    id: str
    provider: str
    credentialType: CredentialsType
    displayName: str
    scope: CredentialOwnerType
    createdByUserId: str
    lastUsedAt: Optional[datetime]
    expiresAt: Optional[datetime]
    createdAt: Optional[datetime]
    metadata: Optional[dict[str, Any]]
    payload: NotRequired[dict[str, Any]]


_cryptor: JSONCryptor | None = None


def _get_cryptor() -> JSONCryptor:
    """Lazy singleton — instantiating JSONCryptor at import time makes every
    importer of this module (the integrations router, and transitively the
    whole app) require ENCRYPTION_KEY, which breaks key-less contexts like the
    CI OpenAPI export."""
    global _cryptor
    if _cryptor is None:
        _cryptor = JSONCryptor()
    return _cryptor


async def get_scoped_credentials(
    user_id: str,
    organization_id: str,
    team_id: str | None = None,
    provider: str | None = None,
) -> list[CredentialMetadata]:
    """Get credentials visible to the user in the current org/team context.

    Resolution order (per plan 3D):
    1. USER credentials created by this user in this org
    2. TEAM credentials for the active team (if a team is set)
    3. ORG credentials for the active org

    Returns a list of credential metadata dicts (not decrypted payloads).
    """
    results: list[CredentialMetadata] = []

    # 1. User-scoped credentials
    user_where: dict = {
        "organizationId": organization_id,
        "ownerType": CredentialOwnerType.USER,
        "ownerId": user_id,
        "status": CredentialStatus.ACTIVE,
    }
    if provider:
        user_where["provider"] = provider

    user_creds = await prisma.integrationcredential.find_many(where=user_where)
    results.extend(
        _cred_to_metadata(c, scope=CredentialOwnerType.USER) for c in user_creds
    )

    # 2. Team-scoped credentials (only if a team context is active)
    if team_id:
        ws_where: dict = {
            "organizationId": organization_id,
            "ownerType": CredentialOwnerType.TEAM,
            "ownerId": team_id,
            "status": CredentialStatus.ACTIVE,
        }
        if provider:
            ws_where["provider"] = provider

        ws_creds = await prisma.integrationcredential.find_many(where=ws_where)
        results.extend(
            _cred_to_metadata(c, scope=CredentialOwnerType.TEAM) for c in ws_creds
        )

    # 3. Org-scoped credentials
    org_where: dict = {
        "organizationId": organization_id,
        "ownerType": CredentialOwnerType.ORG,
        "ownerId": organization_id,
        "status": CredentialStatus.ACTIVE,
    }
    if provider:
        org_where["provider"] = provider

    org_creds = await prisma.integrationcredential.find_many(where=org_where)
    results.extend(
        _cred_to_metadata(c, scope=CredentialOwnerType.ORG) for c in org_creds
    )

    return results


async def get_credential_by_id(
    credential_id: str,
    user_id: str,
    organization_id: str,
    team_id: str | None = None,
    decrypt: bool = False,
) -> Optional[CredentialMetadata]:
    """Get a specific credential by ID if the user has access.

    Access rules (enforced HERE, not trusted to callers):
    - Revoked (soft-deleted) rows are invisible, same as the list paths
    - USER creds: only the creating user can access
    - TEAM creds: only via the matching active team, or verified team
      membership when the caller's active context is a different team
    - ORG creds: any org member (``organization_id`` comes from a
      membership-verified RequestContext)
    """
    cred = await prisma.integrationcredential.find_unique(where={"id": credential_id})
    if cred is None or cred.organizationId != organization_id:
        return None

    # A revoked credential must stay revoked on every path. The list queries
    # already filter on `status`; without the same filter here a soft-deleted
    # credential would still be readable — and with decrypt=True, usable — by ID.
    if cred.status != CredentialStatus.ACTIVE:
        return None

    if cred.ownerType == CredentialOwnerType.USER and cred.createdByUserId != user_id:
        return None

    if cred.ownerType == CredentialOwnerType.TEAM and cred.ownerId != team_id:
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
        result["payload"] = _get_cryptor().decrypt(cred.encryptedPayload)

    return result


async def create_credential(
    organization_id: str,
    owner_type: CredentialOwnerType,
    owner_id: str,  # userId, teamId, or orgId
    provider: str,
    credential_type: str,
    display_name: str,
    payload: dict,
    user_id: str,
    expires_at: datetime | None = None,
    metadata: dict | None = None,
) -> CredentialMetadata:
    """Create a new scoped credential.

    For TEAM-owned credentials pass ``owner_type=CredentialOwnerType.TEAM`` and
    ``owner_id=<teamId>``; the dedicated ``teamId`` FK is derived from
    ``owner_id`` so the two can never desync (the FK is what gives
    ``onDelete: Cascade`` cleanup when the team is deleted). The read path
    (:func:`get_scoped_credentials` / :func:`get_credential_by_id`) resolves
    TEAM rows by ``ownerType="TEAM"`` + ``ownerId=<teamId>`` +
    ``organizationId``, so those three fields are the load-bearing shape.
    """
    # Generate the id up front so the encrypted payload's id matches the row's
    # primary key. Otherwise a decrypted read (via CREDENTIALS_ADAPTER) would
    # surface the client-supplied id from the blob instead of the authoritative
    # row id, breaking id-based resolution.
    credential_id = str(uuid4())
    encrypted = _get_cryptor().encrypt({**payload, "id": credential_id})

    # `Organization` is a *required* relation on IntegrationCredential, so the
    # query engine rejects a raw `organizationId` scalar with
    # MissingRequiredValueError — it has to be given in `connect` form.
    data: IntegrationCredentialCreateInput = {
        "id": credential_id,
        "Organization": {"connect": {"id": organization_id}},
        "ownerType": owner_type,
        "ownerId": owner_id,
        "provider": provider,
        "credentialType": credential_type,
        "displayName": display_name,
        "encryptedPayload": encrypted,
        "createdByUserId": user_id,
        "expiresAt": expires_at,
    }
    if owner_type == CredentialOwnerType.TEAM:
        # The teamId relation is named `Workspace` on this model (see its
        # @relation("TeamCredentials")), and it only ever points at the owning
        # team. Non-TEAM rows must omit the key entirely rather than pass None.
        data["Workspace"] = {"connect": {"id": owner_id}}
    if metadata is not None:
        # `metadata` is a `Json?` column: a raw dict is not a valid input value.
        data["metadata"] = SafeJson(metadata)

    cred = await prisma.integrationcredential.create(data=data)

    return _cred_to_metadata(cred, scope=owner_type)


async def list_team_credentials(
    organization_id: str,
    team_id: str,
    provider: str | None = None,
) -> list[CredentialMetadata]:
    """List active TEAM-owned credentials for a team (metadata only, no secrets).

    Matches the TEAM branch of :func:`get_scoped_credentials`. Callers MUST
    have verified the requester's team membership before exposing the result.
    """
    where: dict = {
        "organizationId": organization_id,
        "ownerType": CredentialOwnerType.TEAM,
        "ownerId": team_id,
        "status": CredentialStatus.ACTIVE,
    }
    if provider:
        where["provider"] = provider

    creds = await prisma.integrationcredential.find_many(where=where)
    return [_cred_to_metadata(c, scope=CredentialOwnerType.TEAM) for c in creds]


async def delete_team_credential(
    credential_id: str,
    team_id: str,
    organization_id: str,
) -> None:
    """Soft-delete a TEAM-owned credential (``status`` -> ``'revoked'``).

    Scoped to a single team: the credential must be an active TEAM-owned row
    belonging to exactly ``team_id`` within ``organization_id``. This prevents
    a team admin from revoking a *different* team's credential by ID
    (cross-team escalation inside a shared org). The caller MUST have verified
    team-admin authorization for ``team_id`` first.

    The scope is expressed as the ``where`` of a single ``update_many``, so the
    ownership check and the write are one atomic statement (and one round-trip)
    instead of a find-then-update that another writer can slip between.

    Raises:
        ValueError: If no matching active TEAM credential exists for this
            team/org.
    """
    revoked = await prisma.integrationcredential.update_many(
        where={
            "id": credential_id,
            "organizationId": organization_id,
            "ownerType": CredentialOwnerType.TEAM,
            "ownerId": team_id,
            "status": CredentialStatus.ACTIVE,
        },
        data={"status": CredentialStatus.REVOKED},
    )
    if revoked == 0:
        raise ValueError(f"Credential {credential_id} not found")


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
        data={"status": CredentialStatus.REVOKED},
    )


def _cred_to_metadata(cred, scope: CredentialOwnerType) -> CredentialMetadata:
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
        "metadata": cred.metadata,
    }
