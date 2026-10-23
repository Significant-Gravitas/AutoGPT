"""Workload identities, and the short-lived tokens minted for them.

A workload identity lets a workload outside AutoGPT (a CI job, a deployed app)
act as its owner on the External API by exchanging an OIDC token from its own
platform (see ``workload_exchange``), instead of holding a long-lived API key.
"""

import hashlib
import logging
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

from prisma.enums import APIKeyPermission, WorkloadIdentityStatus
from prisma.models import WorkloadAccessToken as PrismaWorkloadAccessToken
from prisma.models import WorkloadIdentity as PrismaWorkloadIdentity
from pydantic import BaseModel, Field, model_validator

from backend.util.exceptions import NotFoundError
from backend.util.json import SafeJson

from .base import APIAuthorizationInfo
from .workload_trust import ClaimCondition, check_trust_policy

logger = logging.getLogger(__name__)

WORKLOAD_TOKEN_PREFIX = "agpt_wt_"
MIN_TOKEN_TTL_SECONDS = 300
MAX_TOKEN_TTL_SECONDS = 3600
# How stale an identity's recorded last use may get; see api_key.py.
LAST_USED_RESOLUTION = timedelta(minutes=5)


class InvalidWorkloadTokenError(Exception):
    """The bearer token is not a current workload token."""


class _WorkloadIdentityFields(BaseModel):
    """What a workload identity trusts and what it may do."""

    name: str = Field(min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    issuer: str = Field(
        min_length=1,
        max_length=512,
        description="Exact `iss` of the workload's OIDC tokens",
    )
    audience: str = Field(
        min_length=1,
        max_length=512,
        description="Exact `aud` the workload requests its tokens for",
    )
    claim_conditions: list[ClaimCondition] = Field(min_length=1, max_length=20)
    permissions: list[APIKeyPermission] = Field(min_length=1)
    token_ttl_seconds: int = Field(
        900, ge=MIN_TOKEN_TTL_SECONDS, le=MAX_TOKEN_TTL_SECONDS
    )


class WorkloadIdentitySpec(_WorkloadIdentityFields):
    """A new identity: its trust settings must pin tokens to one workload."""

    @model_validator(mode="after")
    def _pins_one_workload(self) -> "WorkloadIdentitySpec":
        check_trust_policy(self.issuer, self.audience, self.claim_conditions)
        return self


class WorkloadIdentityInfo(_WorkloadIdentityFields):
    # Not a WorkloadIdentitySpec: a stored identity was checked when created,
    # and re-checking on read would break it if the policy rules tighten.
    id: str
    status: WorkloadIdentityStatus
    created_at: datetime
    last_used_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None
    user_id: str
    organization_id: Optional[str] = None

    @staticmethod
    def from_db(row: PrismaWorkloadIdentity) -> "WorkloadIdentityInfo":
        return WorkloadIdentityInfo(
            id=row.id,
            name=row.name,
            description=row.description,
            issuer=row.issuer,
            audience=row.audience,
            claim_conditions=[
                ClaimCondition.model_validate(c) for c in row.claimConditions
            ],
            permissions=[APIKeyPermission(p) for p in row.permissions],
            token_ttl_seconds=row.tokenTtlSeconds,
            status=WorkloadIdentityStatus(row.status),
            created_at=row.createdAt,
            last_used_at=row.lastUsedAt,
            revoked_at=row.revokedAt,
            user_id=row.userId,
            organization_id=row.organizationId,
        )


class WorkloadTokenInfo(APIAuthorizationInfo):
    id: str
    workload_identity_id: str
    subject: str


async def create_workload_identity(
    user_id: str, spec: WorkloadIdentitySpec, organization_id: str | None = None
) -> WorkloadIdentityInfo:
    row = await PrismaWorkloadIdentity.prisma().create(
        data={
            "id": str(uuid.uuid4()),
            "name": spec.name,
            "description": spec.description,
            "issuer": spec.issuer,
            "audience": spec.audience,
            "claimConditions": SafeJson(
                [c.model_dump() for c in spec.claim_conditions]
            ),
            "permissions": list(spec.permissions),
            "tokenTtlSeconds": spec.token_ttl_seconds,
            "userId": user_id,
            "organizationId": organization_id,
        }
    )
    return WorkloadIdentityInfo.from_db(row)


async def list_workload_identities(
    user_id: str, organization_id: str | None = None
) -> list[WorkloadIdentityInfo]:
    rows = await PrismaWorkloadIdentity.prisma().find_many(
        where=_owned_by(user_id, organization_id),
        order={"createdAt": "desc"},
        take=100,
    )
    return [WorkloadIdentityInfo.from_db(row) for row in rows]


async def get_workload_identity(identity_id: str) -> WorkloadIdentityInfo | None:
    """Any identity by ID, for the token exchange; never expose to users."""
    row = await PrismaWorkloadIdentity.prisma().find_unique(where={"id": identity_id})
    return WorkloadIdentityInfo.from_db(row) if row else None


async def get_user_workload_identity(
    identity_id: str, user_id: str, organization_id: str | None = None
) -> WorkloadIdentityInfo:
    row = await PrismaWorkloadIdentity.prisma().find_first(
        where={"id": identity_id, **_owned_by(user_id, organization_id)}
    )
    if not row:
        raise NotFoundError(f"Workload identity #{identity_id} not found")
    return WorkloadIdentityInfo.from_db(row)


async def set_workload_identity_status(
    identity_id: str,
    user_id: str,
    status: Literal[WorkloadIdentityStatus.ACTIVE, WorkloadIdentityStatus.DISABLED],
    organization_id: str | None = None,
) -> WorkloadIdentityInfo:
    """Pause or resume an identity. A revoked identity stays revoked."""
    identity = await get_user_workload_identity(identity_id, user_id, organization_id)
    if identity.status == WorkloadIdentityStatus.REVOKED:
        raise ValueError(f"Workload identity #{identity_id} has been revoked")
    row = await PrismaWorkloadIdentity.prisma().update(
        where={"id": identity.id}, data={"status": status}
    )
    if not row:
        raise NotFoundError(f"Workload identity #{identity_id} not found")
    return WorkloadIdentityInfo.from_db(row)


async def revoke_workload_identity(
    identity_id: str, user_id: str, organization_id: str | None = None
) -> WorkloadIdentityInfo:
    """Revoke an identity for good, along with every token minted for it."""
    identity = await get_user_workload_identity(identity_id, user_id, organization_id)
    now = datetime.now(timezone.utc)
    row = await PrismaWorkloadIdentity.prisma().update(
        where={"id": identity.id},
        data={"status": WorkloadIdentityStatus.REVOKED, "revokedAt": now},
    )
    if not row:
        raise NotFoundError(f"Workload identity #{identity_id} not found")
    await PrismaWorkloadAccessToken.prisma().update_many(
        where={"workloadIdentityId": identity.id, "revokedAt": None},
        data={"revokedAt": now},
    )
    return WorkloadIdentityInfo.from_db(row)


async def mint_workload_token(
    identity: WorkloadIdentityInfo,
    *,
    subject: str,
    subject_jti: str | None,
    scopes: list[APIKeyPermission],
) -> tuple[WorkloadTokenInfo, str]:
    """Store a new token for the identity; returns it with its plaintext."""
    plaintext = WORKLOAD_TOKEN_PREFIX + secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(
        seconds=identity.token_ttl_seconds
    )
    row = await PrismaWorkloadAccessToken.prisma().create(
        data={
            "id": str(uuid.uuid4()),
            "token": _hash_token(plaintext),
            "expiresAt": expires_at,
            "workloadIdentityId": identity.id,
            "scopes": list(scopes),
            "subject": subject[:512],
            "subjectJti": subject_jti[:256] if subject_jti else None,
        }
    )
    return _token_info(row, identity), plaintext


async def validate_workload_token(token: str) -> WorkloadTokenInfo:
    """Raises InvalidWorkloadTokenError unless the token is current and its
    identity is still active."""
    row = await PrismaWorkloadAccessToken.prisma().find_unique(
        where={"token": _hash_token(token)}, include={"WorkloadIdentity": True}
    )
    if not row or not row.WorkloadIdentity:
        raise InvalidWorkloadTokenError("Workload token not found")
    if row.revokedAt is not None:
        raise InvalidWorkloadTokenError("Workload token has been revoked")
    if row.expiresAt < datetime.now(timezone.utc):
        raise InvalidWorkloadTokenError("Workload token expired")
    identity = WorkloadIdentityInfo.from_db(row.WorkloadIdentity)
    if identity.status != WorkloadIdentityStatus.ACTIVE:
        raise InvalidWorkloadTokenError("Workload identity is not active")
    return _token_info(row, identity)


async def record_workload_identity_use(identity: WorkloadIdentityInfo) -> None:
    """Stamp the identity's last use; a failed write is logged, not raised."""
    now = datetime.now(timezone.utc)
    if identity.last_used_at and now - identity.last_used_at < LAST_USED_RESOLUTION:
        return
    try:
        await PrismaWorkloadIdentity.prisma().update(
            where={"id": identity.id}, data={"lastUsedAt": now}
        )
    except Exception as e:
        logger.warning(f"Failed to record use of workload identity #{identity.id}: {e}")


async def cleanup_expired_workload_tokens() -> int:
    return await PrismaWorkloadAccessToken.prisma().delete_many(
        where={"expiresAt": {"lt": datetime.now(timezone.utc)}}
    )


def _owned_by(user_id: str, organization_id: str | None) -> dict:
    if organization_id is None:
        return {"userId": user_id}
    return {
        "userId": user_id,
        "OR": [{"organizationId": organization_id}, {"organizationId": None}],
    }


def _token_info(
    row: PrismaWorkloadAccessToken, identity: WorkloadIdentityInfo
) -> WorkloadTokenInfo:
    return WorkloadTokenInfo(
        id=row.id,
        type="workload",
        user_id=identity.user_id,
        scopes=[APIKeyPermission(s) for s in row.scopes],
        created_at=row.createdAt,
        expires_at=row.expiresAt,
        revoked_at=row.revokedAt,
        organization_id=identity.organization_id,
        workload_identity_id=identity.id,
        subject=row.subject,
    )


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()
