"""Which accounts may use organizations.

An admin grants organizations to an account. That account's team organizations
pass the access on to their active members and to the people they have invited,
so a customer's whole team can join without an admin enabling each person.
A grant or revoke reaches the account itself at once and its organizations'
members and invitees within ``ACCESS_CACHE_SECONDS``; it never removes
memberships, invitations or data.
"""

import logging
from datetime import datetime
from typing import Literal

import prisma.types
from prisma import Prisma
from prisma.models import AuditLog, OrgAccessGrant, User
from pydantic import BaseModel, computed_field

from backend.data.db import query_raw_with_schema, transaction
from backend.data.db_accessors import org_access_db
from backend.util.cache import cached
from backend.util.exceptions import NotFoundError
from backend.util.json import SafeJson

logger = logging.getLogger(__name__)

# How long an account's answer is shared across processes before the database
# is asked again. A grant or revoke drops the account's own answer at once.
ACCESS_CACHE_SECONDS = 60
AUDIT_ENTITY = "OrgAccessGrant"


class OrgAccessGrantInfo(BaseModel):
    user_id: str
    email: str
    name: str | None
    granted_at: datetime
    granted_by_user_id: str
    granted_by_email: str | None
    note: str | None


class InheritedOrgAccess(BaseModel):
    """A team organization that gives an account access without a grant of its own."""

    org_id: str
    org_name: str
    via: Literal["membership", "invitation"]


class OrgAccessStatus(BaseModel):
    user_id: str
    email: str
    grant: OrgAccessGrantInfo | None
    inherited_from: list[InheritedOrgAccess]

    @computed_field
    @property
    def has_access(self) -> bool:
        return self.grant is not None or bool(self.inherited_from)


class OrgAccessGrantPage(BaseModel):
    grants: list[OrgAccessGrantInfo]
    total: int
    page: int
    page_size: int


@cached(maxsize=4096, ttl_seconds=ACCESS_CACHE_SECONDS, shared_cache=True)
async def has_org_access(user_id: str) -> bool:
    """Whether *user_id* may use organizations through a grant: its own, or one
    held by an owner of a team organization it belongs to or is invited to.

    Works in any process; the shared cache is read before the DatabaseManager is.
    """
    return await org_access_db().query_org_access(user_id)


async def query_org_access(user_id: str) -> bool:
    """The uncached read behind :func:`has_org_access`."""
    rows = await query_raw_with_schema(_HAS_ACCESS_QUERY, user_id)
    return bool(rows and rows[0]["allowed"])


async def get_org_access_status(user_id: str) -> OrgAccessStatus:
    user = await User.prisma().find_unique(where={"id": user_id})
    if user is None:
        raise NotFoundError(f"User #{user_id} not found")
    grant = await OrgAccessGrant.prisma().find_unique(where={"userId": user_id})
    inherited = await query_raw_with_schema(_INHERITED_QUERY, user_id)
    grants = await _grant_infos([(grant, user)] if grant else [])
    return OrgAccessStatus(
        user_id=user.id,
        email=user.email,
        grant=grants[0] if grants else None,
        inherited_from=[InheritedOrgAccess(**row) for row in inherited],
    )


async def list_org_access_grants(
    search: str | None, page: int, page_size: int
) -> OrgAccessGrantPage:
    term = (search or "").strip()
    where: prisma.types.OrgAccessGrantWhereInput = {}
    if term:
        where["User"] = {
            "is": {
                "OR": [
                    {"email": {"contains": term, "mode": "insensitive"}},
                    {"name": {"contains": term, "mode": "insensitive"}},
                ]
            }
        }
    total = await OrgAccessGrant.prisma().count(where=where)
    rows = await OrgAccessGrant.prisma().find_many(
        where=where,
        include={"User": True},
        order=[{"createdAt": "desc"}, {"userId": "asc"}],
        skip=(page - 1) * page_size,
        take=page_size,
    )
    grants = await _grant_infos([(row, row.User) for row in rows if row.User])
    return OrgAccessGrantPage(
        grants=grants, total=total, page=page, page_size=page_size
    )


async def grant_org_access(
    user_id: str, admin_user_id: str, note: str | None
) -> OrgAccessStatus:
    """Give *user_id* organizations. An account that already has a grant keeps
    the original one and its note."""
    if await User.prisma().find_unique(where={"id": user_id}) is None:
        raise NotFoundError(f"User #{user_id} not found")
    clean_note = (note or "").strip() or None
    async with transaction() as tx:
        existing = await OrgAccessGrant.prisma(tx).find_unique(
            where={"userId": user_id}
        )
        if existing is None:
            await OrgAccessGrant.prisma(tx).create(
                data={
                    "userId": user_id,
                    "grantedByUserId": admin_user_id,
                    "note": clean_note,
                }
            )
            await _audit(tx, admin_user_id, user_id, "grant", clean_note)
    has_org_access.cache_delete(user_id)
    logger.info(f"Admin {admin_user_id} enabled organizations for user {user_id}")
    return await get_org_access_status(user_id)


async def revoke_org_access(user_id: str, admin_user_id: str) -> OrgAccessStatus:
    """Take back *user_id*'s grant. Its team organizations stop passing access
    on; memberships and data stay as they are."""
    async with transaction() as tx:
        deleted = await OrgAccessGrant.prisma(tx).delete_many(where={"userId": user_id})
        if deleted:
            await _audit(tx, admin_user_id, user_id, "revoke", None)
    has_org_access.cache_delete(user_id)
    logger.info(f"Admin {admin_user_id} disabled organizations for user {user_id}")
    return await get_org_access_status(user_id)


async def find_user_id_by_email(email: str) -> str | None:
    user = await User.prisma().find_first(
        where={"email": {"equals": email.strip(), "mode": "insensitive"}}
    )
    return user.id if user else None


async def _grant_infos(
    pairs: list[tuple[OrgAccessGrant, User]],
) -> list[OrgAccessGrantInfo]:
    admin_ids = list({grant.grantedByUserId for grant, _ in pairs})
    admins = (
        await User.prisma().find_many(where={"id": {"in": admin_ids}})
        if admin_ids
        else []
    )
    admin_emails = {admin.id: admin.email for admin in admins}
    return [
        OrgAccessGrantInfo(
            user_id=user.id,
            email=user.email,
            name=user.name,
            granted_at=grant.createdAt,
            granted_by_user_id=grant.grantedByUserId,
            granted_by_email=admin_emails.get(grant.grantedByUserId),
            note=grant.note,
        )
        for grant, user in pairs
    ]


async def _audit(
    tx: Prisma, admin_user_id: str, user_id: str, action: str, note: str | None
) -> None:
    data: prisma.types.AuditLogCreateInput = {
        "actorUserId": admin_user_id,
        "entityType": AUDIT_ENTITY,
        "entityId": user_id,
        "action": f"org_access.{action}",
    }
    if note:
        data["afterJson"] = SafeJson({"note": note})
    await AuditLog.prisma(tx).create(data=data)


# A team organization passes access on when one of its active owners holds a
# grant. Personal, archived and deleted organizations never do.
_ENTITLED_ORG = """
    o."isPersonal" = false
    AND o."archivedAt" IS NULL
    AND o."deletedAt" IS NULL
    AND EXISTS (
        SELECT 1
        FROM {schema_prefix}"OrgMember" owner
        JOIN {schema_prefix}"OrgAccessGrant" og ON og."userId" = owner."userId"
        WHERE owner."orgId" = o."id"
          AND owner."isOwner" = true
          AND owner."status" = 'ACTIVE'
    )
"""

_MEMBER_ORGS = (
    """
    SELECT o."id" AS org_id, o."name" AS org_name, 'membership' AS via
    FROM {schema_prefix}"OrgMember" m
    JOIN {schema_prefix}"Organization" o ON o."id" = m."orgId"
    WHERE m."userId" = $1 AND m."status" = 'ACTIVE' AND"""
    + _ENTITLED_ORG
)

_INVITING_ORGS = (
    """
    SELECT o."id" AS org_id, o."name" AS org_name, 'invitation' AS via
    FROM {schema_prefix}"OrgInvitation" i
    JOIN {schema_prefix}"Organization" o ON o."id" = i."orgId"
    WHERE (
        i."targetUserId" = $1
        OR lower(i."email") = (
            SELECT lower(u."email") FROM {schema_prefix}"User" u WHERE u."id" = $1
        )
    )
      AND i."acceptedAt" IS NULL
      AND i."revokedAt" IS NULL
      AND i."expiresAt" > now()
      AND"""
    + _ENTITLED_ORG
)

_HAS_ACCESS_QUERY = (
    """
    SELECT (
        EXISTS (
            SELECT 1 FROM {schema_prefix}"OrgAccessGrant" g WHERE g."userId" = $1
        )
        OR EXISTS ("""
    + _MEMBER_ORGS
    + """)
        OR EXISTS ("""
    + _INVITING_ORGS
    + """)
    ) AS allowed
"""
)

_INHERITED_QUERY = (
    """
    SELECT DISTINCT ON (org_id) org_id, org_name, via
    FROM ("""
    + _MEMBER_ORGS
    + " UNION ALL "
    + _INVITING_ORGS
    + """) inherited
    ORDER BY org_id, via DESC
"""
)
