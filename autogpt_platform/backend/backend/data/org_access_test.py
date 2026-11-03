"""Who may use organizations, against the real database.

An admin's grant turns them on for one account. Its team organizations pass
them on to their active members and to the people they have invited; an
archived organization, a suspended member and a spent invitation pass nothing.
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from prisma.models import AuditLog, Organization, OrgInvitation, OrgMember

from backend.api.features.orgs.db import create_org
from backend.data import org_access
from backend.data.user import get_or_create_user
from backend.util.test import SpinTestServer

# These read the real grants (see `no_org_access_grants` in the conftest).
pytestmark = pytest.mark.real_org_access


async def _user(label: str) -> tuple[str, str]:
    suffix = uuid.uuid4().hex[:8]
    email = f"org-access-{label}-{suffix}@example.com"
    user = await get_or_create_user(
        {"sub": str(uuid.uuid4()), "email": email, "name": label}
    )
    return user.id, email


async def _team_org(owner_id: str) -> str:
    suffix = uuid.uuid4().hex[:8]
    return (await create_org(f"Team {suffix}", f"team-{suffix}", owner_id)).id


async def _access(user_id: str) -> bool:
    """A fresh answer, for accounts whose access changed through someone else."""
    org_access.has_org_access.cache_delete(user_id)
    return await org_access.has_org_access(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_grant_turns_organizations_on_and_a_revoke_off(
    server: SpinTestServer,
):
    user_id, _ = await _user("owner")
    admin_id, _ = await _user("admin")
    assert not await _access(user_id)

    # The cached "no" is dropped by the grant, and the cached "yes" by the
    # revoke, so the account's own answer changes at once on every process.
    status = await org_access.grant_org_access(user_id, admin_id, " Team trial ")
    assert status.has_access
    assert status.grant is not None and status.grant.note == "Team trial"
    assert await org_access.has_org_access(user_id)

    status = await org_access.revoke_org_access(user_id, admin_id)
    assert not status.has_access
    assert not await org_access.has_org_access(user_id)

    audit = await AuditLog.prisma().find_many(
        where={"entityType": org_access.AUDIT_ENTITY, "entityId": user_id},
        order={"createdAt": "asc"},
    )
    assert [(row.action, row.actorUserId) for row in audit] == [
        ("org_access.grant", admin_id),
        ("org_access.revoke", admin_id),
    ]


@pytest.mark.asyncio(loop_scope="session")
async def test_granting_again_keeps_the_first_grant(server: SpinTestServer):
    user_id, _ = await _user("owner")
    first_admin, _ = await _user("admin-a")
    second_admin, _ = await _user("admin-b")

    await org_access.grant_org_access(user_id, first_admin, "First")
    status = await org_access.grant_org_access(user_id, second_admin, "Second")

    assert status.grant is not None
    assert (status.grant.note, status.grant.granted_by_user_id) == (
        "First",
        first_admin,
    )
    assert (
        await AuditLog.prisma().count(
            where={"entityType": org_access.AUDIT_ENTITY, "entityId": user_id}
        )
        == 1
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_a_team_organization_passes_access_to_its_active_members(
    server: SpinTestServer,
):
    owner_id, _ = await _user("owner")
    member_id, _ = await _user("member")
    admin_id, _ = await _user("admin")
    org_id = await _team_org(owner_id)
    member = await OrgMember.prisma().create(
        data={"orgId": org_id, "userId": member_id, "status": "ACTIVE"}
    )
    assert not await _access(member_id)

    await org_access.grant_org_access(owner_id, admin_id, None)
    assert await _access(member_id)
    status = await org_access.get_org_access_status(member_id)
    assert status.grant is None
    assert [(i.org_id, i.via) for i in status.inherited_from] == [
        (org_id, "membership")
    ]

    await OrgMember.prisma().update(
        where={"id": member.id}, data={"status": "SUSPENDED"}
    )
    assert not await _access(member_id)

    await OrgMember.prisma().update(where={"id": member.id}, data={"status": "ACTIVE"})
    await Organization.prisma().update(
        where={"id": org_id}, data={"archivedAt": datetime.now(timezone.utc)}
    )
    assert not await _access(member_id)

    await Organization.prisma().update(where={"id": org_id}, data={"archivedAt": None})
    await org_access.revoke_org_access(owner_id, admin_id)
    assert not await _access(member_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_pending_invitation_passes_access_and_a_spent_one_does_not(
    server: SpinTestServer,
):
    owner_id, _ = await _user("owner")
    invitee_id, invitee_email = await _user("invitee")
    admin_id, _ = await _user("admin")
    org_id = await _team_org(owner_id)
    await org_access.grant_org_access(owner_id, admin_id, None)
    now = datetime.now(timezone.utc)
    pending = {
        "expiresAt": now + timedelta(days=7),
        "revokedAt": None,
        "acceptedAt": None,
    }
    invitation = await OrgInvitation.prisma().create(
        data={
            "orgId": org_id,
            # Addresses are matched without regard to case.
            "email": invitee_email.upper(),
            "invitedByUserId": owner_id,
            "teamIds": [],
            **pending,
        }
    )
    assert await _access(invitee_id)
    status = await org_access.get_org_access_status(invitee_id)
    assert [(i.org_id, i.via) for i in status.inherited_from] == [
        (org_id, "invitation")
    ]

    for spent in (
        {"expiresAt": now - timedelta(minutes=1)},
        {"revokedAt": now},
        {"acceptedAt": now},
    ):
        await OrgInvitation.prisma().update(
            where={"id": invitation.id}, data={**pending, **spent}
        )
        assert not await _access(invitee_id), spent


@pytest.mark.asyncio(loop_scope="session")
async def test_an_owner_without_a_grant_passes_nothing_on(server: SpinTestServer):
    owner_id, _ = await _user("owner")
    member_id, _ = await _user("member")
    org_id = await _team_org(owner_id)
    await OrgMember.prisma().create(
        data={"orgId": org_id, "userId": member_id, "status": "ACTIVE"}
    )

    assert not await _access(owner_id)
    assert not await _access(member_id)
