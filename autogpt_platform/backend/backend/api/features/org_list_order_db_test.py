"""The order of the org list (GET /api/orgs), against Postgres: the user's
personal org first, then their other orgs in the order they joined. The
frontend starts a tab in the first personal org listed, so that has to be the
org a request without an org header defaults to.

Lives outside ``orgs/``, whose tests mock Prisma; these need only Postgres, so
they connect Prisma themselves.
"""

import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from pydantic import BaseModel

from backend.api.features.orgs.db import _find_personal_org_member, list_user_orgs
from backend.data.db import prisma

# Each org the user is in, as (name, is personal, joined), in the order the
# rows are written: neither that order nor the join order puts the personal
# orgs first.
USER_ORGS = [
    ("team-b", False, datetime(2026, 3, 1, tzinfo=timezone.utc)),
    ("personal-new", True, datetime(2026, 4, 1, tzinfo=timezone.utc)),
    ("team-a", False, datetime(2026, 1, 1, tzinfo=timezone.utc)),
    ("personal-old", True, datetime(2026, 2, 1, tzinfo=timezone.utc)),
]


class Memberships(BaseModel):
    user: str
    orgs: dict[str, str]


@pytest_asyncio.fixture(loop_scope="session")
async def memberships() -> AsyncIterator[Memberships]:
    connected_here = not prisma.is_connected()
    if connected_here:
        await prisma.connect()
    suffix = uuid.uuid4().hex[:8]
    user, other = f"org-order-{suffix}", f"org-order-other-{suffix}"
    org_ids: dict[str, str] = {}
    try:
        for uid in (user, other):
            await prisma.user.create(data={"id": uid, "email": f"{uid}@example.com"})
        for name, is_personal, joined in USER_ORGS:
            org_ids[name] = await _join(user, name, suffix, is_personal, joined)
        # Someone else's personal org, joined before any of the user's.
        org_ids["other-personal"] = await _join(
            other,
            "other-personal",
            suffix,
            True,
            datetime(2025, 12, 1, tzinfo=timezone.utc),
        )
        yield Memberships(user=user, orgs=org_ids)
    finally:
        try:
            if prisma.is_connected():
                await prisma.user.delete_many(where={"id": {"in": [user, other]}})
                await prisma.organization.delete_many(
                    where={"id": {"in": list(org_ids.values())}}
                )
        finally:
            if connected_here and prisma.is_connected():
                await prisma.disconnect()


async def _join(
    user_id: str, name: str, suffix: str, is_personal: bool, joined: datetime
) -> str:
    org = await prisma.organization.create(
        data={"name": name, "slug": f"{name}-{suffix}", "isPersonal": is_personal}
    )
    await prisma.orgmember.create(
        data={
            "orgId": org.id,
            "userId": user_id,
            "status": "ACTIVE",
            "isOwner": True,
            "createdAt": joined,
            "joinedAt": joined,
        }
    )
    return org.id


@pytest.mark.asyncio(loop_scope="session")
async def test_personal_orgs_come_first_then_the_order_the_user_joined(
    memberships: Memberships,
):
    orgs = await list_user_orgs(memberships.user)

    assert [org.id for org in orgs] == [
        memberships.orgs[name]
        for name in ("personal-old", "personal-new", "team-a", "team-b")
    ]


@pytest.mark.asyncio(loop_scope="session")
async def test_the_first_personal_org_listed_is_the_one_requests_default_to(
    memberships: Memberships,
):
    orgs = await list_user_orgs(memberships.user)
    default = await _find_personal_org_member(memberships.user)

    assert default is not None
    assert orgs[0].id == default.orgId == memberships.orgs["personal-old"]
