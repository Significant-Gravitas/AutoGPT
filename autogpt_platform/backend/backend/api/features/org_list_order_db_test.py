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

import fastapi
import pytest
import pytest_asyncio
from autogpt_libs.auth.dependencies import get_request_context
from pydantic import BaseModel

from backend.api.features.orgs.db import list_user_orgs
from backend.data.db import prisma, transaction


class Join(BaseModel):
    """One membership: the org it is in, and how and when the user joined."""

    org: str
    joined: datetime
    personal: bool = False
    owner: bool = False
    status: str = "ACTIVE"


# The user's memberships in the order the rows are written: neither that order
# nor the join order puts the personal orgs first.
USER_JOINS = [
    Join(org="team-b", joined=datetime(2026, 3, 1, tzinfo=timezone.utc)),
    Join(
        org="personal-new",
        joined=datetime(2026, 4, 1, tzinfo=timezone.utc),
        personal=True,
        owner=True,
    ),
    Join(org="team-a", joined=datetime(2026, 1, 1, tzinfo=timezone.utc), owner=True),
    Join(
        org="personal-old",
        joined=datetime(2026, 2, 1, tzinfo=timezone.utc),
        personal=True,
        owner=True,
    ),
    # A team org the user was removed from: no longer one of theirs.
    Join(
        org="team-left",
        joined=datetime(2025, 11, 1, tzinfo=timezone.utc),
        status="REMOVED",
    ),
]

# Someone else's personal org, joined before any of the user's.
OTHER_JOIN = Join(
    org="other-personal",
    joined=datetime(2025, 12, 1, tzinfo=timezone.utc),
    personal=True,
    owner=True,
)


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
        for join in USER_JOINS:
            org_ids[join.org] = await _join(user, join, suffix)
        org_ids[OTHER_JOIN.org] = await _join(other, OTHER_JOIN, suffix)
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


async def _join(user_id: str, join: Join, suffix: str) -> str:
    """Write the org and the membership in one transaction, so a membership
    that fails leaves no org behind that the teardown can't see."""
    async with transaction() as tx:
        org = await tx.organization.create(
            data={
                "name": join.org,
                "slug": f"{join.org}-{suffix}",
                "isPersonal": join.personal,
            }
        )
        await tx.orgmember.create(
            data={
                "orgId": org.id,
                "userId": user_id,
                "status": join.status,
                "isOwner": join.owner,
                "createdAt": join.joined,
                "joinedAt": join.joined,
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
    # A request with no org header, resolved as every route resolves it.
    request = fastapi.Request(
        {"type": "http", "method": "GET", "path": "/api/orgs", "headers": []}
    )
    context = await get_request_context(request, {"sub": memberships.user})

    assert orgs[0].id == context.org_id == memberships.orgs["personal-old"]
