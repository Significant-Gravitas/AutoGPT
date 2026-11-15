"""Ownership and membership changes, on a real database.

Outside `orgs/`, whose conftest replaces the `server` fixture with a no-op.
"""

import asyncio
import uuid

import fastapi
import httpx
import prisma.actions
import prisma.models
import pytest
from autogpt_libs.auth import get_request_context
from autogpt_libs.auth.models import RequestContext

from backend.api.features.orgs import db as orgs_db
from backend.api.features.orgs.routes import router as orgs_router
from backend.api.rest_api import app as rest_app
from backend.data.user import get_or_create_user
from backend.util.exceptions import NotFoundError
from backend.util.test import SpinTestServer


@pytest.mark.asyncio(loop_scope="session")
async def test_two_concurrent_transfers_by_one_owner_leave_one_owner(
    server: SpinTestServer, monkeypatch: pytest.MonkeyPatch
):
    owner, first, second = await _user(), await _user(), await _user()
    org = await _create_org(owner)
    for member in (first, second):
        await orgs_db.add_org_member(org.id, member)
    _meet_the_other_call_at_the_owner_row(monkeypatch, owner)

    results = await asyncio.gather(
        orgs_db.transfer_ownership(org.id, owner, first),
        orgs_db.transfer_ownership(org.id, owner, second),
        return_exceptions=True,
    )

    assert sorted(type(r).__name__ for r in results) == ["NoneType", "ValueError"]
    assert await _owners(org.id) in ([first], [second])


@pytest.mark.asyncio(loop_scope="session")
async def test_a_transfer_to_a_non_member_keeps_the_owner(server: SpinTestServer):
    owner, stranger = await _user(), await _user()
    org = await _create_org(owner)

    with pytest.raises(NotFoundError):
        await orgs_db.transfer_ownership(org.id, owner, stranger)

    assert await _owners(org.id) == [owner]


@pytest.mark.asyncio(loop_scope="session")
async def test_a_member_who_belongs_to_no_other_org_is_removed(
    server: SpinTestServer,
):
    owner = await _user()
    org = await _create_org(owner)
    member = await _user_without_an_org()
    await orgs_db.add_org_member(org.id, member)

    response = await _as_owner(owner, org.id).delete(f"/orgs/{org.id}/members/{member}")

    assert response.status_code == 204, response.text
    assert not await prisma.models.OrgMember.prisma().find_many(
        where={"orgId": org.id, "userId": member}
    )
    assert not await prisma.models.TeamMember.prisma().find_many(
        where={"userId": member, "Team": {"orgId": org.id}}
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_a_removal_that_fails_part_way_keeps_the_membership(
    server: SpinTestServer, monkeypatch: pytest.MonkeyPatch
):
    owner, member = await _user(), await _user()
    org = await _create_org(owner)
    await orgs_db.add_org_member(org.id, member)

    async def fail(self, *args, **kwargs):
        raise RuntimeError("team memberships could not be deleted")

    monkeypatch.setattr(prisma.actions.TeamMemberActions, "delete_many", fail)

    with pytest.raises(RuntimeError):
        await orgs_db.remove_org_member(org.id, member, requesting_user_id=owner)

    assert await prisma.models.OrgMember.prisma().find_unique(
        where={"orgId_userId": {"orgId": org.id, "userId": member}}
    )


def _meet_the_other_call_at_the_owner_row(
    monkeypatch: pytest.MonkeyPatch, owner_id: str
) -> None:
    """Each call waits for the other once it has read or written the owner's row,
    so both get past the owner check together unless the database serializes
    them; then the first one's wait times out, which is the passing outcome."""
    barrier = asyncio.Barrier(2)
    actions = prisma.actions.OrgMemberActions

    for name in ("find_unique", "update_many"):
        real = getattr(actions, name)

        async def call(self, *args, _real=real, **kwargs):
            result = await _real(self, *args, **kwargs)
            where = kwargs.get("where") or {}
            if owner_id in (
                where.get("userId"),
                where.get("orgId_userId", {}).get("userId"),
            ):
                try:
                    await asyncio.wait_for(barrier.wait(), timeout=2)
                except (TimeoutError, asyncio.BrokenBarrierError):
                    pass
            return result

        monkeypatch.setattr(actions, name, call)


def _as_owner(owner_id: str, org_id: str) -> httpx.AsyncClient:
    app = fastapi.FastAPI(exception_handlers=rest_app.exception_handlers)
    app.include_router(orgs_router, prefix="/orgs")
    app.dependency_overrides[get_request_context] = lambda: RequestContext(
        user_id=owner_id,
        org_id=org_id,
        team_id=None,
        is_org_owner=True,
        is_org_admin=True,
        is_org_billing_manager=False,
        is_team_admin=False,
        is_team_billing_manager=False,
        seat_status="ACTIVE",
    )
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    return httpx.AsyncClient(transport=transport, base_url="http://orgs")


async def _user() -> str:
    user = await get_or_create_user(
        {
            "sub": str(uuid.uuid4()),
            "email": f"org-membership-{uuid.uuid4().hex[:8]}@example.com",
            "name": "Org Membership",
        }
    )
    return user.id


async def _user_without_an_org() -> str:
    """The row the sign-up hook writes, before anything creates a personal org."""
    user = await prisma.models.User.prisma().create(
        data={
            "id": str(uuid.uuid4()),
            "email": f"org-membership-{uuid.uuid4().hex[:8]}@example.com",
        }
    )
    return user.id


async def _owners(org_id: str) -> list[str]:
    owners = await prisma.models.OrgMember.prisma().find_many(
        where={"orgId": org_id, "isOwner": True}
    )
    return [m.userId for m in owners]


async def _create_org(user_id: str):
    slug = f"org-membership-{uuid.uuid4().hex[:12]}"
    return await orgs_db.create_org("Org Membership", slug, user_id)
