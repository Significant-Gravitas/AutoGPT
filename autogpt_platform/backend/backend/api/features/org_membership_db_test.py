"""Ownership and membership changes, on a real database.

Outside `orgs/`, whose conftest replaces the `server` fixture with a no-op.
"""

import asyncio
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable

import fastapi
import httpx
import prisma.actions
import prisma.models
import pytest
import pytest_asyncio
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
    server: SpinTestServer, users: list[str], monkeypatch: pytest.MonkeyPatch
):
    owner, first, second = await _user(users), await _user(users), await _user(users)
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
async def test_a_transfer_to_a_non_member_keeps_the_owner(
    server: SpinTestServer, users: list[str]
):
    owner, stranger = await _user(users), await _user(users)
    org = await _create_org(owner)

    with pytest.raises(NotFoundError):
        await orgs_db.transfer_ownership(org.id, owner, stranger)

    assert await _owners(org.id) == [owner]


@pytest.mark.asyncio(loop_scope="session")
async def test_a_member_who_belongs_to_no_other_org_is_removed(
    server: SpinTestServer, users: list[str]
):
    owner = await _user(users)
    org = await _create_org(owner)
    member = await _user_without_an_org(users)
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
    server: SpinTestServer, users: list[str], monkeypatch: pytest.MonkeyPatch
):
    owner, member = await _user(users), await _user(users)
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


@pytest.mark.parametrize(
    "change_meanwhile, refusal, member_stays",
    [
        pytest.param(
            lambda org, owner, member: orgs_db.transfer_ownership(org, owner, member),
            ValueError,
            True,
            id="made-owner",
        ),
        pytest.param(
            lambda org, owner, member: orgs_db.remove_org_member(
                org, member, requesting_user_id=owner
            ),
            NotFoundError,
            False,
            id="removed",
        ),
    ],
)
@pytest.mark.asyncio(loop_scope="session")
async def test_a_removal_says_what_happened_to_the_member_meanwhile(
    server: SpinTestServer,
    users: list[str],
    monkeypatch: pytest.MonkeyPatch,
    change_meanwhile: Callable[[str, str, str], Awaitable[None]],
    refusal: type[Exception],
    member_stays: bool,
):
    owner, member = await _user(users), await _user(users)
    org = await _create_org(owner)
    await orgs_db.add_org_member(org.id, member)
    _after_the_first_read_of(
        monkeypatch, member, lambda: change_meanwhile(org.id, owner, member)
    )

    with pytest.raises(refusal) as raised:
        await orgs_db.remove_org_member(org.id, member, requesting_user_id=owner)

    assert type(raised.value) is refusal  # NotFoundError is a ValueError too
    assert await _owners(org.id) == ([member] if member_stays else [owner])
    teams = await prisma.models.TeamMember.prisma().find_many(
        where={"userId": member, "Team": {"orgId": org.id}}
    )
    assert bool(teams) == member_stays


def _after_the_first_read_of(
    monkeypatch: pytest.MonkeyPatch, user_id: str, act: Callable[[], Awaitable[None]]
) -> None:
    """Run *act* once, right after the first read of *user_id*'s membership row."""
    real = prisma.actions.OrgMemberActions.find_unique
    pending = [act]

    async def find_unique(self, *args, **kwargs):
        result = await real(self, *args, **kwargs)
        key = (kwargs.get("where") or {}).get("orgId_userId", {})
        if pending and key.get("userId") == user_id:
            await pending.pop()()
        return result

    monkeypatch.setattr(prisma.actions.OrgMemberActions, "find_unique", find_unique)


def _meet_the_other_call_at_the_owner_row(
    monkeypatch: pytest.MonkeyPatch, owner_id: str
) -> None:
    """Each call waits for the other once it has read or written the owner's row,
    so both get past the owner check together unless the database serializes
    them. Only a write may wait in vain, because the database holds the second
    write until the first commits; reads that miss each other fail the test."""
    barrier = asyncio.Barrier(2)
    actions = prisma.actions.OrgMemberActions

    for name, may_wait_in_vain in (("find_unique", False), ("update_many", True)):
        real = getattr(actions, name)

        async def call(self, *args, _real=real, _lenient=may_wait_in_vain, **kwargs):
            result = await _real(self, *args, **kwargs)
            where = kwargs.get("where") or {}
            if owner_id in (
                where.get("userId"),
                where.get("orgId_userId", {}).get("userId"),
            ):
                try:
                    # hang-cap: correct code holds the other call, so this runs out.
                    await asyncio.wait_for(barrier.wait(), timeout=2)
                except (TimeoutError, asyncio.BrokenBarrierError):
                    if not _lenient:
                        raise
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


@pytest_asyncio.fixture(loop_scope="session")
async def users() -> AsyncIterator[list[str]]:
    """Ids of the users a test makes; they and every org they belong to go after it."""
    made: list[str] = []
    try:
        yield made
    finally:
        await prisma.models.Organization.prisma().delete_many(
            where={"Members": {"some": {"userId": {"in": made}}}}
        )
        await prisma.models.User.prisma().delete_many(where={"id": {"in": made}})


async def _user(users: list[str]) -> str:
    user = await get_or_create_user(
        {
            "sub": str(uuid.uuid4()),
            "email": f"org-membership-{uuid.uuid4().hex[:8]}@example.com",
            "name": "Org Membership",
        }
    )
    users.append(user.id)
    return user.id


async def _user_without_an_org(users: list[str]) -> str:
    """The row the sign-up hook writes, before anything creates a personal org."""
    user = await prisma.models.User.prisma().create(
        data={
            "id": str(uuid.uuid4()),
            "email": f"org-membership-{uuid.uuid4().hex[:8]}@example.com",
        }
    )
    users.append(user.id)
    return user.id


async def _owners(org_id: str) -> list[str]:
    owners = await prisma.models.OrgMember.prisma().find_many(
        where={"orgId": org_id, "isOwner": True}
    )
    return [m.userId for m in owners]


async def _create_org(user_id: str):
    slug = f"org-membership-{uuid.uuid4().hex[:12]}"
    return await orgs_db.create_org("Org Membership", slug, user_id)
