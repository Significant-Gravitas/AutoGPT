"""The org and workspace counts hold under concurrent writes; real database.

Outside `orgs/`, whose conftest replaces the `server` fixture with a no-op.
"""

import asyncio
import uuid

import prisma.actions
import prisma.errors
import prisma.models
import pytest

from backend.api.features.orgs import db as orgs_db
from backend.api.features.orgs import team_db
from backend.data.user import get_or_create_user
from backend.util.test import SpinTestServer


@pytest.mark.asyncio(loop_scope="session")
async def test_concurrent_org_creates_at_the_bound_save_only_one(
    server: SpinTestServer, monkeypatch: pytest.MonkeyPatch
):
    user_id = await _user()
    monkeypatch.setattr(orgs_db, "MAX_ORGS_PER_USER", await _memberships(user_id) + 1)
    _count_then_meet_the_other_call(monkeypatch, prisma.actions.OrgMemberActions)

    results = await asyncio.gather(
        _create_org(user_id), _create_org(user_id), return_exceptions=True
    )

    assert sorted(type(r).__name__ for r in results) == ["OrgResponse", "ValueError"]
    assert await _memberships(user_id) == orgs_db.MAX_ORGS_PER_USER


@pytest.mark.asyncio(loop_scope="session")
async def test_concurrent_workspace_creates_at_the_bound_save_only_one(
    server: SpinTestServer, monkeypatch: pytest.MonkeyPatch
):
    user_id = await _user()
    org = await _create_org(user_id)
    monkeypatch.setattr(team_db, "MAX_TEAMS_PER_ORG", await _teams(org.id) + 1)
    _count_then_meet_the_other_call(monkeypatch, prisma.actions.TeamActions)

    results = await asyncio.gather(
        team_db.create_team(org.id, "one", user_id),
        team_db.create_team(org.id, "two", user_id),
        return_exceptions=True,
    )

    assert sorted(type(r).__name__ for r in results) == ["TeamResponse", "ValueError"]
    assert await _teams(org.id) == team_db.MAX_TEAMS_PER_ORG


@pytest.mark.asyncio(loop_scope="session")
async def test_joining_an_org_at_the_bound_is_refused(
    server: SpinTestServer, monkeypatch: pytest.MonkeyPatch
):
    """An invitation adds the membership through `add_org_member`."""
    user_id, owner_id = await _user(), await _user()
    joined, other = await _create_org(owner_id), await _create_org(owner_id)
    await orgs_db.add_org_member(joined.id, user_id)
    monkeypatch.setattr(orgs_db, "MAX_ORGS_PER_USER", await _memberships(user_id))

    with pytest.raises(ValueError, match="at most"):
        await orgs_db.add_org_member(other.id, user_id)
    # Re-joining an org already counted is the duplicate it always was.
    with pytest.raises(prisma.errors.UniqueViolationError):
        await orgs_db.add_org_member(joined.id, user_id)
    assert await _memberships(user_id) == orgs_db.MAX_ORGS_PER_USER


@pytest.mark.asyncio(loop_scope="session")
async def test_converting_a_personal_org_at_the_bound_is_refused(
    server: SpinTestServer, monkeypatch: pytest.MonkeyPatch
):
    """The conversion adds a new personal org, so it counts like a create."""
    user_id = await _user()
    personal = await prisma.models.Organization.prisma().find_first_or_raise(
        where={"bootstrapUserId": user_id, "isPersonal": True}
    )
    monkeypatch.setattr(orgs_db, "MAX_ORGS_PER_USER", await _memberships(user_id))

    with pytest.raises(ValueError, match="at most"):
        await orgs_db.convert_personal_org(personal.id, user_id)
    assert (await orgs_db.get_org(personal.id)).is_personal
    assert await _memberships(user_id) == orgs_db.MAX_ORGS_PER_USER

    before = await _memberships(user_id)
    monkeypatch.setattr(orgs_db, "MAX_ORGS_PER_USER", before + 1)
    assert not (await orgs_db.convert_personal_org(personal.id, user_id)).is_personal
    assert await _memberships(user_id) == before + 1


def _count_then_meet_the_other_call(monkeypatch: pytest.MonkeyPatch, actions) -> None:
    """Both calls reach the count together unless the lock serializes them, in
    which case the first one's wait times out: that is the passing outcome."""
    barrier = asyncio.Barrier(2)
    real_count = actions.count

    async def count(self, *args, **kwargs):
        result = await real_count(self, *args, **kwargs)
        try:
            await asyncio.wait_for(barrier.wait(), timeout=2)
        except (TimeoutError, asyncio.BrokenBarrierError):
            pass
        return result

    monkeypatch.setattr(actions, "count", count)


async def _user() -> str:
    user = await get_or_create_user(
        {
            "sub": str(uuid.uuid4()),
            "email": f"org-count-{uuid.uuid4().hex[:8]}@example.com",
            "name": "Org Count",
        }
    )
    return user.id


async def _create_org(user_id: str):
    slug = f"org-count-{uuid.uuid4().hex[:12]}"
    return await orgs_db.create_org("Org Count", slug, user_id)


async def _memberships(user_id: str) -> int:
    return await prisma.models.OrgMember.prisma().count(
        where={"userId": user_id, "status": "ACTIVE", "Org": {"deletedAt": None}}
    )


async def _teams(org_id: str) -> int:
    return await prisma.models.Team.prisma().count(
        where={"orgId": org_id, "archivedAt": None}
    )
