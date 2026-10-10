"""Tests for workspace member guards in team_db:

- #15259: leave/demote must not leave a workspace without an admin.
- #15264: adding an existing member is a 409 ConflictError, not a 500.
- #15265: updating a non-member is a 404 NotFoundError, not StopIteration.

Like the other org tests these run without a database: ``prisma`` is replaced
by a small in-memory store that implements the TeamMember queries team_db
uses, so the real guard logic runs against real-looking rows.
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from prisma.errors import UniqueViolationError

from backend.api.features.orgs import team_db
from backend.util.exceptions import ConflictError, NotFoundError

WS = "ws-1"
ORG = "org-1"


class _TeamMembers:
    def __init__(self):
        self.rows: dict[tuple[str, str], SimpleNamespace] = {}

    def add(self, user_id, *, is_admin=False, status="ACTIVE"):
        self.rows[(WS, user_id)] = SimpleNamespace(
            id=f"tm-{user_id}",
            teamId=WS,
            userId=user_id,
            isAdmin=is_admin,
            isBillingManager=False,
            status=status,
            joinedAt=datetime(2026, 1, 1, tzinfo=timezone.utc),
            User=SimpleNamespace(email=f"{user_id}@example.com", name=user_id),
        )

    @staticmethod
    def _key(where):
        k = where["teamId_userId"]
        return (k["teamId"], k["userId"])

    @staticmethod
    def _matches(row, where):
        return all(getattr(row, f) == v for f, v in where.items())

    async def find_unique(self, where, include=None):
        return self.rows.get(self._key(where))

    async def find_many(self, where, include=None):
        return [r for r in self.rows.values() if self._matches(r, where)]

    async def count(self, where):
        return len(await self.find_many(where))

    async def create(self, data, include=None):
        if (data["teamId"], data["userId"]) in self.rows:
            raise UniqueViolationError(
                {"user_facing_error": {"message": "Unique constraint failed"}}
            )
        self.add(data["userId"], is_admin=data.get("isAdmin", False))
        return self.rows[(data["teamId"], data["userId"])]

    async def update(self, where, data):
        row = self.rows.get(self._key(where))
        if row is None:
            return None  # prisma-client-py returns None for a missing row
        for field, value in data.items():
            setattr(row, field, value)
        return row

    async def delete(self, where):
        return self.rows.pop(self._key(where), None)

    async def delete_many(self, where):
        doomed = [k for k, r in self.rows.items() if self._matches(r, where)]
        for k in doomed:
            del self.rows[k]
        return len(doomed)


class _Store:
    def __init__(self):
        self.teammember = _TeamMembers()
        self.team = SimpleNamespace(find_unique=self._find_team)
        self.orgmember = SimpleNamespace(find_unique=self._find_org_member)

    async def _find_team(self, where):
        return SimpleNamespace(id=WS, orgId=ORG, isDefault=False)

    async def _find_org_member(self, where):
        return SimpleNamespace(orgId=ORG, userId=where["orgId_userId"]["userId"])


@pytest.fixture
def store():
    fake = _Store()
    with patch.object(team_db, "prisma", fake):
        yield fake


# --- #15259 ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_sole_admin_cannot_leave(store):
    store.teammember.add("admin", is_admin=True)
    store.teammember.add("member")

    with pytest.raises(ValueError, match="last workspace admin"):
        await team_db.leave_team(WS, "admin")
    assert (WS, "admin") in store.teammember.rows


@pytest.mark.asyncio
async def test_sole_admin_cannot_be_demoted(store):
    store.teammember.add("admin", is_admin=True)

    with pytest.raises(ValueError, match="last workspace admin"):
        await team_db.update_team_member(WS, "admin", False, None)
    assert store.teammember.rows[(WS, "admin")].isAdmin is True


@pytest.mark.asyncio
async def test_one_of_two_admins_can_leave_or_be_demoted(store):
    store.teammember.add("a1", is_admin=True)
    store.teammember.add("a2", is_admin=True)

    updated = await team_db.update_team_member(WS, "a1", False, None)
    assert updated.is_admin is False

    store.teammember.rows[(WS, "a1")].isAdmin = True
    await team_db.leave_team(WS, "a2")
    assert (WS, "a2") not in store.teammember.rows


@pytest.mark.asyncio
async def test_non_admin_can_leave(store):
    store.teammember.add("admin", is_admin=True)
    store.teammember.add("member")

    await team_db.leave_team(WS, "member")
    assert (WS, "member") not in store.teammember.rows


@pytest.mark.asyncio
async def test_remove_still_guards_last_admin(store):
    store.teammember.add("admin", is_admin=True)

    with pytest.raises(ValueError, match="Cannot remove the last workspace admin"):
        await team_db.remove_team_member(WS, "admin")


# --- #15264 ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_adding_existing_member_is_a_conflict(store):
    await team_db.add_team_member(WS, "u1", ORG)

    with pytest.raises(ConflictError, match="already a member"):
        await team_db.add_team_member(WS, "u1", ORG)


@pytest.mark.asyncio
async def test_add_race_unique_violation_is_a_conflict(store):
    # Another request inserts the row between the existence check and create.
    real_find_unique = store.teammember.find_unique

    async def miss_then_real(where, include=None):
        store.teammember.find_unique = real_find_unique
        return None

    store.teammember.add("u1")
    store.teammember.find_unique = miss_then_real

    with pytest.raises(ConflictError):
        await team_db.add_team_member(WS, "u1", ORG)


# --- #15265 ---------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("flags", [(True, None), (None, None)])
async def test_updating_non_member_is_not_found(store, flags):
    store.teammember.add("admin", is_admin=True)

    with pytest.raises(NotFoundError):
        await team_db.update_team_member(WS, "stranger", *flags)


@pytest.mark.asyncio
async def test_update_existing_member_returns_new_flags(store):
    store.teammember.add("admin", is_admin=True)
    store.teammember.add("member")

    updated = await team_db.update_team_member(WS, "member", True, True)
    assert (updated.is_admin, updated.is_billing_manager) == (True, True)
