"""Workspace membership guards in ``orgs.team_db`` against the real database.

Lives outside ``orgs/`` because that package's conftest stubs out the
``server`` fixture, and these tests need Postgres: the property under test is
that two concurrent admin-roster changes cannot both pass the last-admin check,
which an in-memory store cannot show.
"""

import asyncio
import contextlib
import uuid
from collections.abc import Awaitable

import pytest
from prisma.models import Organization, OrgMember, Team, TeamMember, User

from backend.api.features.orgs import team_db
from backend.util.exceptions import ConflictError, NotFoundError
from backend.util.test import SpinTestServer


async def _team(*admins: str, members: tuple[str, ...] = ()) -> tuple[str, str]:
    """Create an org with a non-default team; returns (org_id, team_id)."""
    suffix = uuid.uuid4().hex[:8]
    org = await Organization.prisma().create(
        data={"name": f"Guards {suffix}", "slug": f"guards-{suffix}"}
    )
    team = await Team.prisma().create(
        data={"name": "Squad", "orgId": org.id, "isDefault": False}
    )
    for user_id in (*admins, *members):
        await _org_member(org.id, user_id)
    for user_id in admins:
        await TeamMember.prisma().create(
            data={
                "teamId": team.id,
                "userId": user_id,
                "isAdmin": True,
                "status": "ACTIVE",
            }
        )
    for user_id in members:
        await TeamMember.prisma().create(
            data={"teamId": team.id, "userId": user_id, "status": "ACTIVE"}
        )
    return org.id, team.id


async def _org_member(org_id: str, user_id: str) -> None:
    await User.prisma().create(data={"id": user_id, "email": f"{user_id}@example.com"})
    await OrgMember.prisma().create(
        data={"orgId": org_id, "userId": user_id, "status": "ACTIVE"}
    )


async def _admins(team_id: str) -> set[str]:
    rows = await TeamMember.prisma().find_many(
        where={"teamId": team_id, "isAdmin": True, "status": "ACTIVE"}
    )
    return {r.userId for r in rows}


def _user() -> str:
    return f"guard-{uuid.uuid4()}"


async def _race(mocker, first: Awaitable, second: Awaitable) -> list:
    """Run two roster changes so that, absent the admin lock, both count first.

    The first caller to reach the admin count is held there until the second
    reaches it too. Without the lock that always happens, both see two admins,
    and both write. With the lock the second is parked on it and never gets
    there, so the hold times out and the changes run in sequence. The timeout
    only bounds how long the passing case waits; it cannot make it fail.
    """
    real_count = team_db._count_active_admins
    both_counting = asyncio.Event()
    entered = 0

    async def gated(tx, ws_id: str) -> int:
        nonlocal entered
        entered += 1
        mine = entered
        if mine > 1:
            both_counting.set()
        count = await real_count(tx, ws_id)
        if mine == 1:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(both_counting.wait(), timeout=2)
        return count

    mocker.patch.object(team_db, "_count_active_admins", gated)
    outcomes = await asyncio.gather(first, second, return_exceptions=True)
    assert entered == 2, "both changes must have reached the admin count"
    return outcomes


def _refused(outcomes: list) -> list:
    unexpected = [
        o
        for o in outcomes
        if isinstance(o, BaseException) and type(o) is not ValueError
    ]
    assert not unexpected, unexpected
    return [o for o in outcomes if isinstance(o, ValueError)]


@pytest.mark.asyncio(loop_scope="session")
async def test_two_admins_demoting_each_other_keep_one_admin(
    server: SpinTestServer, mocker
):
    a1, a2 = _user(), _user()
    _, team = await _team(a1, a2)

    outcomes = await _race(
        mocker,
        team_db.update_team_member(team, a1, False, None),
        team_db.update_team_member(team, a2, False, None),
    )

    assert len(_refused(outcomes)) == 1, outcomes
    assert len(await _admins(team)) == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_two_admins_leaving_together_keep_one_admin(
    server: SpinTestServer, mocker
):
    a1, a2 = _user(), _user()
    member = _user()
    _, team = await _team(a1, a2, members=(member,))

    outcomes = await _race(
        mocker, team_db.leave_team(team, a1), team_db.leave_team(team, a2)
    )

    assert len(_refused(outcomes)) == 1, outcomes
    assert len(await _admins(team)) == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_leaving_while_the_other_admin_is_removed_keeps_one_admin(
    server: SpinTestServer, mocker
):
    a1, a2 = _user(), _user()
    _, team = await _team(a1, a2)

    outcomes = await _race(
        mocker, team_db.leave_team(team, a1), team_db.remove_team_member(team, a2)
    )

    assert len(_refused(outcomes)) == 1, outcomes
    assert len(await _admins(team)) == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_the_sole_admin_cannot_leave_or_be_demoted(server: SpinTestServer):
    admin, member = _user(), _user()
    _, team = await _team(admin, members=(member,))

    with pytest.raises(ValueError, match="last workspace admin"):
        await team_db.leave_team(team, admin)
    with pytest.raises(ValueError, match="last workspace admin"):
        await team_db.update_team_member(team, admin, False, None)
    with pytest.raises(ValueError, match="last workspace admin"):
        await team_db.remove_team_member(team, admin)

    promoted = await team_db.update_team_member(team, member, True, None)
    assert promoted.is_admin is True
    await team_db.leave_team(team, admin)
    assert await _admins(team) == {member}


@pytest.mark.asyncio(loop_scope="session")
async def test_adding_a_member_twice_at_once_adds_them_once(server: SpinTestServer):
    admin, newcomer = _user(), _user()
    org, team = await _team(admin)
    await _org_member(org, newcomer)

    outcomes = await asyncio.gather(
        team_db.add_team_member(team, newcomer, org),
        team_db.add_team_member(team, newcomer, org),
        return_exceptions=True,
    )

    conflicts = [o for o in outcomes if isinstance(o, ConflictError)]
    added = [o for o in outcomes if not isinstance(o, BaseException)]
    assert (len(conflicts), len(added)) == (1, 1), outcomes
    assert (
        await TeamMember.prisma().count(where={"teamId": team, "userId": newcomer}) == 1
    )
    with pytest.raises(ConflictError):
        await team_db.add_team_member(team, newcomer, org)


@pytest.mark.asyncio(loop_scope="session")
async def test_updating_a_non_member_is_not_found_and_writes_nothing(
    server: SpinTestServer,
):
    admin, outsider = _user(), _user()
    _, team = await _team(admin)

    with pytest.raises(NotFoundError):
        await team_db.update_team_member(team, outsider, True, None)
    assert await TeamMember.prisma().count(where={"teamId": team}) == 1
