"""What blocks a deletion, and which orgs go with the account: only orgs no
one else is an active member of. A shared org it alone owns waits for a
decision, the deletion waits out the undo window after the lock unless
forced, a malformed ID is refused outright, and an ID with no account left
needs --force."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.data.account_deletion import preflight
from backend.data.account_deletion.deleted_user import DELETED_USER_ID
from backend.data.account_deletion.models import Subject

USER_ID = "3f0b8c1e-4c2a-4f7e-9a51-0c6f2d8e1a77"
OTHER = "9d4e2b7a-1111-4c2a-9a51-0c6f2d8e1a77"
THIRD = "5a6b7c8d-2222-4c2a-9a51-0c6f2d8e1a77"
NOW = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)
LOCKED_LONG_AGO = Subject(
    user_id=USER_ID, has_user_row=True, locked_at=NOW - timedelta(days=15)
)


def _org(org_id: str, *, personal: bool = False, customer: str | None = None):
    return SimpleNamespace(id=org_id, isPersonal=personal, stripeCustomerId=customer)


def _member(org, user_id: str, *, owner: bool, status: str = "ACTIVE"):
    return SimpleNamespace(
        orgId=org.id, userId=user_id, isOwner=owner, status=status, Org=org
    )


def _table(find_many) -> MagicMock:
    table = MagicMock()
    table.prisma.return_value.find_many = AsyncMock(side_effect=find_many)
    return table


@pytest.fixture
def orgs(monkeypatch):
    """Personal org (alone), a team org it alone owns with a member, a team
    org with another owner, a team org it is the only member of, a team org
    whose only other member is suspended and one whose other owner was
    removed."""
    personal = _org("personal", personal=True, customer="cus_personal")
    owned_shared = _org("owned-shared")
    co_owned = _org("co-owned")
    solo_team = _org("solo-team")
    suspended_other = _org("suspended-other")
    removed_owner = _org("removed-owner")
    members = {
        "personal": [_member(personal, USER_ID, owner=True)],
        "owned-shared": [
            _member(owned_shared, USER_ID, owner=True),
            _member(owned_shared, OTHER, owner=False),
        ],
        "co-owned": [
            _member(co_owned, USER_ID, owner=True),
            _member(co_owned, OTHER, owner=True),
        ],
        "solo-team": [_member(solo_team, USER_ID, owner=False)],
        "suspended-other": [
            _member(suspended_other, USER_ID, owner=True),
            _member(suspended_other, OTHER, owner=True, status="SUSPENDED"),
        ],
        "removed-owner": [
            _member(removed_owner, USER_ID, owner=True),
            _member(removed_owner, OTHER, owner=False),
            _member(removed_owner, THIRD, owner=True, status="REMOVED"),
        ],
    }

    async def find_members(where, include=None):
        if "orgId" not in where:
            return [m for ms in members.values() for m in ms if m.userId == USER_ID]
        return [
            m
            for m in members[where["orgId"]]
            if m.userId != USER_ID
            and ("status" not in where or m.status == where["status"])
        ]

    monkeypatch.setattr(preflight, "OrgMember", _table(find_members))
    monkeypatch.setattr(
        preflight, "Organization", _table(AsyncMock(return_value=[personal]))
    )
    return members


@pytest.mark.asyncio
async def test_only_orgs_nobody_else_is_active_in_go_with_the_account(orgs):
    sole, solely_owned_shared = await preflight._orgs(USER_ID)

    # A suspended member is not another member: the org goes, and its
    # subscription is cancelled with the account's.
    assert sorted(org.id for org in sole) == [
        "personal",
        "solo-team",
        "suspended-other",
    ]
    # A removed owner is not another owner.
    assert sorted(org.id for org in solely_owned_shared) == [
        "owned-shared",
        "removed-owner",
    ]


@pytest.mark.asyncio
async def test_a_shared_org_without_another_owner_blocks(orgs):
    blockers = await preflight.find_blockers(LOCKED_LONG_AGO, now=NOW)

    assert [(b.name, b.count) for b in blockers] == [
        ("shared_orgs_without_another_owner", 2)
    ]


@pytest.fixture
def no_orgs(monkeypatch):
    monkeypatch.setattr(preflight, "_orgs", AsyncMock(return_value=([], [])))


@pytest.mark.asyncio
async def test_an_account_never_locked_waits_for_a_lock(no_orgs):
    subject = Subject(user_id=USER_ID, has_user_row=True)

    blockers = await preflight.find_blockers(subject, now=NOW)

    assert [(b.name, b.count) for b in blockers] == [("not_locked", 14)]
    assert "--lock --apply" in blockers[0].resolution


@pytest.mark.asyncio
async def test_a_recent_lock_waits_out_the_undo_window(no_orgs):
    subject = Subject(
        user_id=USER_ID, has_user_row=True, locked_at=NOW - timedelta(days=10)
    )

    (blocker,) = await preflight.find_blockers(subject, now=NOW)

    assert (blocker.name, blocker.count) == ("undo_window_open", 4)
    assert "2026-10-19 12:00" in blocker.resolution


@pytest.mark.asyncio
async def test_after_the_undo_window_nothing_blocks(no_orgs):
    subject = Subject(
        user_id=USER_ID, has_user_row=True, locked_at=NOW - timedelta(days=14)
    )

    assert await preflight.find_blockers(subject, now=NOW) == []


@pytest.mark.asyncio
async def test_force_skips_the_undo_window_but_not_other_blockers(orgs):
    subject = Subject(user_id=USER_ID, has_user_row=True)

    blockers = await preflight.find_blockers(subject, force=True, now=NOW)

    assert [b.name for b in blockers] == ["shared_orgs_without_another_owner"]


@pytest.mark.asyncio
async def test_without_an_account_row_there_is_no_lock_to_wait_for(no_orgs):
    leftover_identity = Subject(user_id=USER_ID, has_auth_identity=True)

    assert await preflight.find_blockers(leftover_identity, now=NOW) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_id", ["", "x", "../other", "a b c d e f g h", "%" * 9])
async def test_a_malformed_id_is_refused_before_anything_else(no_orgs, bad_id):
    subject = Subject(user_id=bad_id, has_user_row=True, has_auth_identity=True)

    blockers = await preflight.find_blockers(subject, force=True, now=NOW)

    assert [b.name for b in blockers] == ["invalid_account_id"]
    preflight._orgs.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_unknown_account_needs_force(no_orgs):
    nothing_left = Subject(user_id=USER_ID)

    blockers = await preflight.find_blockers(nothing_left, now=NOW)

    assert [(b.name, b.count) for b in blockers] == [("account_not_found", 0)]
    assert "--force" in blockers[0].resolution


@pytest.mark.asyncio
async def test_force_finishes_an_account_whose_rows_are_gone(no_orgs):
    nothing_left = Subject(user_id=USER_ID)

    assert await preflight.find_blockers(nothing_left, force=True, now=NOW) == []


@pytest.mark.asyncio
async def test_the_placeholder_account_is_never_deleted(no_orgs):
    placeholder = Subject(user_id=DELETED_USER_ID, has_user_row=True)

    blockers = await preflight.find_blockers(placeholder, force=True, now=NOW)

    assert [b.name for b in blockers] == ["system_account"]


@pytest.mark.asyncio
async def test_stripe_customers_of_the_account_and_its_sole_orgs(monkeypatch):
    subscriptions = _table(
        AsyncMock(return_value=[SimpleNamespace(stripeCustomerId="cus_org_sub")])
    )
    monkeypatch.setattr(preflight, "OrganizationSubscription", subscriptions)
    user = SimpleNamespace(stripeCustomerId="cus_user")

    customers = await preflight._stripe_customers(
        user, [_org("a", customer="cus_user"), _org("b", customer=None)]
    )

    assert customers == ["cus_user", "cus_org_sub"]
