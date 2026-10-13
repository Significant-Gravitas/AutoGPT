"""What blocks a deletion, and which orgs go with the account: only orgs no
one else belongs to. A shared org it alone owns, or its agents in other
accounts' libraries, wait for a decision."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.data.account_deletion import preflight
from backend.data.account_deletion.models import Subject
from backend.data.account_deletion.preflight import SharedAgentCounts

USER_ID = "3f0b8c1e-4c2a-4f7e-9a51-0c6f2d8e1a77"
OTHER = "9d4e2b7a-1111-4c2a-9a51-0c6f2d8e1a77"


def _org(org_id: str, *, personal: bool = False, customer: str | None = None):
    return SimpleNamespace(id=org_id, isPersonal=personal, stripeCustomerId=customer)


def _member(org, user_id: str, *, owner: bool):
    return SimpleNamespace(orgId=org.id, userId=user_id, isOwner=owner, Org=org)


def _table(find_many) -> MagicMock:
    table = MagicMock()
    table.prisma.return_value.find_many = AsyncMock(side_effect=find_many)
    return table


@pytest.fixture
def orgs(monkeypatch):
    """Personal org (alone), a team org it alone owns with a member, a team
    org with another owner, and a team org it is the only member of."""
    personal = _org("personal", personal=True, customer="cus_personal")
    owned_shared = _org("owned-shared")
    co_owned = _org("co-owned")
    solo_team = _org("solo-team")
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
    }

    async def find_members(where, include=None):
        if "orgId" not in where:
            return [m for ms in members.values() for m in ms if m.userId == USER_ID]
        return [m for m in members[where["orgId"]] if m.userId != USER_ID]

    monkeypatch.setattr(preflight, "OrgMember", _table(find_members))
    monkeypatch.setattr(
        preflight, "Organization", _table(AsyncMock(return_value=[personal]))
    )
    return members


@pytest.mark.asyncio
async def test_only_orgs_nobody_else_is_in_go_with_the_account(orgs):
    sole, solely_owned_shared = await preflight._orgs(USER_ID)

    assert sorted(org.id for org in sole) == ["personal", "solo-team"]
    assert [org.id for org in solely_owned_shared] == ["owned-shared"]


@pytest.mark.asyncio
async def test_a_shared_org_without_another_owner_blocks(orgs):
    blockers = await preflight.find_blockers(
        Subject(user_id=USER_ID), SharedAgentCounts(), remove_shared_agents=False
    )

    assert [(b.name, b.count) for b in blockers] == [
        ("shared_orgs_without_another_owner", 1)
    ]


@pytest.mark.asyncio
async def test_agents_in_other_libraries_block_unless_removal_is_allowed(
    monkeypatch,
):
    monkeypatch.setattr(preflight, "_orgs", AsyncMock(return_value=([], [])))
    shared = SharedAgentCounts(library_entries=2, presets=1)

    blocked = await preflight.find_blockers(
        Subject(user_id=USER_ID), shared, remove_shared_agents=False
    )
    allowed = await preflight.find_blockers(
        Subject(user_id=USER_ID), shared, remove_shared_agents=True
    )

    assert [(b.name, b.count) for b in blocked] == [
        ("agents_used_by_other_accounts", 3)
    ]
    assert allowed == []


@pytest.mark.asyncio
async def test_another_accounts_listing_of_its_agent_always_blocks(monkeypatch):
    monkeypatch.setattr(preflight, "_orgs", AsyncMock(return_value=([], [])))

    blockers = await preflight.find_blockers(
        Subject(user_id=USER_ID),
        SharedAgentCounts(foreign_listings=1),
        remove_shared_agents=True,
    )

    assert [b.name for b in blockers] == ["agents_listed_by_other_accounts"]


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
