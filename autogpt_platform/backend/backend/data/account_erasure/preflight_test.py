"""What blocks an erasure, and which orgs go with the account: only orgs no
one else is an active member of. A shared org it alone owns, and an OAuth app
of its that other accounts still use, wait for a decision, an account that is not closed is refused unless the caller closes
it first, and a malformed ID is refused outright."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.enums import AccountDeletionSource

from backend.data.account_erasure import preflight
from backend.data.account_erasure.deleted_user import DELETED_USER_ID
from backend.data.account_erasure.models import Subject

USER_ID = "3f0b8c1e-4c2a-4f7e-9a51-0c6f2d8e1a77"
OTHER = "9d4e2b7a-1111-4c2a-9a51-0c6f2d8e1a77"
THIRD = "5a6b7c8d-2222-4c2a-9a51-0c6f2d8e1a77"
CLOSED = Subject(
    user_id=USER_ID, has_user_row=True, request_source=AccountDeletionSource.SETTINGS
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
    monkeypatch.setattr(preflight, "_oauth_apps_others_use", AsyncMock(return_value=0))
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
    blockers = await preflight.find_blockers(CLOSED)

    assert [(b.name, b.count) for b in blockers] == [
        ("shared_orgs_without_another_owner", 2)
    ]


@pytest.fixture
def no_orgs(monkeypatch):
    monkeypatch.setattr(preflight, "_orgs", AsyncMock(return_value=([], [])))
    monkeypatch.setattr(preflight, "_oauth_apps_others_use", AsyncMock(return_value=0))


@pytest.mark.asyncio
async def test_oauth_apps_other_accounts_use_block(no_orgs):
    """Deleting them would cut those accounts off, and handing them over
    would leave the developer's secret and redirect URIs with no owner here:
    the operator moves or disables them first."""
    preflight._oauth_apps_others_use.return_value = 2

    blockers = await preflight.find_blockers(CLOSED)

    assert [(b.name, b.count) for b in blockers] == [("oauth_apps_in_use", 2)]
    assert "ownerId" in blockers[0].resolution
    assert "isActive" in blockers[0].resolution


@pytest.mark.asyncio
async def test_only_active_apps_with_another_accounts_live_grant_count(
    monkeypatch,
):
    apps = MagicMock()
    apps.prisma.return_value.count = AsyncMock(return_value=1)
    monkeypatch.setattr(preflight, "OAuthApplication", apps)

    assert await preflight._oauth_apps_others_use(USER_ID) == 1

    where = apps.prisma.return_value.count.await_args.kwargs["where"]
    assert where["ownerId"] == USER_ID
    assert where["isActive"] is True
    grants = [next(iter(clause.items())) for clause in where["OR"]]
    assert [relation for relation, _ in grants] == ["RefreshTokens", "AccessTokens"]
    for _, grant in grants:
        live = grant["some"]
        assert live["userId"] == {"not": USER_ID}
        assert live["revokedAt"] is None
        assert "gt" in live["expiresAt"]


@pytest.mark.asyncio
async def test_an_open_account_is_not_erased(no_orgs):
    open_account = Subject(user_id=USER_ID, has_user_row=True)

    blockers = await preflight.find_blockers(open_account)

    assert [(b.name, b.count) for b in blockers] == [("not_closed", 0)]
    assert "--close" in blockers[0].resolution


@pytest.mark.asyncio
async def test_an_account_the_caller_closes_first_is_not_blocked(no_orgs):
    open_account = Subject(user_id=USER_ID, has_user_row=True)

    assert await preflight.find_blockers(open_account, closing=True) == []


BANNED = Subject(user_id=USER_ID, has_user_row=True, banned_for_another_reason=True)


@pytest.mark.asyncio
async def test_closing_refuses_an_account_banned_for_another_reason(no_orgs):
    """The Settings route refuses it too: deleting must not stand in for an
    abuse ban."""
    blockers = await preflight.find_blockers(BANNED, closing=True)

    assert [(b.name, b.count) for b in blockers] == [("banned", 1)]
    assert "--allow-banned" in blockers[0].resolution


@pytest.mark.asyncio
async def test_the_operator_can_allow_closing_a_banned_account(no_orgs):
    assert await preflight.find_blockers(BANNED, closing=True, allow_banned=True) == []


@pytest.mark.asyncio
async def test_a_banned_account_closed_already_is_erased(no_orgs):
    closed = BANNED.model_copy(update={"request_source": AccountDeletionSource.EMAIL})

    assert await preflight.find_blockers(closed) == []


@pytest.mark.parametrize(
    "identity, banned",
    [
        (None, False),
        (SimpleNamespace(banned=None, banReason=None), False),
        (SimpleNamespace(banned=False, banReason="spam"), False),
        (SimpleNamespace(banned=True, banReason="account_deleted"), False),
        (SimpleNamespace(banned=True, banReason="spam"), True),
        (SimpleNamespace(banned=True, banReason=None), True),
    ],
)
def test_only_a_ban_closing_did_not_set_counts_as_another_reason(identity, banned):
    assert preflight.banned_for_another_reason(identity) is banned


@pytest.mark.asyncio
async def test_a_closed_account_is_erased(no_orgs):
    assert await preflight.find_blockers(CLOSED) == []


@pytest.mark.asyncio
async def test_a_closed_account_whose_rows_are_gone_is_finished(no_orgs):
    """A rerun after a partial erasure: the request is still there."""
    nothing_left = Subject(
        user_id=USER_ID, request_source=AccountDeletionSource.SETTINGS
    )

    assert await preflight.find_blockers(nothing_left) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_id", ["", "x", "../other", "a b c d e f g h", "%" * 9])
async def test_a_malformed_id_is_refused_before_anything_else(no_orgs, bad_id):
    subject = Subject(user_id=bad_id, has_user_row=True, has_auth_identity=True)

    blockers = await preflight.find_blockers(subject, closing=True)

    assert [b.name for b in blockers] == ["invalid_account_id"]
    preflight._orgs.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_placeholder_account_is_never_deleted(no_orgs):
    placeholder = Subject(user_id=DELETED_USER_ID, has_user_row=True)

    blockers = await preflight.find_blockers(placeholder, closing=True)

    assert [b.name for b in blockers] == ["system_account"]


@pytest.mark.asyncio
async def test_stripe_customers_of_its_sole_orgs_but_not_its_own(monkeypatch):
    """Closing the account cancelled its own customer's subscriptions; the
    erasure covers only the sole orgs' other customers."""
    subscriptions = _table(
        AsyncMock(return_value=[SimpleNamespace(stripeCustomerId="cus_org_sub")])
    )
    monkeypatch.setattr(preflight, "OrganizationSubscription", subscriptions)
    user = SimpleNamespace(stripeCustomerId="cus_user")

    customers = await preflight._org_stripe_customers(
        user,
        [
            _org("a", customer="cus_user"),
            _org("b", customer=None),
            _org("c", customer="cus_org"),
        ],
    )

    assert customers == ["cus_org", "cus_org_sub"]
