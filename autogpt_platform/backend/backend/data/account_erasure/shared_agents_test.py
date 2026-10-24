"""Agents other accounts still use are handed to the "deleted user" account
with their listings, in one transaction, instead of deleted; a dry run only
counts, and an account with none never creates the placeholder."""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.data.account_erasure import shared_agents
from backend.data.account_erasure.deleted_user import DELETED_USER_ID
from backend.data.account_erasure.models import StepStatus, Subject

USER_ID = "3f0b8c1e-4c2a-4f7e-9a51-0c6f2d8e1a77"
SUBJECT = Subject(
    user_id=USER_ID,
    shared_graph_ids=["graph-published", "graph-sub"],
    kept_listing_ids=["listing-1"],
)


@pytest.fixture
def db(monkeypatch):
    tx = MagicMock(name="tx")

    @asynccontextmanager
    async def transaction(timeout):
        yield tx

    listings = MagicMock()
    listings.prisma.return_value.find_many = AsyncMock(
        return_value=[SimpleNamespace(id="listing-1", slug="weather")]
    )
    listings.prisma.return_value.find_first = AsyncMock(return_value=None)
    profiles = MagicMock()
    profiles.prisma.return_value.find_unique = AsyncMock(
        return_value=SimpleNamespace(id="profile-of-user")
    )
    execute = AsyncMock(return_value=3)
    ensure = AsyncMock(return_value="profile-of-placeholder")
    clear = MagicMock()
    monkeypatch.setattr(shared_agents, "transaction", transaction)
    monkeypatch.setattr(shared_agents, "StoreListing", listings)
    monkeypatch.setattr(shared_agents, "Profile", profiles)
    monkeypatch.setattr(shared_agents, "execute_raw_with_schema", execute)
    monkeypatch.setattr(shared_agents, "ensure_deleted_user", ensure)
    monkeypatch.setattr(shared_agents, "clear_store_caches", clear)
    monkeypatch.setattr(
        shared_agents,
        "query_raw_with_schema",
        AsyncMock(return_value=[{"n": 2}]),
    )
    return SimpleNamespace(
        tx=tx,
        listings=listings,
        execute=execute,
        ensure=ensure,
        clear=clear,
    )


def _listing_update(execute: AsyncMock):
    return next(
        call
        for call in execute.await_args_list
        if 'UPDATE {schema_prefix}"StoreListing" ' in call.args[0]
    )


@pytest.mark.asyncio
async def test_a_dry_run_counts_and_changes_nothing(db):
    report = await shared_agents.transfer_shared_agents(SUBJECT, dry_run=True)

    assert report.status == StepStatus.WOULD_DO
    assert report.counts == {
        "agents": 2,
        "listings": 1,
        "library_entries_relabelled": 2,
    }
    db.ensure.assert_not_awaited()
    db.execute.assert_not_awaited()
    db.clear.assert_not_called()


@pytest.mark.asyncio
async def test_an_account_nobody_else_uses_creates_no_placeholder(db):
    lonely = Subject(user_id=USER_ID)

    report = await shared_agents.transfer_shared_agents(lonely, dry_run=False)

    assert report.status == StepStatus.DONE
    assert set(report.counts.values()) == {0}
    db.ensure.assert_not_awaited()
    db.execute.assert_not_awaited()
    db.clear.assert_not_called()


@pytest.mark.asyncio
async def test_agents_and_listings_go_to_the_placeholder_in_one_transaction(db):
    report = await shared_agents.transfer_shared_agents(SUBJECT, dry_run=False)

    assert report.status == StepStatus.DONE
    db.ensure.assert_awaited_once_with(db.tx)
    for call in db.execute.await_args_list:
        assert call.kwargs["client"] is db.tx
    graphs = next(
        call
        for call in db.execute.await_args_list
        if 'UPDATE {schema_prefix}"AgentGraph"' in call.args[0]
    )
    # Only this account's rows, only the shared ones.
    assert '"userId" = $2' in graphs.args[0]
    assert graphs.args[1:] == (DELETED_USER_ID, USER_ID, SUBJECT.shared_graph_ids)
    listing = _listing_update(db.execute)
    assert listing.args[1:] == (DELETED_USER_ID, "weather", "listing-1", USER_ID)
    db.clear.assert_called_once()


@pytest.mark.asyncio
async def test_other_accounts_library_entries_show_the_placeholder_as_creator(db):
    await shared_agents.transfer_shared_agents(SUBJECT, dry_run=False)

    relabel = next(
        call
        for call in db.execute.await_args_list
        if 'UPDATE {schema_prefix}"LibraryAgent"' in call.args[0]
    )
    assert relabel.args[1:] == ("profile-of-placeholder", "profile-of-user", USER_ID)
    assert '"userId" <> $3' in relabel.args[0]


@pytest.mark.asyncio
async def test_pending_submissions_of_kept_listings_are_withdrawn(db):
    await shared_agents.transfer_shared_agents(SUBJECT, dry_run=False)

    withdraw = next(
        call
        for call in db.execute.await_args_list
        if 'UPDATE {schema_prefix}"StoreListingVersion"' in call.args[0]
    )
    assert "'PENDING'" in withdraw.args[0]
    assert withdraw.args[1:] == (SUBJECT.kept_listing_ids,)


@pytest.mark.asyncio
async def test_a_slug_the_placeholder_already_uses_gets_a_suffix(db):
    taken = {"weather", "weather-2"}

    async def find_first(where):
        return SimpleNamespace(id="other") if where["slug"] in taken else None

    db.listings.prisma.return_value.find_first = AsyncMock(side_effect=find_first)

    await shared_agents.transfer_shared_agents(SUBJECT, dry_run=False)

    assert _listing_update(db.execute).args[2] == "weather-3"


@pytest.mark.asyncio
async def test_the_plan_reads_shared_graphs_then_their_listings(monkeypatch):
    query = AsyncMock(
        side_effect=[
            [{"id": "graph-published"}, {"id": "graph-sub"}],
            [{"id": "listing-1"}],
        ]
    )
    monkeypatch.setattr(shared_agents, "query_raw_with_schema", query)

    plan = await shared_agents.plan_transfer(USER_ID)

    assert plan.graph_ids == ["graph-published", "graph-sub"]
    assert plan.listing_ids == ["listing-1"]
    graphs_sql, *graph_args = query.await_args_list[0].args
    assert "WITH RECURSIVE" in graphs_sql
    # Org-home and team agents in an org someone else is active in are seen
    # by that org's members (`tenancy.visibility_filter`), so they count.
    assert '"OrgMember"' in graphs_sql and "'ACTIVE'" in graphs_sql
    assert 'g."teamId"' in graphs_sql
    assert graph_args[0] == USER_ID
    assert query.await_args_list[1].args[1:] == (
        USER_ID,
        ["graph-published", "graph-sub"],
    )


@pytest.mark.asyncio
async def test_no_shared_graphs_skips_the_listing_lookup(monkeypatch):
    query = AsyncMock(return_value=[])
    monkeypatch.setattr(shared_agents, "query_raw_with_schema", query)

    plan = await shared_agents.plan_transfer(USER_ID)

    assert plan.graph_ids == [] and plan.listing_ids == []
    assert query.await_count == 1
