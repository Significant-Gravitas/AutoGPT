"""Listings and reviews reference the account with ON DELETE RESTRICT, so the
marketplace step removes them (and the listings' search embeddings) before the
account's rows go. Other accounts' library entries of its agents go only when
the operator allowed it."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.data.account_deletion import marketplace
from backend.data.account_deletion.models import StepStatus, Subject

USER_ID = "3f0b8c1e-4c2a-4f7e-9a51-0c6f2d8e1a77"
SUBJECT = Subject(user_id=USER_ID)


def _table(count: int = 0, rows: list | None = None) -> MagicMock:
    table = MagicMock()
    actions = table.prisma.return_value
    actions.count = AsyncMock(return_value=count)
    actions.find_many = AsyncMock(return_value=rows or [])
    actions.delete_many = AsyncMock(return_value=count)
    return table


@pytest.fixture
def store(monkeypatch):
    tables = SimpleNamespace(
        versions=_table(rows=[SimpleNamespace(id="v1"), SimpleNamespace(id="v2")]),
        reviews=_table(count=3),
        listings=_table(count=1),
        embeddings=_table(count=2),
        raw=AsyncMock(return_value=4),
    )
    monkeypatch.setattr(marketplace, "StoreListingVersion", tables.versions)
    monkeypatch.setattr(marketplace, "StoreListingReview", tables.reviews)
    monkeypatch.setattr(marketplace, "StoreListing", tables.listings)
    monkeypatch.setattr(marketplace, "UnifiedContentEmbedding", tables.embeddings)
    monkeypatch.setattr(marketplace, "execute_raw_with_schema", tables.raw)
    return tables


def _deleted(table: MagicMock) -> list[dict]:
    return [
        c.kwargs["where"] for c in table.prisma.return_value.delete_many.await_args_list
    ]


@pytest.mark.asyncio
async def test_a_dry_run_counts_and_deletes_nothing(store):
    report = await marketplace.delete_marketplace(SUBJECT, dry_run=True)

    assert report.status == StepStatus.WOULD_DO
    assert report.counts == {
        "reviews": 3,
        "listings": 1,
        "listing_versions": 2,
        "listing_embeddings": 2,
        "shared_agent_rows": 0,
    }
    for table in (store.reviews, store.listings, store.embeddings):
        table.prisma.return_value.delete_many.assert_not_awaited()
    store.raw.assert_not_awaited()


@pytest.mark.asyncio
async def test_reviews_and_listings_of_the_account_are_deleted(store):
    report = await marketplace.delete_marketplace(SUBJECT, dry_run=False)

    assert report.status == StepStatus.DONE
    assert _deleted(store.reviews) == [{"reviewByUserId": USER_ID}]
    assert _deleted(store.listings) == [{"owningUserId": USER_ID}]
    (embeddings,) = _deleted(store.embeddings)
    assert embeddings["contentId"] == {"in": ["v1", "v2"]}
    # Other accounts' library entries stay unless removal was allowed.
    store.raw.assert_not_awaited()


@pytest.mark.asyncio
async def test_shared_agents_go_only_when_allowed(store):
    report = await marketplace.delete_marketplace(
        SUBJECT, dry_run=False, shared_agent_rows=4, remove_shared_agents=True
    )

    statements = [call.args[0] for call in store.raw.await_args_list]
    assert any('"LibraryAgent"' in sql for sql in statements)
    assert any('"AgentPreset"' in sql for sql in statements)
    # Only other accounts' rows on this account's graphs.
    assert all('"userId" <> $1' in sql for sql in statements)
    assert all(call.args[1] == USER_ID for call in store.raw.await_args_list)
    assert report.counts["shared_agent_rows"] == 4


@pytest.mark.asyncio
async def test_a_second_run_finds_nothing(monkeypatch, store):
    for table in (store.reviews, store.listings, store.embeddings):
        table.prisma.return_value.count = AsyncMock(return_value=0)
    store.versions.prisma.return_value.find_many = AsyncMock(return_value=[])

    report = await marketplace.delete_marketplace(SUBJECT, dry_run=False)

    assert set(report.counts.values()) == {0}
