"""Listings and reviews reference the account with ON DELETE RESTRICT, so the
marketplace step removes them (and the listings' search embeddings) before the
account's rows go. Listings handed to the "deleted user" account stay. Skill
listings always go: left to the account row, they would stay with no owner,
which reads as written by the platform."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.data.account_erasure import marketplace
from backend.data.account_erasure.models import StepStatus, Subject

USER_ID = "3f0b8c1e-4c2a-4f7e-9a51-0c6f2d8e1a77"
SUBJECT = Subject(user_id=USER_ID)
OWN_LISTINGS = {"owningUserId": USER_ID, "id": {"not_in": []}}


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
        skills=_table(count=2),
        skill_versions=_table(count=3),
        skill_files=_table(count=4),
    )
    monkeypatch.setattr(marketplace, "StoreListingVersion", tables.versions)
    monkeypatch.setattr(marketplace, "StoreListingReview", tables.reviews)
    monkeypatch.setattr(marketplace, "StoreListing", tables.listings)
    monkeypatch.setattr(marketplace, "UnifiedContentEmbedding", tables.embeddings)
    monkeypatch.setattr(marketplace, "SkillListing", tables.skills)
    monkeypatch.setattr(marketplace, "SkillListingVersion", tables.skill_versions)
    monkeypatch.setattr(marketplace, "SkillListingFile", tables.skill_files)
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
        "skill_listings": 2,
        "skill_listing_versions": 3,
        "skill_listing_files": 4,
    }
    for table in (store.reviews, store.listings, store.embeddings, store.skills):
        table.prisma.return_value.delete_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_reviews_and_listings_of_the_account_are_deleted(store):
    report = await marketplace.delete_marketplace(SUBJECT, dry_run=False)

    assert report.status == StepStatus.DONE
    assert _deleted(store.reviews) == [{"reviewByUserId": USER_ID}]
    assert _deleted(store.listings) == [OWN_LISTINGS]
    (embeddings,) = _deleted(store.embeddings)
    assert embeddings["contentId"] == {"in": ["v1", "v2"]}


@pytest.mark.asyncio
async def test_every_skill_listing_of_the_account_is_deleted(store):
    """Even with agent listings handed over: installs are copies, so nobody
    needs the listing, and its versions and files go with it (cascade)."""
    kept = SUBJECT.model_copy(update={"kept_listing_ids": ["listing-kept"]})

    report = await marketplace.delete_marketplace(kept, dry_run=False)

    assert _deleted(store.skills) == [{"owningUserId": USER_ID}]
    assert report.counts["skill_listing_files"] == 4
    files = store.skill_files.prisma.return_value.count.await_args
    assert files.kwargs["where"] == {
        "SkillListingVersion": {
            "is": {"SkillListing": {"is": {"owningUserId": USER_ID}}}
        }
    }


@pytest.mark.asyncio
async def test_listings_handed_to_the_placeholder_are_kept(store):
    kept = SUBJECT.model_copy(update={"kept_listing_ids": ["listing-kept"]})

    await marketplace.delete_marketplace(kept, dry_run=False)

    (listings,) = _deleted(store.listings)
    assert listings == {"owningUserId": USER_ID, "id": {"not_in": ["listing-kept"]}}
    versions = store.versions.prisma.return_value.find_many.await_args
    assert versions.kwargs["where"] == {"StoreListing": {"is": listings}}


@pytest.mark.asyncio
async def test_a_second_run_finds_nothing(monkeypatch, store):
    for table in (
        store.reviews,
        store.listings,
        store.embeddings,
        store.skills,
        store.skill_versions,
        store.skill_files,
    ):
        table.prisma.return_value.count = AsyncMock(return_value=0)
    store.versions.prisma.return_value.find_many = AsyncMock(return_value=[])

    report = await marketplace.delete_marketplace(SUBJECT, dry_run=False)

    assert set(report.counts.values()) == {0}
