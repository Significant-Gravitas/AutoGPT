from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.api.features.store import category_classifier
from backend.api.features.store.categories import StoreCategory
from backend.cli.store import _run_backfill


def _version(id: str, name: str, categories: list[str]):
    return SimpleNamespace(
        id=id,
        name=name,
        subHeading="sub heading",
        description="description",
        categories=categories,
    )


@pytest.fixture
def backfill(monkeypatch):
    """Drive _run_backfill against in-memory listings, recording every write."""
    import prisma.models

    from backend.data import db as data_db

    monkeypatch.setattr(data_db, "connect", AsyncMock())
    monkeypatch.setattr(data_db, "disconnect", AsyncMock())

    state = SimpleNamespace(versions=[], classified={}, writes=[], classify_calls=[])

    async def classify(name, sub_heading, description):
        state.classify_calls.append(name)
        return state.classified.get(name)

    monkeypatch.setattr(category_classifier, "classify_category", classify)

    client = MagicMock()
    client.find_many = AsyncMock(side_effect=lambda **_: state.versions)
    client.update = AsyncMock(
        side_effect=lambda where, data: state.writes.append(
            (where["id"], data["categories"])
        )
    )
    monkeypatch.setattr(
        prisma.models.StoreListingVersion, "prisma", MagicMock(return_value=client)
    )
    return state


async def test_a_dry_run_classifies_but_writes_nothing(backfill):
    backfill.versions = [_version("v1", "Lead Finder", [])]
    backfill.classified = {"Lead Finder": StoreCategory.SALES}

    await _run_backfill(apply=False, limit=None, concurrency=5)

    assert backfill.classify_calls == ["Lead Finder"]
    assert backfill.writes == []


async def test_apply_writes_the_classified_category(backfill):
    backfill.versions = [_version("v1", "Lead Finder", [])]
    backfill.classified = {"Lead Finder": StoreCategory.SALES}

    await _run_backfill(apply=True, limit=None, concurrency=5)

    assert backfill.writes == [("v1", ["sales"])]


async def test_a_canonical_listing_is_neither_classified_nor_written(backfill):
    backfill.versions = [_version("v1", "Already Filed", ["sales"])]

    await _run_backfill(apply=True, limit=None, concurrency=5)

    assert backfill.classify_calls == []
    assert backfill.writes == []


async def test_a_legacy_alias_is_folded_without_the_classifier(backfill):
    backfill.versions = [_version("v1", "Blog Writer", ["writing"])]

    await _run_backfill(apply=True, limit=None, concurrency=5)

    assert backfill.classify_calls == []
    assert backfill.writes == [("v1", ["content"])]


async def test_a_listing_the_classifier_declines_is_left_alone(backfill):
    backfill.versions = [_version("v1", "Mystery", [])]

    await _run_backfill(apply=True, limit=None, concurrency=5)

    assert backfill.classify_calls == ["Mystery"]
    assert backfill.writes == []


async def test_limit_caps_the_classifier_but_not_the_folds(backfill):
    backfill.versions = [
        _version("v1", "One", []),
        _version("v2", "Two", []),
        _version("v3", "Folds", ["creative"]),
    ]
    backfill.classified = {"One": StoreCategory.SALES, "Two": StoreCategory.SALES}

    await _run_backfill(apply=True, limit=1, concurrency=5)

    assert backfill.classify_calls == ["One"]
    assert backfill.writes == [("v1", ["sales"]), ("v3", ["content"])]
