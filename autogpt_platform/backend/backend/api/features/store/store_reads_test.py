"""Marketplace pages read a page and its total side by side: each stand-in
below answers only once both of its reads are being awaited."""

from contextlib import AbstractContextManager
from unittest.mock import AsyncMock, MagicMock, patch

import prisma.models
import pytest

from backend.api.features.store import db as store_db
from backend.api.features.store import skill_db
from backend.util.testing import Rendezvous


def _page_and_total(
    model: type, rows: list, total: int
) -> AbstractContextManager[object]:
    """Stand in for ``model``'s page read and count, which must meet."""
    meeting = Rendezvous(2)
    client = MagicMock()
    client.find_many = meeting.returning(rows)
    client.count = meeting.returning(total)
    return patch.object(model, "prisma", return_value=client)


@pytest.mark.asyncio
async def test_the_skills_page_and_its_total_are_read_side_by_side() -> None:
    with _page_and_total(prisma.models.SkillListingVersion, [], 41):
        response = await skill_db.get_marketplace_skills(page=3, page_size=20)

    assert response.skills == []
    assert response.pagination.total_items == 41
    assert response.pagination.total_pages == 3


@pytest.mark.asyncio
async def test_the_unsearched_agents_listing_and_its_total_are_read_side_by_side() -> (
    None
):
    with _page_and_total(prisma.models.StoreAgent, [], 17):
        agents, total = await store_db._fallback_store_agent_search(
            search_query=None,
            featured=True,
            creators=None,
            category=None,
            sorted_by=None,
            page=1,
            page_size=20,
        )

    assert (agents, total) == ([], 17)


@pytest.mark.asyncio
async def test_the_creators_page_and_its_total_are_read_side_by_side() -> None:
    with _page_and_total(prisma.models.Creator, [], 45):
        response = await store_db.get_store_creators(page=2, page_size=20)

    assert response.creators == []
    assert response.pagination.total_items == 45
    assert response.pagination.total_pages == 3


@pytest.mark.asyncio
async def test_a_filtered_creators_total_counts_what_its_page_lists() -> None:
    meeting = Rendezvous(2)
    client = MagicMock()
    client.find_many = AsyncMock(side_effect=meeting.returning([]))
    client.count = AsyncMock(side_effect=meeting.returning(0))
    with patch.object(prisma.models.Creator, "prisma", return_value=client):
        await store_db.get_store_creators(featured=True, search_query="agent")

    page_where = client.find_many.call_args.kwargs["where"]
    assert client.count.call_args.kwargs["where"] == page_where
    assert page_where["is_featured"] is True
    assert "OR" in page_where


@pytest.mark.asyncio
async def test_my_unpublished_agents_and_their_total_are_read_side_by_side() -> None:
    with _page_and_total(prisma.models.LibraryAgent, [], 5):
        response = await store_db.get_my_agents("user-1", page=1, page_size=2)

    assert response.agents == []
    assert response.pagination.total_items == 5
    assert response.pagination.total_pages == 3
