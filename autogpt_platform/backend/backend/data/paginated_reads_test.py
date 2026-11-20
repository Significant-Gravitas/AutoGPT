"""Paginated lists read a page and its total side by side: each stand-in below
answers only once both of its reads are being awaited."""

from contextlib import AbstractContextManager
from unittest.mock import AsyncMock, MagicMock, patch

import prisma.models
import pytest

from backend.data import execution as execution_db
from backend.data import graph as graph_db
from backend.data import human_review
from backend.util.testing import Rendezvous


def _page_and_total(model: type, total: int) -> AbstractContextManager[object]:
    """Stand in for ``model``'s page read and count, which must meet."""
    meeting = Rendezvous(2)
    client = MagicMock()
    client.find_many = meeting.returning([])
    client.count = meeting.returning(total)
    return patch.object(model, "prisma", return_value=client)


@pytest.mark.asyncio
async def test_a_runs_page_and_its_total_are_read_side_by_side() -> None:
    with _page_and_total(prisma.models.AgentGraphExecution, 51):
        result = await execution_db.get_graph_executions_paginated(
            "user-1", graph_id="graph-1", page=2, page_size=25
        )

    assert result.executions == []
    assert (result.pagination.total_items, result.pagination.total_pages) == (51, 3)


@pytest.mark.asyncio
async def test_a_graphs_page_and_its_total_are_read_side_by_side() -> None:
    with _page_and_total(prisma.models.AgentGraph, 26):
        graphs, pagination = await graph_db.list_graphs_paginated(
            "user-1", page=1, page_size=25
        )

    assert graphs == []
    assert (pagination.total_items, pagination.total_pages) == (26, 2)


@pytest.mark.asyncio
async def test_a_reviews_page_and_its_total_are_read_side_by_side() -> None:
    with (
        _page_and_total(prisma.models.PendingHumanReview, 3),
        patch.object(
            human_review,
            "_enrich_pending_reviews",
            AsyncMock(side_effect=lambda user_id, reviews: reviews),
        ),
    ):
        reviews, pagination = await human_review.get_reviews(
            "user-1", page=1, page_size=2
        )

    assert reviews == []
    assert (pagination.total_items, pagination.total_pages) == (3, 2)
