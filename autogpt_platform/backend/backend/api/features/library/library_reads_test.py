"""Library reads that don't need each other's answers run side by side: each
stand-in below answers only once all of them are being awaited."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import prisma.models
import pytest

from backend.api.features.library import db as library_db
from backend.api.features.library import model as library_model
from backend.data import graph as graph_db
from backend.util.testing import Rendezvous

USER_ID = "user-1"


@pytest.mark.asyncio
async def test_a_library_page_its_total_and_schedules_are_read_side_by_side() -> None:
    meeting = Rendezvous(3)
    agents = MagicMock()
    agents.find_many = meeting.returning([])
    agents.count = meeting.returning(61)
    execution_counts = AsyncMock(return_value={})
    with (
        patch.object(prisma.models.LibraryAgent, "prisma", return_value=agents),
        patch.object(
            library_db, "_fetch_schedule_info", meeting.returning({"g1": "soon"})
        ),
        patch.object(library_db, "_fetch_execution_counts", execution_counts),
    ):
        response = await library_db.list_library_agents(USER_ID, page=2, page_size=30)

    assert response.agents == []
    assert (response.pagination.total_items, response.pagination.total_pages) == (
        61,
        3,
    )
    execution_counts.assert_awaited_once_with(USER_ID, [])


@pytest.mark.asyncio
async def test_a_library_agents_sub_graphs_and_schedule_are_read_side_by_side() -> None:
    meeting = Rendezvous(2)
    row = SimpleNamespace(agentGraphId="g1", AgentGraph=SimpleNamespace(id="g1"))
    agents = MagicMock()
    agents.find_first = AsyncMock(return_value=row)
    from_db = MagicMock(return_value="library agent")
    with (
        patch.object(prisma.models.LibraryAgent, "prisma", return_value=agents),
        patch.object(graph_db, "get_sub_graphs", meeting.returning(["sub"])),
        patch.object(
            library_db, "_fetch_schedule_info", meeting.returning({"g1": "soon"})
        ),
        patch.object(library_model.LibraryAgent, "from_db", from_db),
    ):
        agent = await library_db.get_library_agent_by_graph_id(USER_ID, "g1")

    assert agent == "library agent"
    from_db.assert_called_once_with(
        row, sub_graphs=["sub"], schedule_info={"g1": "soon"}
    )


@pytest.mark.asyncio
async def test_a_presets_page_and_its_total_are_read_side_by_side() -> None:
    meeting = Rendezvous(2)
    presets = MagicMock()
    presets.find_many = meeting.returning([])
    presets.count = meeting.returning(4)
    with patch.object(prisma.models.AgentPreset, "prisma", return_value=presets):
        response = await library_db.list_presets(USER_ID, page=1, page_size=3)

    assert response.presets == []
    assert (response.pagination.total_items, response.pagination.total_pages) == (
        4,
        2,
    )
