"""Library reads that don't need each other's answers run side by side: each
stand-in below answers only once all of them are being awaited."""

import asyncio
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
async def test_a_library_page_counts_its_runs_while_its_schedules_are_read() -> None:
    """The page, its total and the schedules meet; the run counts then start
    from the page alone, so a scheduler still answering doesn't hold them."""
    meeting = Rendezvous(3)
    counts_started = asyncio.Event()

    async def read_schedules(*_args: object, **_kwargs: object) -> dict[str, str]:
        await meeting.arrive()
        async with asyncio.timeout(10):
            await counts_started.wait()
        return {"g1": "soon"}

    async def count_runs(*_args: object, **_kwargs: object) -> dict[str, int]:
        counts_started.set()
        return {}

    agents = MagicMock()
    agents.find_many = meeting.returning([])
    agents.count = meeting.returning(61)
    execution_counts = AsyncMock(side_effect=count_runs)
    with (
        patch.object(prisma.models.LibraryAgent, "prisma", return_value=agents),
        patch.object(library_db, "_fetch_schedule_info", read_schedules),
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
async def test_a_library_page_gives_each_entry_its_workflows_run_count() -> None:
    row = SimpleNamespace(id="ua1", agentGraphId="g1")
    agents = MagicMock()
    agents.find_many = AsyncMock(return_value=[row])
    agents.count = AsyncMock(return_value=1)
    execution_counts = AsyncMock(return_value={"g1": 7})
    entry = library_model.LibraryAgent.model_construct(id="ua1")
    from_db = MagicMock(return_value=entry)
    with (
        patch.object(prisma.models.LibraryAgent, "prisma", return_value=agents),
        patch.object(library_db, "_fetch_schedule_info", AsyncMock(return_value={})),
        patch.object(library_db, "_fetch_execution_counts", execution_counts),
        patch.object(library_model.LibraryAgent, "from_db", from_db),
    ):
        response = await library_db.list_library_agents(USER_ID)

    assert response.agents == [entry]
    execution_counts.assert_awaited_once_with(USER_ID, ["g1"])
    from_db.assert_called_once_with(row, execution_count_override=7, schedule_info={})


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
