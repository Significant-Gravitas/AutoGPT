import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import prisma.models
import pytest

from backend.api.features.library import db as library_db
from backend.copilot.tools._test_data import _ensure_db_connected
from backend.copilot.tools.utils import get_or_create_library_agent
from backend.data.graph import Graph, GraphSettings, create_graph
from backend.data.user import get_or_create_user
from backend.util.json import SafeJson


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("state", ["isArchived", "isDeleted"])
async def test_reusing_a_set_aside_entry_keeps_its_settings(state: str):
    """On Postgres: running an agent whose entry is archived or removed reuses
    that entry with the settings its user chose, and leaves an archived one
    archived."""
    await _ensure_db_connected()
    user = await get_or_create_user(
        {"sub": f"test-user-{uuid.uuid4()}", "email": f"{uuid.uuid4()}@example.com"}
    )
    graph = await create_graph(
        Graph(id=str(uuid.uuid4()), version=1, name="t", description=""), user.id
    )

    try:
        with (
            patch.object(library_db, "add_generated_agent_image", AsyncMock()),
            patch.object(library_db, "schedule_library_agent_embedding", MagicMock()),
            patch.object(
                library_db, "_fetch_schedule_info", AsyncMock(return_value={})
            ),
        ):
            [entry] = await library_db.create_library_agent(
                graph, user.id, create_library_agents_for_sub_graphs=False
            )
            chosen = GraphSettings(builder_chat_session_id=str(uuid.uuid4()))
            await prisma.models.LibraryAgent.prisma().update(
                where={"id": entry.id},
                data={state: True, "settings": SafeJson(chosen.model_dump())},
            )
            await get_or_create_library_agent(graph, user.id)

        row = await prisma.models.LibraryAgent.prisma().find_unique_or_raise(
            where={"id": entry.id}
        )
    finally:
        await prisma.models.LibraryAgent.prisma().delete_many(
            where={"agentGraphId": graph.id}
        )
        await prisma.models.AgentGraph.prisma().delete_many(where={"id": graph.id})

    assert GraphSettings.model_validate(row.settings) == chosen
    assert row.isArchived is (state == "isArchived")
