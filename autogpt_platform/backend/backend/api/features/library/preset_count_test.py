"""An account holds at most `MAX_PRESETS_PER_USER` presets; real database."""

import asyncio
import uuid

import prisma.models
import pytest

from backend.api.features.library import db as library_db
from backend.api.features.library.model import LibraryAgentPresetCreatable
from backend.api.model import CreateGraph
from backend.data.user import get_or_create_user
from backend.usecases.sample import create_test_graph
from backend.util.exceptions import InvalidInputError
from backend.util.test import SpinTestServer

_CAP = 2


@pytest.mark.asyncio(loop_scope="session")
async def test_an_account_at_the_cap_saves_no_preset_until_it_deletes_one(
    server: SpinTestServer, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(library_db, "MAX_PRESETS_PER_USER", _CAP)
    user_id, graph = await _user_with_graph(server)
    other_user_id, other_graph = await _user_with_graph(server)
    saved = [await _save(user_id, graph) for _ in range(_CAP)]

    with pytest.raises(InvalidInputError, match=f"at most {_CAP} presets"):
        await _save(user_id, graph)
    assert await _held(user_id) == _CAP

    # Another account's presets are its own.
    await _save(other_user_id, other_graph)

    await library_db.delete_preset(user_id, saved[0].id)
    await _save(user_id, graph)
    assert await _held(user_id) == _CAP


@pytest.mark.asyncio(loop_scope="session")
async def test_concurrent_creates_at_the_cap_save_only_one(
    server: SpinTestServer, monkeypatch: pytest.MonkeyPatch
):
    """Two concurrent creates can't both pass the count check and both
    insert — `pg_advisory_xact_lock` serializes them, so the second sees the
    first's row before deciding."""
    monkeypatch.setattr(library_db, "MAX_PRESETS_PER_USER", _CAP)
    user_id, graph = await _user_with_graph(server)
    for _ in range(_CAP - 1):
        await _save(user_id, graph)

    results = await asyncio.gather(
        _save(user_id, graph), _save(user_id, graph), return_exceptions=True
    )

    assert sum(not isinstance(r, BaseException) for r in results) == 1
    assert sum(isinstance(r, InvalidInputError) for r in results) == 1
    assert await _held(user_id) == _CAP


async def _user_with_graph(server: SpinTestServer):
    user = await get_or_create_user(
        {
            "sub": str(uuid.uuid4()),
            "email": f"preset-count-{uuid.uuid4().hex[:8]}@example.com",
            "name": "Preset Count",
        }
    )
    graph = await server.agent_server.test_create_graph(
        CreateGraph(graph=create_test_graph()), user.id
    )
    return user.id, graph


async def _save(user_id: str, graph):
    return await library_db.create_preset(
        user_id,
        LibraryAgentPresetCreatable(
            graph_id=graph.id,
            graph_version=graph.version,
            name="Preset",
            description="",
            inputs={},
            credentials={},
            is_active=True,
        ),
    )


async def _held(user_id: str) -> int:
    return await prisma.models.AgentPreset.prisma().count(
        where={"userId": user_id, "isDeleted": False}
    )
