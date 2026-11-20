"""Real-database coverage for moving library agents between folders.

``bulk_move_agents_to_folder`` resolves every requested ID before it writes:
a library agent ID or a graph ID moves the agent, and any ID it can't
resolve fails the whole call with ``NotFoundError`` so nothing moves.
"""

import uuid
from contextlib import asynccontextmanager

import pytest
from prisma.models import AgentGraph, LibraryAgent, LibraryFolder, User

from backend.data.db import transaction
from backend.data.user import get_or_create_user
from backend.util.exceptions import NotFoundError
from backend.util.test import SpinTestServer

from .db import bulk_move_agents_to_folder

_seeded_user_ids: list[str] = []


@pytest.fixture(autouse=True)
async def _clean_seeded_rows():
    """Delete what each test made; every row belongs to a user from ``_user``."""
    yield
    user_ids = list(_seeded_user_ids)
    _seeded_user_ids.clear()
    if not user_ids:
        return
    # An entry blocks the delete of its folder and graph, so entries go first.
    await LibraryAgent.prisma().delete_many(where={"userId": {"in": user_ids}})
    await LibraryFolder.prisma().delete_many(where={"userId": {"in": user_ids}})
    await AgentGraph.prisma().delete_many(where={"userId": {"in": user_ids}})
    await User.prisma().delete_many(where={"id": {"in": user_ids}})


async def _user() -> str:
    user = await get_or_create_user(
        {
            "sub": str(uuid.uuid4()),
            "email": f"lib-folders-{uuid.uuid4().hex[:8]}@example.com",
            "name": "Folder Owner",
        }
    )
    _seeded_user_ids.append(user.id)
    return user.id


async def _folder(user_id: str) -> str:
    folder = await LibraryFolder.prisma().create(
        data={"userId": user_id, "name": f"Folder {uuid.uuid4().hex[:8]}"}
    )
    return folder.id


async def _agent(user_id: str, **flags: bool) -> LibraryAgent:
    graph = await AgentGraph.prisma().create(
        data={
            "id": str(uuid.uuid4()),
            "version": 1,
            "name": "Folder Move Test",
            "userId": user_id,
            "isActive": True,
        }
    )
    return await LibraryAgent.prisma().create(
        data={
            "userId": user_id,
            "agentGraphId": graph.id,
            "agentGraphVersion": graph.version,
            "isCreatedByUser": True,
            **flags,
        }
    )


async def _entries_of_one_graph(
    user_id: str, organizations: list[str | None]
) -> list[LibraryAgent]:
    """One library entry per version of a new graph, tagged in turn."""
    graph_id = str(uuid.uuid4())
    entries = []
    for version, organization_id in enumerate(organizations, start=1):
        await AgentGraph.prisma().create(
            data={
                "id": graph_id,
                "version": version,
                "name": "Folder Move Test",
                "userId": user_id,
            }
        )
        entries.append(
            await LibraryAgent.prisma().create(
                data={
                    "userId": user_id,
                    "agentGraphId": graph_id,
                    "agentGraphVersion": version,
                    "organizationId": organization_id,
                }
            )
        )
    return entries


async def _folder_of(library_agent_id: str) -> str | None:
    row = await LibraryAgent.prisma().find_unique(where={"id": library_agent_id})
    assert row is not None
    return row.folderId


@pytest.mark.asyncio(loop_scope="session")
async def test_library_agent_id_moves_the_agent(server: SpinTestServer):
    user = await _user()
    folder = await _folder(user)
    agent = await _agent(user)

    moved = await bulk_move_agents_to_folder([agent.id], folder, user)

    assert [a.id for a in moved] == [agent.id]
    assert await _folder_of(agent.id) == folder


@pytest.mark.asyncio(loop_scope="session")
async def test_graph_id_moves_its_library_agent(server: SpinTestServer):
    user = await _user()
    folder = await _folder(user)
    agent = await _agent(user)

    moved = await bulk_move_agents_to_folder([agent.agentGraphId], folder, user)

    assert [a.id for a in moved] == [agent.id]
    assert await _folder_of(agent.id) == folder


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("by", ["library_id", "graph_id"])
async def test_archived_agent_can_be_moved(server: SpinTestServer, by: str):
    user = await _user()
    folder = await _folder(user)
    agent = await _agent(user, isArchived=True)
    requested = agent.id if by == "library_id" else agent.agentGraphId

    moved = await bulk_move_agents_to_folder([requested], folder, user)

    assert [a.id for a in moved] == [agent.id]
    assert await _folder_of(agent.id) == folder


@pytest.mark.asyncio(loop_scope="session")
async def test_organization_moves_only_its_own_and_untagged_entries(
    server: SpinTestServer,
):
    user = await _user()
    folder = await _folder(user)
    own, other, untagged = await _entries_of_one_graph(user, ["org-a", "org-b", None])

    moved = await bulk_move_agents_to_folder(
        [own.agentGraphId], folder, user, organization_id="org-a"
    )

    assert sorted(a.id for a in moved) == sorted([own.id, untagged.id])
    assert await _folder_of(other.id) is None
    with pytest.raises(NotFoundError, match=other.id):
        await bulk_move_agents_to_folder(
            [other.id], folder, user, organization_id="org-a"
        )
    assert await _folder_of(other.id) is None


@pytest.mark.asyncio(loop_scope="session")
async def test_entry_re_tagged_after_the_lookup_is_not_moved(
    server: SpinTestServer, monkeypatch: pytest.MonkeyPatch
):
    user = await _user()
    folder = await _folder(user)
    (entry,) = await _entries_of_one_graph(user, ["org-a"])

    @asynccontextmanager
    async def re_tag_then_transaction():
        # Another request moves the entry to org-b after the move's lookup.
        await LibraryAgent.prisma().update(
            where={"id": entry.id}, data={"organizationId": "org-b"}
        )
        async with transaction() as tx:
            yield tx

    monkeypatch.setattr(
        "backend.api.features.library.db.transaction", re_tag_then_transaction
    )
    moved = await bulk_move_agents_to_folder(
        [entry.id], folder, user, organization_id="org-a"
    )

    assert moved == []
    assert await _folder_of(entry.id) is None


@pytest.mark.asyncio(loop_scope="session")
async def test_unknown_id_is_not_found(server: SpinTestServer):
    user = await _user()
    folder = await _folder(user)
    unknown = str(uuid.uuid4())

    with pytest.raises(NotFoundError, match=unknown):
        await bulk_move_agents_to_folder([unknown], folder, user)


@pytest.mark.asyncio(loop_scope="session")
async def test_id_not_owned_by_user_is_not_found(server: SpinTestServer):
    user = await _user()
    folder = await _folder(user)
    other_agent = await _agent(await _user())

    with pytest.raises(NotFoundError, match=other_agent.id):
        await bulk_move_agents_to_folder([other_agent.id], folder, user)
    assert await _folder_of(other_agent.id) is None


@pytest.mark.asyncio(loop_scope="session")
async def test_deleted_agent_is_not_found(server: SpinTestServer):
    user = await _user()
    folder = await _folder(user)
    agent = await _agent(user, isDeleted=True)

    with pytest.raises(NotFoundError, match=agent.id):
        await bulk_move_agents_to_folder([agent.id], folder, user)
    assert await _folder_of(agent.id) is None


@pytest.mark.asyncio(loop_scope="session")
async def test_mixed_valid_and_unknown_ids_move_nothing(server: SpinTestServer):
    user = await _user()
    folder = await _folder(user)
    agent = await _agent(user)
    unknown = str(uuid.uuid4())

    with pytest.raises(NotFoundError, match=unknown) as raised:
        await bulk_move_agents_to_folder([agent.id, unknown], folder, user)
    assert agent.id not in str(raised.value)
    assert await _folder_of(agent.id) is None


@pytest.mark.asyncio(loop_scope="session")
async def test_repeated_move_reports_no_moves(server: SpinTestServer):
    user = await _user()
    folder = await _folder(user)
    agent = await _agent(user)

    first = await bulk_move_agents_to_folder([agent.id], folder, user)
    assert [a.id for a in first] == [agent.id]
    assert await _folder_of(agent.id) == folder

    second = await bulk_move_agents_to_folder([agent.id], folder, user)
    assert second == []
    assert await _folder_of(agent.id) == folder


@pytest.mark.asyncio(loop_scope="session")
async def test_batch_reports_only_the_agents_it_moved(server: SpinTestServer):
    user = await _user()
    folder = await _folder(user)
    already_there = await _agent(user)
    await LibraryAgent.prisma().update_many(
        where={"id": already_there.id}, data={"folderId": folder}
    )
    to_move = await _agent(user)

    moved = await bulk_move_agents_to_folder(
        [already_there.id, to_move.id], folder, user
    )

    assert [a.id for a in moved] == [to_move.id]
    assert await _folder_of(already_there.id) == folder
    assert await _folder_of(to_move.id) == folder


@pytest.mark.asyncio(loop_scope="session")
async def test_agent_named_twice_moves_once(server: SpinTestServer):
    user = await _user()
    folder = await _folder(user)
    agent = await _agent(user)

    moved = await bulk_move_agents_to_folder(
        [agent.id, agent.agentGraphId], folder, user
    )

    assert [a.id for a in moved] == [agent.id]
    assert await _folder_of(agent.id) == folder
