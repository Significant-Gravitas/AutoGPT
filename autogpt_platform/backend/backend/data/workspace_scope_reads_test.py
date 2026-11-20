"""An expert's file scope comes from three reads, and every expert file call
waits on them, so they run together rather than one after another."""

import asyncio
from unittest.mock import MagicMock

import pytest

from backend.data import workspace_scope
from backend.data.workspace_scope import WorkspaceScope


class _Barrier:
    """Lets its callers through only once all of them have started: reads
    issued one after another never get past the first."""

    def __init__(self, parties: int):
        self.parties = parties
        self.arrived = 0
        self.released = asyncio.Event()

    async def wait(self) -> None:
        self.arrived += 1
        if self.arrived == self.parties:
            self.released.set()
        await asyncio.wait_for(self.released.wait(), timeout=1)


@pytest.fixture
def reads(mocker):
    """The expert row and the two session reads, each held at a barrier."""
    barrier = _Barrier(3)
    state = MagicMock(
        expert=MagicMock(),
        own=[MagicMock(id="own-1", expertId="expert-1")],
        delegated=[
            MagicMock(id="delegated-1", expertId="expert-2"),
            MagicMock(id="own-1", expertId="expert-1"),
        ],
    )

    async def find_first(**_):
        await barrier.wait()
        return state.expert

    async def find_many(*, where):
        await barrier.wait()
        return state.own if "expertId" in where else state.delegated

    mocker.patch.object(
        workspace_scope.PrismaExpert,
        "prisma",
        return_value=MagicMock(find_first=find_first),
    )
    mocker.patch.object(
        workspace_scope.PrismaChatSession,
        "prisma",
        return_value=MagicMock(find_many=find_many),
    )
    return state


@pytest.mark.asyncio
async def test_the_three_reads_are_in_flight_together(reads):
    scope = await workspace_scope.resolve_expert_workspace_scope("user-1", "expert-1")

    assert scope.session_ids == ["own-1"]
    assert scope.delegated_session_ids == ["delegated-1"]
    assert scope.owns_skills_folder and scope.reads_user_files


@pytest.mark.asyncio
async def test_a_missing_expert_grants_nothing_whatever_the_sessions_say(reads):
    reads.expert = None

    scope = await workspace_scope.resolve_expert_workspace_scope("user-1", "expert-1")

    assert scope == WorkspaceScope(expert_id="expert-1")
