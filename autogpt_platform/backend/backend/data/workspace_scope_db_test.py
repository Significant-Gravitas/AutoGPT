"""Real-database coverage for the expert scope resolver: own and delegated
conversations, the JSON metadata filter, and two-expert file isolation that
survives new conversations."""

import uuid

import prisma.models
import pytest

from backend.copilot.db import create_chat_session
from backend.copilot.model import ChatSessionMetadata
from backend.data.user import get_or_create_user
from backend.data.workspace import (
    count_workspace_files,
    create_workspace_file,
    get_or_create_workspace,
    list_workspace_files,
)
from backend.data.workspace_scope import resolve_expert_workspace_scope
from backend.util.test import SpinTestServer


async def _user() -> str:
    suffix = uuid.uuid4().hex[:8]
    user = await get_or_create_user(
        {
            "sub": str(uuid.uuid4()),
            "email": f"scope-{suffix}@example.com",
            "name": "Scope Owner",
        }
    )
    return user.id


async def _expert(user_id: str) -> str:
    row = await prisma.models.Expert.prisma().create(
        data={
            "ownerUserId": user_id,
            "name": "Nova",
            "role": "",
            "identity": "nova",
            "skills": [],
        }
    )
    return row.id


@pytest.mark.asyncio(loop_scope="session")
async def test_resolver_collects_own_and_delegated_conversations(
    server: SpinTestServer,
):
    owner = await _user()
    expert_a = await _expert(owner)
    expert_b = await _expert(owner)
    own = await create_chat_session(str(uuid.uuid4()), owner, expert_id=expert_a)
    foreign = await create_chat_session(str(uuid.uuid4()), owner, expert_id=expert_b)
    personal = await create_chat_session(str(uuid.uuid4()), owner)
    delegated = await create_chat_session(
        str(uuid.uuid4()),
        owner,
        expert_id=expert_b,
        metadata=ChatSessionMetadata(
            delegated_by_expert_id=expert_a, delegated_by_session_id=own.session_id
        ),
    )

    scope = await resolve_expert_workspace_scope(owner, expert_a)

    assert scope.session_ids == [own.session_id]
    assert scope.delegated_session_ids == [delegated.session_id]
    assert scope.allows_path(f"/sessions/{own.session_id}/x.txt", write=True)
    assert scope.allows_path(f"/sessions/{delegated.session_id}/x.txt")
    assert not scope.allows_path(f"/sessions/{delegated.session_id}/x.txt", write=True)
    assert not scope.allows_path(f"/sessions/{foreign.session_id}/x.txt")
    assert not scope.allows_path(f"/sessions/{personal.session_id}/x.txt")
    assert scope.allows_path("/skills/autopilot/SKILL.md")
    assert not scope.allows_path("/skills/autopilot/SKILL.md", write=True)


@pytest.mark.asyncio(loop_scope="session")
async def test_resolver_fails_closed_for_foreign_or_archived_expert(
    server: SpinTestServer,
):
    owner = await _user()
    stranger = await _user()
    expert = await _expert(stranger)
    session = await create_chat_session(str(uuid.uuid4()), stranger, expert_id=expert)

    foreign_scope = await resolve_expert_workspace_scope(owner, expert)
    assert foreign_scope.session_ids == []
    assert foreign_scope.delegated_session_ids == []
    assert not foreign_scope.allows_path(f"/sessions/{session.session_id}/x.txt")

    await prisma.models.Expert.prisma().update(
        where={"id": expert}, data={"isArchived": True}
    )
    archived_scope = await resolve_expert_workspace_scope(stranger, expert)
    assert archived_scope.session_ids == []
    assert not archived_scope.allows_path(f"/sessions/{session.session_id}/x.txt")


@pytest.mark.asyncio(loop_scope="session")
async def test_two_experts_files_stay_separate_across_new_conversations(
    server: SpinTestServer,
):
    owner = await _user()
    expert_a = await _expert(owner)
    expert_b = await _expert(owner)
    a_first = await create_chat_session(str(uuid.uuid4()), owner, expert_id=expert_a)
    workspace = await get_or_create_workspace(owner)
    stored = await create_workspace_file(
        workspace_id=workspace.id,
        file_id=str(uuid.uuid4()),
        name="plan.md",
        path=f"/sessions/{a_first.session_id}/plan.md",
        storage_path=f"{workspace.id}/plan.md",
        mime_type="text/markdown",
        size_bytes=4,
    )

    a_new = await create_chat_session(str(uuid.uuid4()), owner, expert_id=expert_a)
    b_new = await create_chat_session(str(uuid.uuid4()), owner, expert_id=expert_b)
    scope_a = await resolve_expert_workspace_scope(owner, expert_a)
    scope_a = scope_a.with_session(a_new.session_id)
    scope_b = await resolve_expert_workspace_scope(owner, expert_b)
    scope_b = scope_b.with_session(b_new.session_id)

    assert scope_a.allows_path(stored.path)
    assert not scope_b.allows_path(stored.path)

    listed_for_a = await list_workspace_files(
        workspace.id, allowed_path_prefixes=scope_a.read_prefixes
    )
    listed_for_b = await list_workspace_files(
        workspace.id, allowed_path_prefixes=scope_b.read_prefixes
    )
    assert [f.id for f in listed_for_a] == [stored.id]
    assert listed_for_b == []
    assert (
        await count_workspace_files(
            workspace.id, allowed_path_prefixes=scope_b.read_prefixes
        )
        == 0
    )
