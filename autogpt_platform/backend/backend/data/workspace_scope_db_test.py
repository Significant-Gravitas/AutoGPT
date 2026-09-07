"""Real-database coverage for the expert scope resolver, including the JSON
metadata filter that grants read access to delegated sub-sessions."""

import uuid

import prisma.models
import pytest

from backend.copilot.db import create_chat_session
from backend.copilot.model import ChatSessionMetadata
from backend.data.user import get_or_create_user
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


async def _expert(user_id: str, skills: list[str]) -> str:
    row = await prisma.models.Expert.prisma().create(
        data={
            "ownerUserId": user_id,
            "name": "Nova",
            "role": "",
            "identity": "nova",
            "skills": skills,
        }
    )
    return row.id


@pytest.mark.asyncio(loop_scope="session")
async def test_resolver_collects_own_delegated_and_skill_grants(
    server: SpinTestServer,
):
    owner = await _user()
    expert_a = await _expert(owner, ["Assigned"])
    expert_b = await _expert(owner, ["other"])
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
    assert scope.allows_path(f"/experts/{expert_a}/skills/mine/SKILL.md", write=True)
    assert not scope.allows_path(f"/experts/{expert_b}/skills/theirs/SKILL.md")
    assert not scope.allows_path("/skills/autopilot/SKILL.md")
    assert not scope.allows_path(f"/sessions/{foreign.session_id}/x.txt")
    assert not scope.allows_path(f"/sessions/{personal.session_id}/x.txt")
    assert scope.allows_path(f"/sessions/{delegated.session_id}/x.txt")
    assert not scope.allows_path(f"/sessions/{delegated.session_id}/x.txt", write=True)


@pytest.mark.asyncio(loop_scope="session")
async def test_resolver_fails_closed_for_foreign_or_archived_expert(
    server: SpinTestServer,
):
    owner = await _user()
    stranger = await _user()
    expert = await _expert(stranger, ["assigned"])
    await create_chat_session(str(uuid.uuid4()), stranger, expert_id=expert)

    foreign_scope = await resolve_expert_workspace_scope(owner, expert)
    assert foreign_scope.read_prefixes == []
    assert not foreign_scope.allows_path(f"/experts/{expert}/skills/x/SKILL.md")

    await prisma.models.Expert.prisma().update(
        where={"id": expert}, data={"isArchived": True}
    )
    archived_scope = await resolve_expert_workspace_scope(stranger, expert)
    assert archived_scope.read_prefixes == []
