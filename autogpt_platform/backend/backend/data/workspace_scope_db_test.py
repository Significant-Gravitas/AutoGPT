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
    assert not scope.allows_path("/skills/autopilot/SKILL.md")
    assert scope.allows_path(f"/experts/{expert_a}/skills/mine/SKILL.md", write=True)
    assert not scope.allows_path(f"/experts/{expert_b}/skills/theirs/SKILL.md")


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
    # A revoked expert keeps no claim on the skills folder named after it.
    for revoked in (foreign_scope, archived_scope):
        assert not revoked.allows_path(f"/experts/{expert}/skills/x/SKILL.md")


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


@pytest.mark.asyncio(loop_scope="session")
async def test_every_expert_reads_the_owners_own_files_and_no_more(
    server: SpinTestServer,
):
    """The user's uploads live outside every managed root, so before the grant
    an expert chat could not read or attach what the user had just uploaded."""
    owner = await _user()
    expert_a = await _expert(owner)
    expert_b = await _expert(owner)
    a_session = await create_chat_session(str(uuid.uuid4()), owner, expert_id=expert_a)
    b_session = await create_chat_session(str(uuid.uuid4()), owner, expert_id=expert_b)
    workspace = await get_or_create_workspace(owner)

    upload = await create_workspace_file(
        workspace_id=workspace.id,
        file_id=str(uuid.uuid4()),
        name="quarterly-report.pdf",
        path=f"/quarterly-report-{uuid.uuid4().hex[:8]}.pdf",
        storage_path=f"{workspace.id}/quarterly.pdf",
        mime_type="application/pdf",
        size_bytes=9,
    )
    a_private = await create_workspace_file(
        workspace_id=workspace.id,
        file_id=str(uuid.uuid4()),
        name="notes.md",
        path=f"/sessions/{a_session.session_id}/notes.md",
        storage_path=f"{workspace.id}/notes.md",
        mime_type="text/markdown",
        size_bytes=4,
    )

    scope_a = await resolve_expert_workspace_scope(owner, expert_a)
    scope_b = await resolve_expert_workspace_scope(owner, expert_b)

    assert scope_a.reads_user_files and scope_b.reads_user_files
    assert scope_a.allows_path(upload.path)
    assert scope_b.allows_path(upload.path)
    assert not scope_a.allows_path(upload.path, write=True)
    # Expert-to-expert isolation is untouched by the grant.
    assert scope_a.allows_path(a_private.path)
    assert not scope_b.allows_path(a_private.path)

    listed_for_b = await list_workspace_files(
        workspace.id,
        allowed_path_prefixes=[
            f"/sessions/{sid}/" for sid in [*scope_b.session_ids, b_session.session_id]
        ],
        include_user_files=scope_b.reads_user_files,
    )
    assert upload.id in [f.id for f in listed_for_b]
    assert a_private.id not in [f.id for f in listed_for_b]
    assert (
        await count_workspace_files(
            workspace.id,
            allowed_path_prefixes=[f"/sessions/{b_session.session_id}/"],
            include_user_files=True,
        )
        >= 1
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_a_revoked_expert_reads_no_user_files(server: SpinTestServer):
    owner = await _user()
    stranger = await _user()
    expert = await _expert(stranger)
    workspace = await get_or_create_workspace(owner)
    upload = await create_workspace_file(
        workspace_id=workspace.id,
        file_id=str(uuid.uuid4()),
        name="private.pdf",
        path=f"/private-{uuid.uuid4().hex[:8]}.pdf",
        storage_path=f"{workspace.id}/private.pdf",
        mime_type="application/pdf",
        size_bytes=3,
    )

    foreign_scope = await resolve_expert_workspace_scope(owner, expert)

    assert not foreign_scope.reads_user_files
    assert not foreign_scope.allows_path(upload.path)
    assert (
        await list_workspace_files(
            workspace.id,
            allowed_path_prefixes=[],
            include_user_files=foreign_scope.reads_user_files,
        )
        == []
    )
