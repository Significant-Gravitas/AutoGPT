"""Real-database coverage for SECRT-3159: hiring an expert installs its skill
packages in the user's workspace, and Files listed every SKILL.md, LICENSE and
reference beside what the expert made for the user.

The skills are written by the writer that installs them, the deliverable by
the tool an expert writes it with, and both are read back through the routes
the Files page calls.
"""

import uuid
from pathlib import Path
from unittest.mock import patch

import prisma
import pytest

from backend.api.features.workspace.routes import (
    get_storage_usage,
    list_workspace_files,
)
from backend.copilot.tools._test_data import make_session
from backend.copilot.tools.workspace_files import WriteWorkspaceFileTool
from backend.data.user import get_or_create_user
from backend.data.workspace import create_workspace_file, get_or_create_workspace
from backend.util.test import SpinTestServer
from backend.util.workspace import SkillPackageFile, WorkspaceManager

_MIGRATION = (
    Path(__file__).parents[4]
    / "migrations/20261023130000_workspace_skill_files_internal/migration.sql"
)

_SKILL_MD = b"---\nname: copywriting\ndescription: Write launch copy\n---\n"
_LICENSE = b"MIT License"
_SENDGRID = b"# SendGrid\nSend the launch email."
_OWN_SKILL = b"---\nname: launch-notes\n---\n"
_BRIEF = "# Dewleaf launch brief"


@pytest.mark.asyncio(loop_scope="session")
async def test_files_lists_the_deliverable_not_the_skills_hiring_installed(
    server: SpinTestServer,
):
    user_id = await _user()
    workspace = await get_or_create_workspace(user_id)
    manager = WorkspaceManager(user_id, workspace.id)
    folder = f"/experts/{uuid.uuid4()}/skills/copywriting"

    with patch(
        "backend.api.features.workspace.embeddings.schedule_workspace_file_embedding"
    ) as indexed:
        # The package hiring installs for the expert, as one change.
        await manager.write_skill_package(
            root=SkillPackageFile(
                path=f"{folder}/SKILL.md",
                content=_SKILL_MD,
                mime_type="text/markdown",
                metadata={"kind": "copilot_skill", "skill_origin": "marketplace"},
            ),
            files=[
                SkillPackageFile(path=f"{folder}/LICENSE", content=_LICENSE),
                SkillPackageFile(
                    path=f"{folder}/references/sendgrid.md", content=_SENDGRID
                ),
            ],
            removed_paths=[],
        )
        # One of the account's own skills, saved a file at a time.
        await manager.write_file(
            _OWN_SKILL, "SKILL.md", path="/skills/launch-notes/SKILL.md"
        )
        # What the expert made for the user.
        session = make_session(user_id)
        await WriteWorkspaceFileTool()._execute(
            user_id=user_id,
            session=session,
            filename="dewleaf-launch-brief.md",
            content=_BRIEF,
        )

    # Files: the deliverable, and nothing the skills registry wrote.
    assert [f.name for f in await _list_files(user_id)] == ["dewleaf-launch-brief.md"]
    assert [f.name for f in await _list_files(user_id, origin="generated")] == [
        "dewleaf-launch-brief.md"
    ]
    # Global search finds the deliverable only.
    assert [call.kwargs["name"] for call in indexed.call_args_list] == [
        "dewleaf-launch-brief.md"
    ]

    # The skills are still there for whoever asks for them.
    everything = await _list_files(user_id, include_internal=True)
    assert sorted(f.path for f in everything if f.is_internal) == sorted(
        [
            f"{folder}/SKILL.md",
            f"{folder}/LICENSE",
            f"{folder}/references/sendgrid.md",
            "/skills/launch-notes/SKILL.md",
        ]
    )
    assert [f.name for f in everything if not f.is_internal] == [
        "dewleaf-launch-brief.md"
    ]

    # Storage: the skills' bytes are named apart from the user's files.
    skills_bytes = len(_SKILL_MD) + len(_LICENSE) + len(_SENDGRID) + len(_OWN_SKILL)
    usage = await get_storage_usage(user_id=user_id)
    assert usage.file_count == 1
    assert usage.internal_bytes == skills_bytes
    assert usage.used_bytes == skills_bytes + len(_BRIEF)

    # The expert still reads its skills.
    listed = await manager.list_files(path=f"{folder}/")
    assert sorted(f.name for f in listed) == ["LICENSE", "SKILL.md", "sendgrid.md"]
    assert await manager.read_file(f"{folder}/references/sendgrid.md") == _SENDGRID


@pytest.mark.asyncio(loop_scope="session")
async def test_the_migration_marks_skill_files_written_before_it(
    server: SpinTestServer,
):
    user_id = await _user()
    workspace = await get_or_create_workspace(user_id)
    legacy = {
        "/experts/e1/skills/copywriting/SKILL.md": {"skill_origin": "marketplace"},
        "/experts/e1/skills/copywriting/LICENSE": {},
        "/experts/e1/skills/email/references/sendgrid.md": {},
        "/skills/launch-notes/SKILL.md": {"skill_origin": "user"},
        "/sessions/s1/dewleaf-launch-brief.md": {"origin": "agent-created"},
        "/my-skills/copywriting/SKILL.md": {},
        "/experts/e1/notes/skills/plan.md": {},
        "/sessions/s1/skills/SKILL.md": {},
    }
    ids = {}
    for path, metadata in legacy.items():
        ids[path] = str(uuid.uuid4())
        await create_workspace_file(
            workspace_id=workspace.id,
            file_id=ids[path],
            name=path.rsplit("/", 1)[1],
            path=path,
            storage_path=f"{workspace.id}{path}",
            mime_type="text/markdown",
            size_bytes=2,
            metadata=metadata,
        )
    # Both were indexed for search before skill files were marked.
    await _embed(user_id, ids["/experts/e1/skills/copywriting/SKILL.md"])
    await _embed(user_id, ids["/sessions/s1/dewleaf-launch-brief.md"])

    await _run_backfill(workspace.id)

    rows = await prisma.models.UserWorkspaceFile.prisma().find_many(
        where={"workspaceId": workspace.id}
    )
    assert {row.path for row in rows if row.isInternal} == {
        "/experts/e1/skills/copywriting/SKILL.md",
        "/experts/e1/skills/copywriting/LICENSE",
        "/experts/e1/skills/email/references/sendgrid.md",
        "/skills/launch-notes/SKILL.md",
    }
    # The skill's search entry is gone; the brief's stays.
    embedded = await prisma.models.UnifiedContentEmbedding.prisma().find_many(
        where={"contentType": "WORKSPACE_FILE", "userId": user_id}
    )
    assert {row.contentId for row in embedded} == {
        ids["/sessions/s1/dewleaf-launch-brief.md"]
    }


async def _user() -> str:
    suffix = uuid.uuid4().hex[:8]
    user = await get_or_create_user(
        {
            "sub": str(uuid.uuid4()),
            "email": f"skill-files-{suffix}@example.com",
            "name": "Maya",
        }
    )
    return user.id


async def _list_files(user_id: str, **filters):
    params = dict(
        session_id=None,
        limit=200,
        offset=0,
        q=None,
        origin=None,
        folder_id=None,
        root_only=False,
        expert_id=None,
        include_user_files=False,
        include_internal=False,
    )
    params.update(filters)
    response = await list_workspace_files(user_id=user_id, **params)
    return response.files


async def _run_backfill(workspace_id: str) -> None:
    """The migration's own statements, confined to this test's workspace."""
    sql = _MIGRATION.read_text()
    update = sql[
        sql.index('UPDATE "UserWorkspaceFile"') : sql.index(";", sql.index("UPDATE"))
    ]
    delete = (
        sql[sql.index('DELETE FROM "UnifiedContentEmbedding"') :].rstrip().rstrip(";")
    )
    client = prisma.get_client()
    await client.execute_raw(f'{update} AND "workspaceId" = $1', workspace_id)
    await client.execute_raw(f'{delete} AND uwf."workspaceId" = $1', workspace_id)


async def _embed(user_id: str, file_id: str) -> None:
    await prisma.get_client().execute_raw(
        'INSERT INTO "UnifiedContentEmbedding" '
        '(id, "updatedAt", "contentType", "contentId", "userId", embedding, "searchableText") '
        "VALUES ($1, now(), 'WORKSPACE_FILE', $2, $3, array_fill(0, ARRAY[1536])::vector, $4)",
        str(uuid.uuid4()),
        file_id,
        user_id,
        "skill",
    )
