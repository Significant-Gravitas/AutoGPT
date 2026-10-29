"""Real-database coverage for SECRT-3140: the user's Files and Home show what
Otto made for them, not the working files it kept for itself.

Every file here is written by the code that writes it in production, and read
back through the route the Files page calls and the feed Home renders.
"""

import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import prisma
import pytest

from backend.api.features.home.recent_work import compose_recent_work
from backend.api.features.home.service import _get_work_events
from backend.api.features.workspace.routes import list_workspace_files
from backend.blocks.airtable.bases import AirtableListBasesBlock
from backend.blocks.airtable.schema import AirtableCreateTableBlock
from backend.copilot.tools._test_data import make_session
from backend.copilot.tools.agent_json_input import write_agent_json_to_workspace
from backend.copilot.tools.base import _persist_and_summarize, _record_activity
from backend.copilot.tools.models import BlockOutputResponse
from backend.copilot.tools.run_block import RunBlockTool
from backend.copilot.tools.workspace_files import WriteWorkspaceFileTool
from backend.data.activity_event import ActivityEventDraft, create_activity_event
from backend.data.user import get_or_create_user
from backend.data.workspace import create_workspace_file, get_or_create_workspace
from backend.util.test import SpinTestServer

# A 1x1 PNG: the sticker the user asked for.
_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

_MIGRATION = (
    Path(__file__).parents[4]
    / "migrations/20261023120000_workspace_file_is_internal/migration.sql"
)


@pytest.mark.asyncio(loop_scope="session")
async def test_files_and_home_show_the_deliverables_not_the_working_files(
    server: SpinTestServer,
):
    user_id = await _user()
    session = make_session(user_id)
    started = datetime.now(UTC) - timedelta(minutes=1)

    # Otto's working files, each by the code that writes it.
    await _persist_and_summarize(
        json.dumps([{"output": "x" * 200}] * 100),
        user_id,
        session.session_id,
        "sdk-926fe642657d",
    )
    await _write(user_id, session, filename="build_state.json", content="{}")
    await write_agent_json_to_workspace(
        {"nodes": []},
        "agent.json",
        user_id,
        session.session_id,
        label="Agent JSON",
        pass_to="create_agent",
        fallback_note="",
    )
    # What the user asked for, and an action taken for them.
    await _write(user_id, session, filename="sticker-01.png", content_base64=_PNG)
    await _run_block(user_id, session, AirtableCreateTableBlock(), "airtable")

    # Files: the page's own route lists the sticker and nothing else.
    listed = await _list_files(user_id, session_id=session.session_id)
    assert [f.name for f in listed] == ["sticker-01.png"]
    assert [f.name for f in await _list_files(user_id)] == ["sticker-01.png"]

    # Otto still reads its working files back.
    workspace = await get_or_create_workspace(user_id)
    rows = await prisma.models.UserWorkspaceFile.prisma().find_many(
        where={"workspaceId": workspace.id, "isDeleted": False}
    )
    assert {row.name: row.isInternal for row in rows} == {
        "sdk-926fe642657d.json": True,
        "build_state.json": True,
        "agent.json": True,
        "sticker-01.png": False,
    }

    # A row logged for the scratchpad before writes stopped logging it.
    scratchpad = next(row for row in rows if row.name == "build_state.json")
    await create_activity_event(
        user_id=user_id,
        draft=ActivityEventDraft(
            category="FILE",
            event_type="file.created",
            title="build_state.json",
            session_id=session.session_id,
            object_id=scratchpad.id,
        ),
    )

    # And an action logged under the block's class name, as rows were before.
    list_bases = AirtableListBasesBlock()
    await create_activity_event(
        user_id=user_id,
        draft=ActivityEventDraft(
            category="INTEGRATION",
            event_type="integration.action",
            title=list_bases.name,
            provider="airtable",
            session_id=session.session_id,
            object_id=list_bases.id,
        ),
    )

    # Home: the sticker and the Airtable actions, named as a person reads them.
    events = await _get_work_events(user_id=user_id, since=started)
    work = compose_recent_work(
        now=datetime.now(UTC),
        executions=[],
        events=events,
        expert_by_id={},
        agent_by_graph={},
        session_titles={},
    )
    titles = [item.title for group in work.groups for item in group.items]
    assert titles == ["Airtable List Bases", "Airtable Create Table", "sticker-01.png"]


@pytest.mark.asyncio(loop_scope="session")
async def test_the_migration_marks_working_files_written_before_it(
    server: SpinTestServer,
):
    user_id = await _user()
    workspace = await get_or_create_workspace(user_id)
    legacy = {
        "sdk-a1b2c3d4e5f6.json": (
            "/sessions/s1/tool-outputs/sdk-a1b2c3d4e5f6.json",
            {"purpose": "tool-output"},
        ),
        "toolu_01ABC.json": ("/sessions/s1/tool-results/toolu_01ABC.json", {}),
        "build_state.json": (
            "/sessions/s1/build_state.json",
            {"origin": "agent-created"},
        ),
        "._browser_state.json": ("/sessions/s1/._browser_state.json", {}),
        "sticker-01.png": ("/sessions/s1/sticker-01.png", {"origin": "agent-created"}),
        "notes.json": ("/my-pipeline/tool-outputs/notes.json", {}),
        "mine.json": ("/sessions/s1/tool-outputs/mine.json", {"origin": "user-upload"}),
    }
    ids = {}
    for name, (path, metadata) in legacy.items():
        ids[name] = str(uuid.uuid4())
        await create_workspace_file(
            workspace_id=workspace.id,
            file_id=ids[name],
            name=name,
            path=path,
            storage_path=f"{workspace.id}{path}",
            mime_type="application/json",
            size_bytes=2,
            metadata=metadata,
        )
    # Both were indexed for search before the column existed.
    await _embed(user_id, ids["build_state.json"])
    await _embed(user_id, ids["sticker-01.png"])

    await _run_backfill(workspace.id)

    rows = await prisma.models.UserWorkspaceFile.prisma().find_many(
        where={"workspaceId": workspace.id}
    )
    assert {row.name for row in rows if row.isInternal} == {
        "sdk-a1b2c3d4e5f6.json",
        "toolu_01ABC.json",
        "build_state.json",
        "._browser_state.json",
    }
    # The scratchpad's search entry is gone; the sticker's stays.
    embedded = await prisma.models.UnifiedContentEmbedding.prisma().find_many(
        where={"contentType": "WORKSPACE_FILE", "userId": user_id}
    )
    assert {row.contentId for row in embedded} == {ids["sticker-01.png"]}


async def _user() -> str:
    suffix = uuid.uuid4().hex[:8]
    user = await get_or_create_user(
        {
            "sub": str(uuid.uuid4()),
            "email": f"working-files-{suffix}@example.com",
            "name": "Kristy",
        }
    )
    return user.id


async def _write(user_id: str, session, **kwargs) -> None:
    tool = WriteWorkspaceFileTool()
    result = await tool._execute(user_id=user_id, session=session, **kwargs)
    await _record_activity(tool, user_id, session, result, kwargs)


async def _run_block(user_id: str, session, block, provider: str) -> None:
    result = BlockOutputResponse(
        message="Block executed",
        block_id=block.id,
        block_name=block.name,
        outputs={},
        provider=provider,
        session_id=session.session_id,
    )
    await _record_activity(RunBlockTool(), user_id, session, result, {})


async def _list_files(user_id: str, session_id: str | None = None):
    response = await list_workspace_files(
        user_id=user_id,
        session_id=session_id,
        limit=200,
        offset=0,
        q=None,
        origin=None,
        folder_id=None,
        root_only=False,
        expert_id=None,
        include_user_files=False,
    )
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
        "file",
    )
