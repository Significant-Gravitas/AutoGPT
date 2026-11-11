"""Real-database coverage for SECRT-3445: once the user deletes a file or a
chat, Home's Recent work stops listing that file and stops linking to that chat.

Files and chats are written and deleted by the code that does it in production,
and read back through the dashboard Home renders.
"""

import uuid

import pytest

from backend.api.features.home.models import HomeRecentWorkItem
from backend.api.features.home.service import build_home_dashboard
from backend.api.features.workspace.routes import delete_workspace_file
from backend.copilot.model import ChatSession, create_chat_session, delete_chat_session
from backend.copilot.tools.base import BaseTool, _record_activity
from backend.copilot.tools.workspace_files import (
    DeleteWorkspaceFileTool,
    WriteWorkspaceFileTool,
)
from backend.data.user import get_or_create_user
from backend.util.test import SpinTestServer


@pytest.mark.asyncio(loop_scope="session")
async def test_recent_work_drops_the_files_the_user_deleted(server: SpinTestServer):
    user_id = await _user()
    chat = await create_chat_session(user_id, dry_run=False)
    await _run(WriteWorkspaceFileTool(), user_id, chat, filename="kept.md")
    removed = await _run(WriteWorkspaceFileTool(), user_id, chat, filename="old.md")
    await _run(WriteWorkspaceFileTool(), user_id, chat, filename="otto-removed.md")

    await delete_workspace_file(user_id=user_id, file_id=removed.file_id)
    await _run(DeleteWorkspaceFileTool(), user_id, chat, path="otto-removed.md")

    items = await _recent_work_items(user_id)
    assert [item.title for item in items] == ["kept.md"]


@pytest.mark.asyncio(loop_scope="session")
async def test_a_file_from_a_deleted_chat_stays_without_its_link(
    server: SpinTestServer,
):
    user_id = await _user()
    live_chat = await create_chat_session(user_id, dry_run=False)
    deleted_chat = await create_chat_session(user_id, dry_run=False)
    await _run(WriteWorkspaceFileTool(), user_id, live_chat, filename="live.md")
    await _run(WriteWorkspaceFileTool(), user_id, deleted_chat, filename="orphan.md")

    assert await delete_chat_session(deleted_chat.session_id, user_id)

    items = await _recent_work_items(user_id)
    assert {item.title: item.link for item in items} == {
        "live.md": f"/copilot?sessionId={live_chat.session_id}",
        "orphan.md": None,
    }


async def _user() -> str:
    suffix = uuid.uuid4().hex[:8]
    user = await get_or_create_user(
        {
            "sub": str(uuid.uuid4()),
            "email": f"recent-work-{suffix}@example.com",
            "name": "Kristy",
        }
    )
    return user.id


async def _run(tool: BaseTool, user_id: str, chat: ChatSession, **kwargs):
    if isinstance(tool, WriteWorkspaceFileTool):
        kwargs.setdefault("content", "notes")
    result = await tool._execute(user_id=user_id, session=chat, **kwargs)
    await _record_activity(tool, user_id, chat, result, kwargs)
    return result


async def _recent_work_items(user_id: str) -> list[HomeRecentWorkItem]:
    dashboard = await build_home_dashboard(user_id=user_id)
    return [item for group in dashboard.recent_work.groups for item in group.items]
