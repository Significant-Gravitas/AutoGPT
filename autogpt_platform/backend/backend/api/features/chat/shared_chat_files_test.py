"""Which workspace files a shared chat serves, end to end.

A large tool output is saved in full as Otto's own working file, and reading
it back names that file's id in the chat, the same way reading one of the
user's files does.  The share serves, counts and allowlists only the user's
file.  Only the database and the file store are faked, so the public route,
the share lookups and the allowlist build all run.
"""

import json
from contextlib import ExitStack
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI
from prisma.models import ChatMessage as PrismaChatMessage
from prisma.models import ChatSession as PrismaChatSession
from prisma.models import SharedChatFile, UserWorkspace, UserWorkspaceFile

from backend.api.features.chat.share import public_router
from backend.copilot.sharing import db as share_db
from backend.copilot.tools.base import _persist_and_summarize
from backend.copilot.tools.workspace_files import WorkspaceFileContentResponse
from backend.data.workspace_internal import is_internal_workspace_file
from backend.util.workspace import WorkspaceManager

SHARE_TOKEN = "550e8400-e29b-41d4-a716-446655440000"
SESSION_ID = "c0ffee00-1111-2222-3333-444444444444"
OWNER_ID = "owner-1"
WORKSPACE_ID = "ws-1"
OUTPUT_ID = "0c1f8a52-6a1e-4c58-9d0b-3f1f6d2a7e11"
REPORT_ID = "5a0e3f4d-2b7c-4e1a-9c3d-8f6b1a2e4d70"
SECRET = "access-canary-71c3"
REPORT_BYTES = b"region,total\nnorth,12\n"
NOW = datetime(2026, 11, 9, 12, tzinfo=UTC)

app = FastAPI()
app.include_router(public_router, prefix="/api/public/shared/chats")


async def _saved_output() -> tuple[UserWorkspaceFile, bytes]:
    """A large block output saved in full, as copilot/tools/base.py saves it."""
    output = {"type": "block_output", "access_token": SECRET, "rows": ["r" * 99] * 600}
    db = AsyncMock()
    db.get_or_create_workspace = AsyncMock(return_value=MagicMock(id=WORKSPACE_ID))
    manager = AsyncMock()
    with (
        patch("backend.copilot.tools.base.workspace_db", return_value=db),
        patch("backend.copilot.tools.base.WorkspaceManager", return_value=manager),
    ):
        await _persist_and_summarize(
            json.dumps(output), OWNER_ID, SESSION_ID, "call-1", False
        )
    write = manager.write_file.await_args.kwargs
    path = WorkspaceManager(OWNER_ID, WORKSPACE_ID, SESSION_ID)._resolve_path(
        "/" + write["path"]
    )
    content = write["content"]
    row = _file_row(
        OUTPUT_ID,
        write["filename"],
        path,
        write["metadata"],
        write["mime_type"],
        content,
    )
    return row, content


def _file_row(
    file_id: str,
    name: str,
    path: str,
    metadata: dict[str, Any],
    mime_type: str,
    content: bytes,
) -> UserWorkspaceFile:
    """A file row as WorkspaceManager.write_file stores it."""
    return UserWorkspaceFile.model_construct(
        id=file_id,
        createdAt=NOW,
        updatedAt=NOW,
        workspaceId=WORKSPACE_ID,
        name=name,
        path=path,
        storagePath=f"local://{WORKSPACE_ID}{path}",
        mimeType=mime_type,
        sizeBytes=len(content),
        checksum=None,
        isDeleted=False,
        deletedAt=None,
        metadata=metadata,
        isInternal=is_internal_workspace_file(name=name, path=path, metadata=metadata),
        folderId=None,
    )


def _read_row(
    sequence: int, file: UserWorkspaceFile, content: str
) -> PrismaChatMessage:
    """A read_workspace_file result, which names the file it read."""
    result = WorkspaceFileContentResponse(
        file_id=file.id,
        name=file.name,
        path=file.path,
        mime_type="text/plain",
        content=content,
        message=f"Read chars 0–{len(content)} of {file.name}",
        session_id=SESSION_ID,
    )
    return PrismaChatMessage.model_construct(
        id=f"msg-{sequence}",
        sessionId=SESSION_ID,
        createdAt=NOW,
        sequence=sequence,
        role="tool",
        content=result.model_dump_json(exclude_none=True),
        toolCalls=None,
        functionCall=None,
    )


def _matches(row: Any, where: dict[str, Any], related: dict[str, Any]) -> bool:
    for field, wanted in where.items():
        if isinstance(wanted, dict) and "is" in wanted:
            if not _matches(related[field], wanted["is"], related):
                return False
        elif isinstance(wanted, dict) and "in" in wanted:
            if getattr(row, field) not in wanted["in"]:
                return False
        elif isinstance(wanted, dict) and "contains" in wanted:
            if wanted["contains"] not in (getattr(row, field) or ""):
                return False
        elif getattr(row, field) != wanted:
            return False
    return True


class _Database:
    """The rows the share code reads, behind the Prisma calls it makes."""

    def __init__(
        self,
        messages: list[PrismaChatMessage],
        files: list[UserWorkspaceFile],
        allowlisted: tuple[str, ...],
    ):
        self.session = PrismaChatSession.model_construct(
            id=SESSION_ID,
            userId=OWNER_ID,
            isShared=True,
            shareToken=SHARE_TOKEN,
            autoShareExecutions=False,
        )
        self.messages = messages
        self.files = {file.id: file for file in files}
        self.allowlist = [
            SharedChatFile.model_construct(
                sessionId=SESSION_ID, fileId=file_id, shareToken=SHARE_TOKEN
            )
            for file_id in allowlisted
        ]

    def table(self, rows: Any, related: dict[str, Any] | None = None) -> MagicMock:
        async def find_many(where: dict[str, Any] | None = None, **_: Any) -> list:
            return [row for row in rows() if _matches(row, where or {}, related or {})]

        async def find_first(
            where: dict[str, Any] | None = None,
            include: dict[str, bool] | None = None,
            **_: Any,
        ) -> Any:
            found = (await find_many(where))[:1]
            if found and include and include.get("File"):
                return found[0].model_copy(update={"File": self.files[found[0].fileId]})
            return found[0] if found else None

        async def count(where: dict[str, Any] | None = None, **_: Any) -> int:
            return len(await find_many(where))

        async def create(data: dict[str, Any], **_: Any) -> SharedChatFile:
            row = SharedChatFile.model_construct(**data)
            self.allowlist.append(row)
            return row

        client = MagicMock()
        for method in (find_many, find_first, count, create):
            setattr(client, method.__name__, AsyncMock(side_effect=method))
        client.find_unique = client.find_first
        return client

    def patched(self) -> ExitStack:
        workspace = UserWorkspace.model_construct(id=WORKSPACE_ID, userId=OWNER_ID)
        tables = {
            PrismaChatSession: self.table(lambda: [self.session]),
            PrismaChatMessage: self.table(lambda: self.messages),
            UserWorkspace: self.table(lambda: [workspace]),
            UserWorkspaceFile: self.table(lambda: list(self.files.values())),
            SharedChatFile: self.table(
                lambda: self.allowlist, related={"Session": self.session}
            ),
        }
        stack = ExitStack()
        for model, client in tables.items():
            stack.enter_context(patch.object(model, "prisma", return_value=client))
        return stack


async def _database(allowlisted: tuple[str, ...] = ()) -> tuple[_Database, bytes]:
    saved, saved_bytes = await _saved_output()
    report = _file_row(
        REPORT_ID,
        "report.csv",
        "/report.csv",
        {"origin": "user-upload"},
        "text/csv",
        REPORT_BYTES,
    )
    messages = [
        _read_row(2, saved, saved_bytes[:30_000].decode()),
        _read_row(4, report, REPORT_BYTES.decode()),
    ]
    return _Database(messages, [saved, report], allowlisted), saved_bytes


@pytest.mark.asyncio
async def test_a_share_allowlists_and_counts_only_the_users_files():
    database, _ = await _database()

    with database.patched():
        created = await share_db._build_shared_chat_files(
            SESSION_ID, SHARE_TOKEN, OWNER_ID
        )
        state = await share_db.get_chat_share_state(SESSION_ID, OWNER_ID)

    assert created == 1
    assert [row.fileId for row in database.allowlist] == [REPORT_ID]
    assert state.file_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "allowlisted",
    [
        pytest.param((), id="found by the re-scan"),
        # Written when the chat was shared, before saved outputs were left out.
        pytest.param((OUTPUT_ID, REPORT_ID), id="allowlisted before"),
    ],
)
async def test_the_share_serves_the_users_file_and_not_a_saved_output(
    allowlisted: tuple[str, ...],
):
    database, saved_bytes = await _database(allowlisted)
    stored = {
        f"local://{WORKSPACE_ID}{file.path}": data
        for file, data in (
            (database.files[OUTPUT_ID], saved_bytes),
            (database.files[REPORT_ID], REPORT_BYTES),
        )
    }

    async def stream_of(storage_path: str):
        yield stored[storage_path]

    storage = MagicMock()
    storage.retrieve = AsyncMock(side_effect=lambda storage_path: stored[storage_path])
    storage.retrieve_stream = MagicMock(side_effect=stream_of)
    base = f"/api/public/shared/chats/{SHARE_TOKEN}/files"

    with (
        database.patched(),
        patch(
            "backend.api.features.workspace.routes.get_workspace_storage",
            AsyncMock(return_value=storage),
        ),
    ):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://s"
        ) as client:
            output = await client.get(f"{base}/{OUTPUT_ID}/download")
            report = await client.get(f"{base}/{REPORT_ID}/download")

    assert SECRET.encode() in saved_bytes
    assert output.status_code == 404
    assert SECRET not in output.text
    assert report.status_code == 200
    assert report.content == REPORT_BYTES
    if not allowlisted:
        assert [row.fileId for row in database.allowlist] == [REPORT_ID]
