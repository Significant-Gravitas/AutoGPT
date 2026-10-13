"""Saved OpenUI views: the user's edits to a view file, and saving a chat's
view as one.

A saved view's edits live in the ``ChatUIDraft`` row tied to its file. A write
must name both the edits revision and the file checksum it was made against,
so a stale tab can never put edits onto a newer program.
"""

import json
from uuid import uuid4

from fastapi import HTTPException
from prisma.models import ChatSession, ChatUIDraft
from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.api.features.store.exceptions import VirusDetectedError, VirusScanError
from backend.copilot.ui_drafts import (
    DraftState,
    PublishedView,
    StateWrite,
    latest_published_view,
)
from backend.data import db
from backend.data.db import execute_raw_with_schema, query_raw_with_schema
from backend.data.openui_files import (
    find_openui_view_file,
    is_openui_file,
    move_chat_edits_to_file,
    openui_filename,
    openui_metadata,
    openui_origin_session_id,
    openui_view_id,
)
from backend.data.workspace import (
    OPENUI_MIME_TYPE,
    WorkspaceFile,
    get_or_create_workspace,
    get_workspace,
    get_workspace_file,
)
from backend.util.workspace import WorkspaceManager
from backend.util.workspace_storage import compute_file_checksum, get_workspace_storage

_NOT_FOUND = "Saved view not found"
_STALE = "The view or its edits changed. Reload it before saving."


class UIFileStateWrite(StateWrite):
    expected_checksum: str = Field(min_length=1, max_length=128)


class UIFileStateResponse(BaseModel):
    file_id: str
    view_id: str
    source: str
    # The file's content checksum; send it back as expected_checksum.
    checksum: str
    state: DraftState
    revision: int
    # The chat that last wrote the view; None once that chat is gone.
    origin_session_id: str | None


class UIDraftSaveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str = Field(min_length=1, max_length=255)

    @field_validator("filename")
    @classmethod
    def plain_name(cls, value: str) -> str:
        name = value.strip()
        if not name or name in {".", ".."} or "/" in name or "\\" in name:
            raise ValueError("File name must be a plain name without slashes")
        return openui_filename(name)


class SavedUIViewResponse(BaseModel):
    file_id: str
    file_name: str
    file_path: str
    file_uri: str
    view_id: str
    checksum: str


async def read_file_state(user_id: str, file_id: str) -> UIFileStateResponse:
    file, view_id = await _owned_view_file(user_id, file_id)
    content = await _content(file)
    row = await ChatUIDraft.prisma().find_unique(where={"fileId": file.id})
    return UIFileStateResponse(
        file_id=file.id,
        view_id=view_id,
        source=content.decode("utf-8", errors="replace"),
        checksum=file.checksum or compute_file_checksum(content),
        state=row.state if row is not None and isinstance(row.state, dict) else {},
        revision=row.revision if row is not None else 0,
        origin_session_id=await _owned_session_id(
            user_id, openui_origin_session_id(file.metadata)
        ),
    )


async def write_file_state(
    user_id: str, file_id: str, request: UIFileStateWrite
) -> UIFileStateResponse:
    file, view_id = await _owned_view_file(user_id, file_id)
    content = await _content(file)
    source = content.decode("utf-8", errors="replace")
    async with db.transaction() as tx:
        locked = await query_raw_with_schema(
            'SELECT f.checksum FROM {schema_prefix}"UserWorkspaceFile" f '
            'JOIN {schema_prefix}"UserWorkspace" w ON w.id = f."workspaceId" '
            'WHERE f.id = $1 AND w."userId" = $2 AND f."isDeleted" = false '
            "FOR UPDATE OF f",
            file.id,
            user_id,
            client=tx,
        )
        if not locked:
            raise HTTPException(404, _NOT_FOUND)
        checksum = compute_file_checksum(content)
        if request.expected_checksum != checksum or locked[0]["checksum"] not in (
            checksum,
            None,
        ):
            raise HTTPException(409, _STALE)
        rows = await query_raw_with_schema(
            'SELECT id, revision FROM {schema_prefix}"ChatUIDraft" '
            'WHERE "fileId" = $1 FOR UPDATE',
            file.id,
            client=tx,
        )
        revision = rows[0]["revision"] if rows else 0
        if revision != request.expected_revision:
            raise HTTPException(409, _STALE)
        revision += 1
        await execute_raw_with_schema(
            'INSERT INTO {schema_prefix}"ChatUIDraft" (id, "fileId", "viewId", sequence, source, state, revision, "updatedAt") '
            "VALUES ($1, $2, $3, 0, $4, $5::jsonb, $6, NOW()) "
            'ON CONFLICT ("fileId") DO UPDATE SET "viewId" = EXCLUDED."viewId", '
            "source = EXCLUDED.source, state = EXCLUDED.state, "
            'revision = EXCLUDED.revision, "updatedAt" = NOW()',
            str(uuid4()),
            file.id,
            view_id,
            source,
            json.dumps(request.state, ensure_ascii=False),
            revision,
            client=tx,
        )
    return UIFileStateResponse(
        file_id=file.id,
        view_id=view_id,
        source=source,
        checksum=checksum,
        state=request.state,
        revision=revision,
        origin_session_id=await _owned_session_id(
            user_id, openui_origin_session_id(file.metadata)
        ),
    )


async def save_draft_as_file(
    user_id: str, session_id: str, view_id: str, request: UIDraftSaveRequest
) -> SavedUIViewResponse:
    """Save the newest revision of a chat's view as a file and move its edits
    onto the file. Saving an already saved view returns that file."""
    view = await latest_published_view(user_id, session_id, view_id)
    saved_id = await find_openui_view_file(user_id, session_id, view_id)
    if saved_id is not None:
        return await _saved_response(user_id, saved_id)
    workspace = await get_or_create_workspace(user_id)
    manager = WorkspaceManager(user_id, workspace.id, session_id)
    try:
        file = await _write_view_file(manager, view, view_id, request.filename)
    except HTTPException as error:
        # A concurrent save of this view under the same name won the path.
        if error.status_code != 409:
            raise
        saved_id = await find_openui_view_file(user_id, session_id, view_id)
        if saved_id is None:
            raise
        return await _saved_response(user_id, saved_id)
    # Two concurrent saves each wrote a file; both keep the oldest.
    canonical = await find_openui_view_file(user_id, session_id, view_id)
    if canonical is not None and canonical != file.id:
        await manager.delete_file(file.id)
        return await _saved_response(user_id, canonical)
    await _move_edits_to_file(user_id, session_id, view_id, file.id, view)
    return _saved(file, view_id)


async def _write_view_file(
    manager: WorkspaceManager, view: PublishedView, view_id: str, filename: str
) -> WorkspaceFile:
    session_id = manager.session_id or ""
    try:
        return await manager.write_file(
            content=view.source.encode(),
            filename=filename,
            mime_type=OPENUI_MIME_TYPE,
            metadata=openui_metadata({"origin": "agent-created"}, view_id, session_id),
        )
    except VirusDetectedError as error:
        raise HTTPException(400, str(error)) from error
    except VirusScanError as error:
        raise HTTPException(503, "File scanning is unavailable. Try again.") from error
    except ValueError as error:
        message = str(error)
        if message.startswith("File already exists"):
            raise HTTPException(409, "A file with this name already exists") from error
        raise HTTPException(400, message) from error


async def _move_edits_to_file(
    user_id: str, session_id: str, view_id: str, file_id: str, view: PublishedView
) -> None:
    """Move the chat's edits row onto the file, or start an empty one."""
    if await move_chat_edits_to_file(user_id, session_id, view_id, file_id):
        return
    await execute_raw_with_schema(
        'INSERT INTO {schema_prefix}"ChatUIDraft" (id, "fileId", "viewId", "toolCallId", sequence, source, state, revision, "updatedAt") '
        "VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, 0, NOW()) "
        'ON CONFLICT ("fileId") DO NOTHING',
        str(uuid4()),
        file_id,
        view_id,
        view.tool_call_id,
        view.sequence,
        view.source,
        "{}",
    )


async def _owned_view_file(user_id: str, file_id: str) -> tuple[WorkspaceFile, str]:
    workspace = await get_workspace(user_id)
    file = await get_workspace_file(file_id, workspace.id) if workspace else None
    view_id = (
        openui_view_id(file.metadata)
        if file is not None and is_openui_file(file)
        else None
    )
    if file is None or view_id is None:
        raise HTTPException(404, _NOT_FOUND)
    return file, view_id


async def _content(file: WorkspaceFile) -> bytes:
    storage = await get_workspace_storage()
    return await storage.retrieve(file.storage_path)


async def _owned_session_id(user_id: str, session_id: str | None) -> str | None:
    if session_id is None:
        return None
    session = await ChatSession.prisma().find_first(
        where={"id": session_id, "userId": user_id}
    )
    return session.id if session is not None else None


async def _saved_response(user_id: str, file_id: str) -> SavedUIViewResponse:
    file, view_id = await _owned_view_file(user_id, file_id)
    return _saved(file, view_id)


def _saved(file: WorkspaceFile, view_id: str) -> SavedUIViewResponse:
    return SavedUIViewResponse(
        file_id=file.id,
        file_name=file.name,
        file_path=file.path,
        file_uri=f"workspace://{file.id}#{OPENUI_MIME_TYPE}",
        view_id=view_id,
        checksum=file.checksum or "",
    )
