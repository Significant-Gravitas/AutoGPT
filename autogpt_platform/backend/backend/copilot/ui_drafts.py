"""User edits to an unsaved OpenUI view, kept per chat (``ChatUIDraft`` rows
keyed by ``(sessionId, viewId)``). Saved views keep theirs on the file; see
``ui_view_files``."""

import json
from collections.abc import Mapping, Sequence
from typing import Self
from uuid import uuid4

from fastapi import HTTPException
from prisma import Prisma
from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.data import db
from backend.data.db import execute_raw_with_schema, query_raw_with_schema
from backend.data.openui_files import find_openui_view_file

# Plain JSON objects: request bodies arrive as JSON and rows hold JSONB. Typed
# as ``object`` rather than ``JsonValue``, whose recursive alias otherwise adds
# an empty ``JsonValue`` schema to the OpenAPI document.
DraftState = dict[str, object]

_VIEW_MESSAGES = (
    'SELECT m."content", m."sequence", m."toolCallId" FROM {schema_prefix}"ChatMessage" m '
    'JOIN {schema_prefix}"ChatSession" s ON s.id = m."sessionId" '
    'WHERE s.id = $1 AND s."userId" = $2 '
    'AND (m."toolCallId" = $3 OR (m.content LIKE \'%"ui_rendered"%\' AND m.content LIKE \'%"view_id"%\')) '
    "AND m.role = 'tool' ORDER BY m.sequence DESC"
)
_LATEST_VIEW_MESSAGES = (
    'SELECT m."content", m."sequence", m."toolCallId" FROM {schema_prefix}"ChatMessage" m '
    'JOIN {schema_prefix}"ChatSession" s ON s.id = m."sessionId" '
    'WHERE s.id = $1 AND s."userId" = $2 '
    "AND m.content LIKE '%\"ui_rendered\"%' AND m.content LIKE '%\"view_id\"%' "
    "AND m.role = 'tool' ORDER BY m.sequence DESC"
)
_CONFLICT = "A newer draft or view exists. Reload it before saving."
_SAVED = "This view is saved as a file. Reload it to keep editing."


class StateWrite(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    expected_revision: int = Field(ge=0, le=2_147_483_646)
    state: DraftState

    @model_validator(mode="after")
    def bounded_state(self) -> Self:
        if len(json.dumps(self.state, ensure_ascii=False).encode()) > 240_000:
            raise ValueError(
                "Draft exceeds 240 KB. Download large data before leaving."
            )
        return self


class UIDraftWrite(StateWrite):
    tool_call_id: str = Field(min_length=1, max_length=256)


class UIDraftResponse(BaseModel):
    revision: int
    source: str
    state: DraftState
    tool_call_id: str
    writable: bool
    # Set once the view is saved as a file: its edits then live on the file
    # (``/api/workspace/files/{file_id}/view-state``), not in this chat.
    file_id: str | None = None


class PublishedView(BaseModel):
    source: str
    sequence: int
    tool_call_id: str | None


async def read_draft(
    user_id: str, session_id: str, view_id: str, tool_call_id: str
) -> UIDraftResponse:
    messages = await query_raw_with_schema(
        _VIEW_MESSAGES, session_id, user_id, tool_call_id
    )
    source, sequence, latest = _published_view(messages, view_id, tool_call_id)
    file_id = await find_openui_view_file(user_id, session_id, view_id)
    if file_id is not None:
        return UIDraftResponse(
            revision=0,
            source=source,
            state={},
            tool_call_id=tool_call_id,
            writable=False,
            file_id=file_id,
        )
    rows = await query_raw_with_schema(
        'SELECT d.* FROM {schema_prefix}"ChatUIDraft" d JOIN {schema_prefix}"ChatSession" s '
        'ON s.id = d."sessionId" WHERE s.id = $1 AND s."userId" = $2 '
        'AND d."viewId" = $3',
        session_id,
        user_id,
        view_id,
    )
    if not rows:
        return UIDraftResponse(
            revision=0,
            source=source,
            state={},
            tool_call_id=tool_call_id,
            writable=latest,
        )
    row = rows[0]
    return UIDraftResponse(
        revision=row["revision"],
        source=row["source"],
        state=row["state"],
        tool_call_id=row["toolCallId"] or tool_call_id,
        writable=latest and sequence >= row["sequence"],
    )


async def write_draft(
    user_id: str, session_id: str, view_id: str, request: UIDraftWrite
) -> UIDraftResponse:
    async with db.transaction() as tx:
        await lock_owned_session(tx, user_id, session_id)
        if await find_openui_view_file(user_id, session_id, view_id):
            raise HTTPException(409, _SAVED)
        messages = await query_raw_with_schema(
            'SELECT content, sequence, "toolCallId" FROM {schema_prefix}"ChatMessage" WHERE "sessionId" = $1 '
            'AND ("toolCallId" = $2 OR (content LIKE \'%"ui_rendered"%\' AND content LIKE \'%"view_id"%\')) '
            "AND role = 'tool' ORDER BY sequence DESC",
            session_id,
            request.tool_call_id,
            client=tx,
        )
        source, sequence, latest = _published_view(
            messages, view_id, request.tool_call_id
        )
        rows = await query_raw_with_schema(
            'SELECT revision, sequence FROM {schema_prefix}"ChatUIDraft" WHERE "sessionId" = $1 AND "viewId" = $2',
            session_id,
            view_id,
            client=tx,
        )
        revision = rows[0]["revision"] if rows else 0
        if (
            not latest
            or revision != request.expected_revision
            or (rows and rows[0]["sequence"] > sequence)
        ):
            raise HTTPException(409, _CONFLICT)
        revision += 1
        await execute_raw_with_schema(
            'INSERT INTO {schema_prefix}"ChatUIDraft" (id, "sessionId", "viewId", "toolCallId", sequence, source, state, revision, "updatedAt") '
            "VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, NOW()) "
            'ON CONFLICT ("sessionId", "viewId") DO UPDATE SET "toolCallId" = EXCLUDED."toolCallId", '
            "sequence = EXCLUDED.sequence, source = EXCLUDED.source, state = EXCLUDED.state, "
            'revision = EXCLUDED.revision, "updatedAt" = NOW()',
            str(uuid4()),
            session_id,
            view_id,
            request.tool_call_id,
            sequence,
            source,
            json.dumps(request.state, ensure_ascii=False),
            revision,
            client=tx,
        )
    return UIDraftResponse(
        revision=revision,
        source=source,
        state=request.state,
        tool_call_id=request.tool_call_id,
        writable=True,
    )


async def lock_owned_session(tx: Prisma, user_id: str, session_id: str) -> None:
    """Lock the user's chat for this transaction; 404 when it is not theirs."""
    owners = await query_raw_with_schema(
        'SELECT id FROM {schema_prefix}"ChatSession" WHERE id = $1 AND "userId" = $2 FOR UPDATE',
        session_id,
        user_id,
        client=tx,
    )
    if not owners:
        raise HTTPException(404, "Interactive view not found")


async def latest_published_view(
    user_id: str, session_id: str, view_id: str
) -> PublishedView:
    """The newest revision of ``view_id`` published in the user's chat."""
    messages = await query_raw_with_schema(_LATEST_VIEW_MESSAGES, session_id, user_id)
    for message in messages:
        content, sequence = message.get("content"), message.get("sequence")
        if not isinstance(content, str) or not isinstance(sequence, int):
            continue
        try:
            source = _source(content, view_id)
        except HTTPException:
            continue
        tool_call_id = message.get("toolCallId")
        return PublishedView(
            source=source,
            sequence=sequence,
            tool_call_id=tool_call_id if isinstance(tool_call_id, str) else None,
        )
    raise HTTPException(404, "Interactive view not found")


def _source(content: str | None, view_id: str) -> str:
    try:
        value = json.loads(content or "null")
    except (ValueError, TypeError):
        value = None
    if (
        not isinstance(value, dict)
        or value.get("type") != "ui_rendered"
        or value.get("view_id") != view_id
        or not isinstance(value.get("source"), str)
    ):
        raise HTTPException(404, "Interactive view not found")
    return value["source"]


def _published_view(
    messages: Sequence[Mapping[str, object]], view_id: str, tool_call_id: str
) -> tuple[str, int, bool]:
    latest = -1
    for message in messages:
        content, sequence = message.get("content"), message.get("sequence")
        if not isinstance(content, str) or not isinstance(sequence, int):
            continue
        try:
            source = _source(content, view_id)
        except HTTPException:
            continue
        latest = max(latest, sequence)
        if message.get("toolCallId") == tool_call_id:
            return source, sequence, sequence == latest
    raise HTTPException(404, "Interactive view not found")
