import json
from collections.abc import Mapping, Sequence
from typing import Annotated

from fastapi import HTTPException
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    WithJsonSchema,
    model_validator,
)

from backend.data import db

DraftState = Annotated[
    dict[str, JsonValue],
    WithJsonSchema({"type": "object", "additionalProperties": True}),
]


class UIDraftWrite(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    tool_call_id: str = Field(min_length=1, max_length=256)
    expected_revision: int = Field(ge=0, le=2_147_483_646)
    state: DraftState

    @model_validator(mode="after")
    def bounded_state(self) -> "UIDraftWrite":
        if len(json.dumps(self.state, ensure_ascii=False).encode()) > 240_000:
            raise ValueError(
                "Draft exceeds 240 KB. Download large data before leaving."
            )
        return self


class UIDraftResponse(BaseModel):
    revision: int
    source: str
    state: DraftState
    tool_call_id: str
    writable: bool


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


async def read_draft(
    user_id: str, session_id: str, view_id: str, tool_call_id: str
) -> UIDraftResponse:
    messages = await db.prisma.query_raw(
        'SELECT m."content", m."sequence", m."toolCallId" FROM "ChatMessage" m '
        'JOIN "ChatSession" s ON s.id = m."sessionId" '
        'WHERE s.id = $1 AND s."userId" = $2 '
        'AND (m."toolCallId" = $3 OR (m.content LIKE \'%"ui_rendered"%\' AND m.content LIKE \'%"view_id"%\')) '
        "AND m.role = 'tool' ORDER BY m.sequence DESC",
        session_id,
        user_id,
        tool_call_id,
    )
    source, sequence, latest = _published_view(messages, view_id, tool_call_id)
    rows = await db.prisma.query_raw(
        'SELECT d.* FROM "ChatUIDraft" d JOIN "ChatSession" s '
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
        tool_call_id=row["toolCallId"],
        writable=latest and sequence >= row["sequence"],
    )


async def write_draft(
    user_id: str, session_id: str, view_id: str, request: UIDraftWrite
) -> UIDraftResponse:
    async with db.transaction() as tx:
        owners = await tx.query_raw(
            'SELECT id FROM "ChatSession" WHERE id = $1 AND "userId" = $2 FOR UPDATE',
            session_id,
            user_id,
        )
        if not owners:
            raise HTTPException(404, "Interactive view not found")
        messages = await tx.query_raw(
            'SELECT content, sequence, "toolCallId" FROM "ChatMessage" WHERE "sessionId" = $1 '
            'AND ("toolCallId" = $2 OR (content LIKE \'%"ui_rendered"%\' AND content LIKE \'%"view_id"%\')) '
            "AND role = 'tool' ORDER BY sequence DESC",
            session_id,
            request.tool_call_id,
        )
        source, sequence, latest = _published_view(
            messages, view_id, request.tool_call_id
        )
        rows = await tx.query_raw(
            'SELECT revision, sequence FROM "ChatUIDraft" WHERE "sessionId" = $1 AND "viewId" = $2',
            session_id,
            view_id,
        )
        revision = rows[0]["revision"] if rows else 0
        if (
            not latest
            or revision != request.expected_revision
            or (rows and rows[0]["sequence"] > sequence)
        ):
            raise HTTPException(
                409, "A newer draft or view exists. Reload it before saving."
            )
        revision += 1
        await tx.execute_raw(
            'INSERT INTO "ChatUIDraft" ("sessionId", "viewId", "toolCallId", sequence, source, state, revision, "updatedAt") '
            "VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, NOW()) "
            'ON CONFLICT ("sessionId", "viewId") DO UPDATE SET "toolCallId" = EXCLUDED."toolCallId", '
            "sequence = EXCLUDED.sequence, source = EXCLUDED.source, state = EXCLUDED.state, "
            'revision = EXCLUDED.revision, "updatedAt" = NOW()',
            session_id,
            view_id,
            request.tool_call_id,
            sequence,
            source,
            json.dumps(request.state, ensure_ascii=False),
            revision,
        )
    return UIDraftResponse(
        revision=revision,
        source=source,
        state=request.state,
        tool_call_id=request.tool_call_id,
        writable=True,
    )
