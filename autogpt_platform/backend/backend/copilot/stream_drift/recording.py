"""Record a copilot turn the way production stores it, serves it and persists it.

A recorded turn is three files under ``test/fixtures/copilot_stream/<name>/``:

- ``entries.jsonl``: the turn's whole Redis stream in XRANGE order, before
  completion trims it, ``{id, data}`` per line with ``data`` parsed from the
  stored JSON;
- ``frames.jsonl``: the SSE frame the stream routes write for each entry;
- ``rows.json``: the rows the session GET reads back from Postgres once the
  turn persisted.

Backend tests check the pipeline still records these files; the frontend drift
suite replays them. To rewrite them, run ``recorded_turns_test.py`` with
``RECORD_COPILOT_STREAM_FIXTURES=1`` and ``DATABASE_URL``/``DIRECT_URL`` on a
throwaway Postgres migrated from the branch (``prisma migrate deploy``).
"""

import json
import os
import re
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import orjson
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

from backend.api.features.chat.routes import _strip_injected_context
from backend.copilot import stream_registry
from backend.copilot.db import get_chat_messages_paginated
from backend.copilot.model import ChatMessage, ChatSession, upsert_chat_session
from backend.copilot.response_model import StreamBaseResponse, StreamError, StreamStatus
from backend.copilot.stream_checkpoint import canonical_digest, canonical_rows
from backend.copilot.stream_heartbeat import wrap_stream_with_heartbeat
from backend.data.redis_client import get_redis_async

from .fold import fold_rows

FIXTURE_ROOT = (
    Path(__file__).resolve().parents[3] / "test" / "fixtures" / "copilot_stream"
)
RECORD_ENV = "RECORD_COPILOT_STREAM_FIXTURES"

CANONICAL_SESSION_ID = "drift-session"
CANONICAL_TURN_ID = "drift-turn"

_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


class RecordedTurn(BaseModel):
    entries: list[dict[str, Any]]
    frames: list[dict[str, Any]]
    rows: list[dict[str, Any]]


async def record_turn(
    engine: AsyncGenerator[StreamBaseResponse, None],
    *,
    session: ChatSession,
    turn_id: str,
) -> RecordedTurn:
    """Run ``engine`` through the route's and the executor's publishing path,
    then read the turn's rows back from the database."""
    session_id = session.session_id
    await stream_registry.create_session(session_id, None, "", "", turn_id=turn_id)
    for status in ("Message received…", "Setting up your environment…"):
        await stream_registry.publish_chunk(
            turn_id, StreamStatus(message=status), session_id=session_id
        )

    error: str | None = None
    published = stream_registry.stream_and_publish(
        session_id=session_id,
        turn_id=turn_id,
        stream=wrap_stream_with_heartbeat(engine),
    )
    try:
        async for chunk in published:
            if isinstance(chunk, StreamError):
                error = chunk.errorText
                break
    finally:
        await published.aclose()

    entries = await read_turn_entries(turn_id)
    with patch.object(
        stream_registry.chat_db(), "set_turn_duration", new=AsyncMock(), create=True
    ):
        await stream_registry.mark_session_completed(
            session_id, error_message=error, turn_id=turn_id
        )
    entries += await read_turn_entries(turn_id, after=entries[-1]["id"])
    redis = await get_redis_async()
    await redis.delete(stream_registry._get_turn_stream_key(turn_id))
    await redis.delete(stream_registry._get_turn_meta_key(turn_id))
    await redis.delete(stream_registry.get_session_meta_key(session_id))
    entries, rows = canonical(
        entries, await persisted_rows(session), session_id=session_id, turn_id=turn_id
    )
    return RecordedTurn(entries=entries, frames=frames_for(entries), rows=rows)


async def persisted_session(
    user_id: str, prompt: str, *, history: Sequence[ChatMessage] = ()
) -> ChatSession:
    """A session whose prompt is already persisted, as the POST route leaves it."""
    session = ChatSession.new(user_id, dry_run=False)
    session.messages.extend([*history, ChatMessage(role="user", content=prompt)])
    await upsert_chat_session(session)
    return session


async def persisted_rows(session: ChatSession) -> list[dict[str, Any]]:
    """The session's rows as ``GET /sessions/{id}`` returns them."""
    page = await get_chat_messages_paginated(
        session.session_id, limit=200, user_id=session.user_id
    )
    assert page is not None, "the session was never persisted"
    return [
        jsonable_encoder(_strip_injected_context(message.model_dump()))
        for message in page.messages
    ]


async def read_turn_entries(turn_id: str, after: str = "0-0") -> list[dict[str, Any]]:
    redis = await get_redis_async()
    key = stream_registry._get_turn_stream_key(turn_id)
    raw = await redis.xrange(key, min=f"({after}" if after != "0-0" else "-")
    [(_, entries)] = stream_registry._stream_entries([(key, raw)])
    return [
        {"id": entry_id, "data": orjson.loads(fields["data"])}
        for entry_id, fields in entries
    ]


def frames_for(
    entries: Sequence[dict[str, Any]], turn_id: str = CANONICAL_TURN_ID
) -> list[dict[str, Any]]:
    """The frames the stream routes write, one per entry they can rebuild."""
    frames = []
    for entry in entries:
        chunk = stream_registry._reconstruct_chunk(entry["data"])
        if chunk is not None:
            sse = stream_registry.sse_frame((f"{turn_id}:{entry['id']}", chunk))
            frames.append({"id": entry["id"], "sse": sse})
    return frames


def canonical(
    entries: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    session_id: str,
    turn_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Replace every value that differs between two recordings of one turn."""
    entry_ids = {entry["id"]: f"{i + 1}-0" for i, entry in enumerate(entries)}
    numbered = [
        {
            **row,
            "id": f"row-{sequence}",
            "sequence": sequence,
            "created_at": f"2026-09-30T00:00:{sequence:02d}Z",
        }
        for sequence, row in enumerate(rows)
    ]
    text = json.dumps(
        {
            "entries": [{**e, "id": entry_ids[e["id"]]} for e in entries],
            "rows": numbered,
        }
    )
    text = text.replace(session_id, CANONICAL_SESSION_ID)
    text = text.replace(turn_id, CANONICAL_TURN_ID)
    uuids: dict[str, str] = {}
    text = _UUID.sub(
        lambda m: uuids.setdefault(
            m.group(0), f"00000000-0000-4000-8000-{len(uuids) + 1:012d}"
        ),
        text,
    )
    parsed = json.loads(text)
    return parsed["entries"], parsed["rows"]


def saving_into(
    saves: list[ChatSession],
) -> Callable[[ChatSession], Awaitable[ChatSession]]:
    """A stand-in for ``upsert_chat_session`` that numbers new rows the way
    the DB does, then keeps a copy of what it saved."""

    async def save(session: ChatSession) -> ChatSession:
        numbered = [m.sequence for m in session.messages if m.sequence is not None]
        next_sequence = max(numbered, default=-1) + 1
        for message in session.messages:
            if message.sequence is None:
                message.sequence = next_sequence
                next_sequence += 1
        saves.append(session.model_copy(deep=True))
        return session

    return save


def assert_fold_matches_rows(turn: RecordedTurn) -> None:
    """At every checkpoint the fold of the entries before it is the persisted
    turn rows it names, and at the end the fold is every persisted turn row."""
    chunks = [entry["data"] for entry in turn.entries]
    checkpoints = [
        i for i, chunk in enumerate(chunks) if chunk["type"] == "data-checkpoint"
    ]
    assert checkpoints, "the turn published no checkpoint"
    starts = {chunks[i]["sequence"] for i in checkpoints}
    assert len(starts) == 1, f"checkpoints name different first rows: {starts}"
    [start] = starts
    for i in checkpoints:
        folded = fold_rows(chunks[:i])
        assert (len(folded), canonical_digest(folded)) == (
            chunks[i]["rows"],
            chunks[i]["digest"],
        ), f"checkpoint {turn.entries[i]['id']} does not match the fold {folded}"
    persisted = canonical_rows([ChatMessage.model_validate(r) for r in turn.rows])
    assert fold_rows(chunks) == persisted[start:]


def fixture_names() -> list[str]:
    return sorted(path.name for path in FIXTURE_ROOT.iterdir() if path.is_dir())


def load_fixture(name: str) -> RecordedTurn:
    directory = FIXTURE_ROOT / name
    return RecordedTurn(
        entries=_read_jsonl(directory / "entries.jsonl"),
        frames=_read_jsonl(directory / "frames.jsonl"),
        rows=json.loads((directory / "rows.json").read_text()),
    )


def check_fixture(name: str, recorded: RecordedTurn) -> None:
    """Fail when the recorded stream does not fold to its rows, or the
    pipeline no longer records the committed fixture."""
    assert_fold_matches_rows(recorded)
    if os.environ.get(RECORD_ENV):
        _write_fixture(name, recorded)
        return
    committed = load_fixture(name)
    assert recorded == committed, (
        f"The pipeline no longer records {FIXTURE_ROOT / name}. If the change is "
        f"intended, re-record with {RECORD_ENV}=1 and check the frontend drift "
        "suite still passes on the new files."
    )


def _write_fixture(name: str, turn: RecordedTurn) -> None:
    directory = FIXTURE_ROOT / name
    directory.mkdir(parents=True, exist_ok=True)
    _write_jsonl(directory / "entries.jsonl", turn.entries)
    _write_jsonl(directory / "frames.jsonl", turn.frames)
    (directory / "rows.json").write_text(
        json.dumps(turn.rows, indent=2, ensure_ascii=False) + "\n"
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _write_jsonl(path: Path, items: Sequence[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items)
    )
