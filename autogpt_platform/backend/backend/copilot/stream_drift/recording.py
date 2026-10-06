"""Record a copilot turn the way production stores it, serves it and persists it.

A recorded turn is three files under ``test/fixtures/copilot_stream/<name>/``:

- ``entries.jsonl``: the turn's Redis stream in XRANGE order, ``{id, data}``
  per line with ``data`` parsed from the stored JSON;
- ``frames.jsonl``: the SSE frame the resume route writes for each entry;
- ``rows.json``: the rows the session GET returns once the turn persisted.

Backend tests check the pipeline still records these files; the frontend drift
suite replays them. Set ``RECORD_COPILOT_STREAM_FIXTURES=1`` to rewrite them.
"""

import json
import os
import re
from collections.abc import AsyncGenerator, Callable, Sequence
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import orjson
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

from backend.copilot import stream_registry
from backend.copilot.model import ChatMessage
from backend.copilot.response_model import StreamBaseResponse, StreamError, StreamStatus
from backend.copilot.stream_heartbeat import wrap_stream_with_heartbeat
from backend.data.redis_client import get_redis_async

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
    session_id: str,
    turn_id: str,
    persisted: Callable[[], Sequence[ChatMessage]],
) -> RecordedTurn:
    """Run ``engine`` through the route's and the executor's publishing path."""
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

    with patch.object(
        stream_registry.chat_db(), "set_turn_duration", new=AsyncMock(), create=True
    ):
        await stream_registry.mark_session_completed(
            session_id, error_message=error, turn_id=turn_id
        )
    entries = await read_turn_entries(turn_id)
    redis = await get_redis_async()
    await redis.delete(stream_registry._get_turn_stream_key(turn_id))
    await redis.delete(stream_registry.get_session_meta_key(session_id))
    return canonical(
        RecordedTurn(
            entries=entries,
            frames=frames_for(entries),
            rows=[jsonable_encoder(message.model_dump()) for message in persisted()],
        ),
        session_id=session_id,
        turn_id=turn_id,
    )


async def read_turn_entries(turn_id: str) -> list[dict[str, Any]]:
    redis = await get_redis_async()
    key = stream_registry._get_turn_stream_key(turn_id)
    [(_, entries)] = stream_registry._stream_entries([(key, await redis.xrange(key))])
    return [
        {"id": entry_id, "data": orjson.loads(fields["data"])}
        for entry_id, fields in entries
    ]


def frames_for(entries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The frames the resume route writes, one per entry it can rebuild."""
    frames = []
    for entry in entries:
        chunk = stream_registry._reconstruct_chunk(entry["data"])
        if chunk is not None:
            frames.append({"id": entry["id"], "sse": chunk.to_sse()})
    return frames


def canonical(turn: RecordedTurn, *, session_id: str, turn_id: str) -> RecordedTurn:
    """Replace every value that differs between two recordings of one turn."""
    entry_ids = {entry["id"]: f"{i + 1}-0" for i, entry in enumerate(turn.entries)}
    rows = [
        {
            **row,
            "id": f"row-{sequence}",
            "sequence": sequence,
            "created_at": f"2026-09-30T00:00:{sequence:02d}Z",
        }
        for sequence, row in enumerate(turn.rows)
    ]
    text = json.dumps(
        {
            "entries": [{**e, "id": entry_ids[e["id"]]} for e in turn.entries],
            "frames": [{**f, "id": entry_ids[f["id"]]} for f in turn.frames],
            "rows": rows,
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
    return RecordedTurn.model_validate_json(text)


def load_fixture(name: str) -> RecordedTurn:
    directory = FIXTURE_ROOT / name
    return RecordedTurn(
        entries=_read_jsonl(directory / "entries.jsonl"),
        frames=_read_jsonl(directory / "frames.jsonl"),
        rows=json.loads((directory / "rows.json").read_text()),
    )


def check_fixture(name: str, recorded: RecordedTurn) -> None:
    """Fail when the pipeline no longer records the committed fixture."""
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
