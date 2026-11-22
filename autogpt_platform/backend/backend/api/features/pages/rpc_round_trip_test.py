"""The page store the copilot executor reaches through the database manager
returns records that survive the trip as models, not as plain dicts."""

import json
from datetime import UTC, datetime

import pytest

from backend.util.service import _build_return_adapter

from . import db
from .db import PageRecord

RECORD = PageRecord(
    id="page-1",
    title="Posts",
    design={"title": "Posts", "sections": [{"kind": "text", "body": "Hi"}]},
    previous_design=None,
    revision=2,
    pinned_library_agent_id="lib-1",
    chat_session_id="chat-1",
    updated_at=datetime(2026, 11, 21, 12, tzinfo=UTC),
)


@pytest.mark.parametrize(
    "func, returns",
    [
        (db.get_page, PageRecord | None),
        (db.create_page, PageRecord),
        (db.save_page_design, PageRecord),
        (db.set_page_pin, PageRecord),
    ],
)
def test_page_records_survive_the_rpc_round_trip(func, returns) -> None:
    adapter = _build_return_adapter(func, returns, func.__name__)

    back = adapter.validate_python(json.loads(RECORD.model_dump_json()))

    assert back == RECORD
