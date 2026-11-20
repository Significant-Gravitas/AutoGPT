"""What a read returns from one of Otto's working files is withheld whole from
a shared chat, as the stored preview of a large tool output is: a window of
it, the whole text or its base64, or a preview of its head.

The rule goes by the file the read names, not by key: base64 quotes no key,
and a window can start at a value whose key lies outside it.  The saved
outputs and the reads are built by the code that makes them:
copilot/tools/base.py saves, ReadWorkspaceFileTool reads.
"""

import base64
import json
import re
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.gate.review import review_payload
from backend.copilot.model import ChatMessage
from backend.copilot.sharing.models import sanitize_chat_message
from backend.copilot.tools._test_data import make_session
from backend.copilot.tools.base import _LARGE_OUTPUT_THRESHOLD, _persist_and_summarize
from backend.copilot.tools.workspace_files import ReadWorkspaceFileTool
from backend.data.workspace import WorkspaceFile
from backend.data.workspace_internal import is_internal_workspace_file
from backend.util.workspace import WorkspaceManager

SECRET = "canary-7c3e-working-file"
SESSION_ID = "c0ffee00-1111-2222-3333-444444444444"
FILE_ID = "0c1f8a52-6a1e-4c58-9d0b-3f1f6d2a7e11"
NOW = datetime(2026, 11, 10, tzinfo=UTC)
REDACTED = "[redacted]"


def _shown(stored: str) -> dict[str, Any]:
    """The tool result a reader of the share sees for the stored row."""
    message = ChatMessage(
        id="m1",
        sequence=0,
        created_at=NOW,
        role="tool",
        content=stored,
        tool_call_id="call-2",
    )
    shown = sanitize_chat_message(message).content
    assert isinstance(shown, str)
    return json.loads(shown)


def _file(name: str, path: str, body: bytes, metadata: dict[str, Any]) -> WorkspaceFile:
    return WorkspaceFile(
        id=FILE_ID,
        workspace_id="ws-1",
        created_at=NOW,
        updated_at=NOW,
        name=name,
        path=path,
        storage_path=f"local://ws-1{path}",
        mime_type="application/json",
        size_bytes=len(body),
        metadata=metadata,
        is_internal=is_internal_workspace_file(name=name, path=path, metadata=metadata),
    )


async def _save(output: Any, *, digest: bool) -> tuple[WorkspaceFile, bytes, str]:
    """Save *output* as copilot/tools/base.py saves a large tool output: the
    file its write makes, the bytes it holds, and the wrapper the chat stores
    in its place."""
    db = AsyncMock()
    db.get_or_create_workspace = AsyncMock(return_value=MagicMock(id="ws-1"))
    manager = AsyncMock()
    with (
        patch("backend.copilot.tools.base.workspace_db", return_value=db),
        patch("backend.copilot.tools.base.WorkspaceManager", return_value=manager),
    ):
        wrapper = await _persist_and_summarize(
            json.dumps(output), "owner-1", SESSION_ID, "call-1", digest
        )
    assert wrapper.startswith("<tool-output-truncated")
    write = manager.write_file.await_args.kwargs
    path = WorkspaceManager("owner-1", "ws-1", SESSION_ID)._resolve_path(
        "/" + write["path"]
    )
    saved = _file(write["filename"], path, write["content"], write["metadata"])
    return saved, write["content"], wrapper


async def _read(file: WorkspaceFile, body: bytes, **kwargs: Any) -> str:
    """The row a read_workspace_file call on *file* stores."""
    manager = AsyncMock()
    manager.get_file_info_by_path = AsyncMock(return_value=file)
    manager.read_file_head_by_id = AsyncMock(side_effect=lambda _id, n: body[:n])
    manager.read_file_by_id = AsyncMock(return_value=body)
    with patch(
        "backend.copilot.tools.workspace_files.get_workspace_manager",
        AsyncMock(return_value=manager),
    ):
        response = await ReadWorkspaceFileTool()._execute(
            user_id="owner-1",
            session=make_session("owner-1"),
            path=file.path,
            **kwargs,
        )
    stored = response.model_dump_json(exclude_none=True)
    # Stored as it is, not saved again behind a preview of its own.
    assert len(stored) <= _LARGE_OUTPUT_THRESHOLD
    return stored


def _block_output(record: dict[str, Any]) -> dict[str, Any]:
    return {"type": "block_output", "outputs": {"response": [record]}}


def _records(rows: int, **fields: Any) -> dict[str, Any]:
    """Short records, so the text is dense in quotes once escaped as JSON."""
    rows_ = [{"id": f"r{n:04d}", "k": "v"} for n in range(rows)]
    return _block_output({**fields, "rows": rows_})


def _what_it_read(read: dict[str, Any]) -> str:
    if "content_base64" in read:
        return base64.b64decode(read["content_base64"]).decode()
    return read["content"]


async def _whole_read_in_base64() -> str:
    # Under the 32 KiB a whole read returns inline, and too long once escaped
    # as JSON for the read to return it as text.
    saved, body, _ = await _save(_records(1_350, access_token=SECRET), digest=True)
    stored = await _read(saved, body)
    assert "content_base64" in json.loads(stored)
    return stored


async def _outline_window(pattern: str) -> str:
    # Each container gets its own window in the outline.  Under a
    # secret-shaped key, a container's window starts at the value: it parses,
    # and the keys in it are not secret-shaped.
    output = _block_output(
        {
            "credentials": {"name": "svc-account", "value": f"cred-{SECRET}"},
            "api_keys": [f"key-{SECRET}", "sk-second"],
            "session": {"cookie": {"name": "sid", "value": f"sid-{SECRET}"}},
            "rows": [{"id": f"r{n:04d}", "note": "x" * 40} for n in range(200)],
        }
    )
    saved, body, wrapper = await _save(output, digest=True)
    assert 'format="outline"' in wrapper
    found = re.search(pattern, wrapper)
    assert found, f"no window for {pattern!r} in the outline"
    offset, length = int(found.group(1)), int(found.group(2))
    return await _read(saved, body, offset=offset, length=length)


async def _window_from_the_value() -> str:
    output = _block_output(
        {
            "items": [f"item {n:05d} " + "x" * 90 for n in range(500)],
            "access_token": SECRET,
            "more": [f"item {n:05d} " + "y" * 90 for n in range(500)],
        }
    )
    saved, body, _ = await _save(output, digest=False)
    # From the value's opening quote, as a read that continues where an
    # earlier one ended can start.
    at = body.decode().index(SECRET) - 1
    return await _read(saved, body, offset=at, length=50_000)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stored_row",
    [
        pytest.param(_whole_read_in_base64, id="whole read in base64"),
        pytest.param(
            lambda: _outline_window(r"credentials=\{…\d+ keys @(\d+)\+(\d+)\}"),
            id="window of an object under a secret-shaped key",
        ),
        pytest.param(
            lambda: _outline_window(r"api_keys=\[…\d+ items @(\d+)\+(\d+)\]"),
            id="window of a list under a secret-shaped key",
        ),
        pytest.param(
            lambda: _outline_window(r"cookie=\{…\d+ keys @(\d+)\+(\d+)\}"),
            id="window of a nested object under a secret-shaped key",
        ),
        pytest.param(_window_from_the_value, id="window from the value"),
    ],
)
async def test_a_read_of_a_saved_output_does_not_show_its_secrets(stored_row: Any):
    stored = await stored_row()
    read = json.loads(stored)
    field = "content_base64" if "content_base64" in read else "content"
    assert SECRET in _what_it_read(read)

    shown = _shown(stored)

    assert shown[field] == REDACTED
    assert SECRET not in json.dumps(shown)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("rows", "read", "field"),
    [
        pytest.param(200, {}, "content", id="whole read as text"),
        pytest.param(1_350, {}, "content_base64", id="whole read in base64"),
        pytest.param(200, {"offset": 10, "length": 500}, "content", id="window"),
        pytest.param(2_000, {}, "preview", id="preview of a larger output"),
    ],
)
async def test_nothing_a_read_returns_from_a_saved_output_is_shown(
    rows: int, read: dict[str, int], field: str
):
    # Nothing in it is secret-shaped: the read is withheld for what it reads.
    saved, body, _ = await _save(_records(rows), digest=True)
    assert saved.is_internal
    stored = await _read(saved, body, **read)
    assert json.loads(stored)[field] not in ("", REDACTED)

    shown = _shown(stored)

    assert shown == {**json.loads(stored), field: REDACTED}


@pytest.mark.asyncio
async def test_a_read_of_a_working_file_known_by_its_name_is_withheld():
    body = json.dumps({"step": 3, "notes": "wired the email block"}).encode()
    path = f"/sessions/{SESSION_ID}/build_state.json"
    scratchpad = _file("build_state.json", path, body, {})
    assert scratchpad.is_internal
    stored = await _read(scratchpad, body)

    shown = _shown(stored)

    assert shown == {**json.loads(stored), "content": REDACTED}


@pytest.mark.asyncio
async def test_a_read_of_the_users_own_file_is_redacted_by_key():
    body = json.dumps({"access_token": SECRET, "note": "quarterly figures"}).encode()
    report = _file("report.json", f"/sessions/{SESSION_ID}/report.json", body, {})
    assert not report.is_internal

    shown = _shown(await _read(report, body))

    assert shown["content"] != REDACTED
    assert json.loads(shown["content"]) == {
        "access_token": REDACTED,
        "note": "quarterly figures",
    }


@pytest.mark.asyncio
async def test_a_window_of_the_users_own_file_is_shown_as_read():
    body = json.dumps({"note": "quarterly figures", "rows": [1, 2, 3]}).encode()
    report = _file("report.json", f"/sessions/{SESSION_ID}/report.json", body, {})
    stored = await _read(report, body, offset=2, length=200)

    assert _shown(stored) == json.loads(stored)


def test_the_approval_card_keeps_what_a_call_writes_to_a_working_file():
    """The card's copy of the arguments must hash back to the review id, so
    it shows content written to a working file as the call will send it."""
    args = {
        "path": f"/sessions/{SESSION_ID}/tool-outputs/notes.json",
        "content": json.dumps({"step": 3}),
    }

    payload = review_payload("write_workspace_file", args)

    assert payload["arguments"] == args
