"""Text in a tool row that doesn't parse is withheld from a shared chat when it
quotes a secret-shaped key, and shown as written otherwise.

The stored wrapper of a large result tells the model to read the saved output
with read_workspace_file, and that read returns a window or a preview of the
saved JSON, which never parses.  Other tools show JSON lines, a dict's repr or
a cut of JSON in a value.  The windows and previews here are built by the
tools that return them, from what copilot/tools/base.py saves.
"""

import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.model import ChatMessage
from backend.copilot.sharing.models import sanitize_chat_message
from backend.copilot.tools._test_data import make_session
from backend.copilot.tools.base import _persist_and_summarize
from backend.copilot.tools.models import BashExecResponse
from backend.copilot.tools.workspace_files import (
    ReadWorkspaceFileTool,
    WorkspaceWriteResponse,
)
from backend.data.workspace import WorkspaceFile
from backend.data.workspace_internal import is_internal_workspace_file
from backend.util.workspace import WorkspaceManager

ACCESS_SECRET = "access-canary-2f6b"
REFRESH_SECRET = "refresh-canary-9a1d"
SECRETS = (ACCESS_SECRET, REFRESH_SECRET)
SESSION_ID = "c0ffee00-1111-2222-3333-444444444444"
FILE_ID = "0c1f8a52-6a1e-4c58-9d0b-3f1f6d2a7e11"


def _message(**fields: Any) -> ChatMessage:
    return ChatMessage(
        id="m1", sequence=0, created_at=datetime(2026, 11, 9, tzinfo=UTC), **fields
    )


def _shown(content: str) -> str:
    message = _message(role="tool", content=content, tool_call_id="call-2")
    return sanitize_chat_message(message).model_dump_json()


def _stdout(text: str) -> BashExecResponse:
    return BashExecResponse(
        stdout=text,
        stderr="",
        exit_code=0,
        message="Command finished with exit code 0",
        session_id=SESSION_ID,
    )


async def _saved_output() -> tuple[WorkspaceFile, bytes]:
    """A large block output saved in full, as copilot/tools/base.py saves it:
    the file row the write makes, and the bytes it holds."""
    output = {
        "type": "block_output",
        "outputs": {
            "response": [
                {
                    "access_token": ACCESS_SECRET,
                    "items": [f"item {n:05d} " + "x" * 90 for n in range(1_000)],
                    "refresh_token": REFRESH_SECRET,
                }
            ]
        },
    }
    db = AsyncMock()
    db.get_or_create_workspace = AsyncMock(return_value=MagicMock(id="ws-1"))
    manager = AsyncMock()
    with (
        patch("backend.copilot.tools.base.workspace_db", return_value=db),
        patch("backend.copilot.tools.base.WorkspaceManager", return_value=manager),
    ):
        stored = await _persist_and_summarize(
            json.dumps(output), "owner-1", SESSION_ID, "call-1", False
        )
    assert stored.startswith("<tool-output-truncated")
    write = manager.write_file.await_args.kwargs
    path = WorkspaceManager("owner-1", "ws-1", SESSION_ID)._resolve_path(
        "/" + write["path"]
    )
    now = datetime(2026, 11, 9, tzinfo=UTC)
    saved = WorkspaceFile(
        id=FILE_ID,
        workspace_id="ws-1",
        created_at=now,
        updated_at=now,
        name=write["filename"],
        path=path,
        storage_path=f"local://ws-1{path}",
        mime_type=write["mime_type"],
        size_bytes=len(write["content"]),
        metadata=write["metadata"],
        is_internal=is_internal_workspace_file(
            name=write["filename"], path=path, metadata=write["metadata"]
        ),
    )
    return saved, write["content"]


async def _read_back(saved: WorkspaceFile, body: bytes, **kwargs: Any) -> str:
    """What a read_workspace_file call on the saved output stores as its row."""
    manager = AsyncMock()
    manager.get_file_info_by_path = AsyncMock(return_value=saved)
    manager.read_file_head_by_id = AsyncMock(side_effect=lambda _id, n: body[:n])
    with patch(
        "backend.copilot.tools.workspace_files.get_workspace_manager",
        AsyncMock(return_value=manager),
    ):
        response = await ReadWorkspaceFileTool()._execute(
            user_id="owner-1",
            session=make_session("owner-1"),
            path=saved.path,
            **kwargs,
        )
    return response.model_dump_json(exclude_none=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "window",
    [
        # As the stored wrapper suggests: from the start, 50,000 at a time.
        pytest.param({"offset": 0, "length": 50_000}, id="first window"),
        # A later window starts inside a value, not at a bracket.
        pytest.param({"offset": -20_000, "length": 50_000}, id="last window"),
        # No range on a large file: its metadata and a preview of its head.
        pytest.param({}, id="preview"),
    ],
)
async def test_a_read_of_a_saved_output_is_withheld(window: dict[str, int]):
    saved, body = await _saved_output()
    assert saved.is_internal
    if window.get("offset", 0) < 0:
        window = {**window, "offset": len(body) + window["offset"]}
    stored = await _read_back(saved, body, **window)
    read = json.loads(stored)
    text = read.get("content") or read.get("preview")
    assert text and any(secret in text for secret in SECRETS)

    shown = json.loads(json.loads(_shown(stored))["content"])

    assert shown["content" if "content" in read else "preview"] == "[redacted]"
    assert shown["path"] == saved.path
    assert not any(secret in json.dumps(shown) for secret in SECRETS)


@pytest.mark.parametrize(
    "result",
    [
        pytest.param(
            _stdout(
                "".join(
                    json.dumps({"access_token": ACCESS_SECRET, "n": n}) + "\n"
                    for n in range(3)
                )
            ),
            id="JSON lines",
        ),
        pytest.param(
            _stdout(str({"access_token": ACCESS_SECRET, "expires_in": 3600})),
            id="a dict's repr",
        ),
        pytest.param(
            _stdout(
                "token response: "
                + json.dumps({"access_token": ACCESS_SECRET, "expires_in": 3600})
            ),
            id="JSON after a label",
        ),
        pytest.param(
            # Its keys end in the backslash before each escaped quote.
            _stdout(json.dumps(json.dumps({"access_token": ACCESS_SECRET}))[:-4]),
            id="JSON quoted in a string, cut",
        ),
        pytest.param(
            WorkspaceWriteResponse(
                file_id=FILE_ID,
                name="token.json",
                path="/token.json",
                mime_type="application/json",
                size_bytes=2_048,
                download_url=f"workspace://{FILE_ID}#application/json",
                content_preview=json.dumps(
                    {"access_token": ACCESS_SECRET, "scope": "s" * 400}
                )[:200]
                + "...",
                message="Wrote token.json to workspace (2,048 bytes)",
                session_id=SESSION_ID,
            ),
            id="a written file's preview",
        ),
    ],
)
def test_a_value_that_does_not_parse_and_names_a_secret_is_withheld(result: Any):
    stored = result.model_dump_json(exclude_none=True)
    assert ACCESS_SECRET in stored

    assert ACCESS_SECRET not in _shown(stored)


def test_a_result_that_does_not_parse_and_names_a_secret_is_withheld():
    # A byte-order mark before the JSON: not read as JSON, nor as a bracket.
    stored = "\ufeff" + json.dumps({"access_token": ACCESS_SECRET})

    assert sanitize_chat_message(_message(role="tool", content=stored)).content == (
        "[redacted]"
    )


def test_a_cut_input_that_names_a_secret_is_withheld():
    # A tool_use block whose input was stored as a JSON string, cut short.
    cut = json.dumps({"headers": {"Authorization": f"Bearer {ACCESS_SECRET}"}})[:-3]
    call = {"type": "tool_use", "id": "toolu_1", "name": "run_block", "input": cut}

    sanitized = sanitize_chat_message(_message(role="assistant", tool_calls=[call]))

    assert sanitized.tool_calls is not None
    assert sanitized.tool_calls[0]["input"] == "[redacted]"


@pytest.mark.parametrize(
    "text",
    [
        "[INFO] build started\n[INFO] build finished in 3.2s",
        "[Read the docs](https://example.com/docs) for details.",
        'Set "verbose": true and "retries": 3 in the config.',
        '{"usage": {"total_tokens": 12, "prompt_tokens": 3',
        "{'status': 'ok', 'author': 'Ada Lovelace'}",
        'Error: "headers": must be an object',
    ],
)
def test_text_that_names_no_secret_is_shown_as_written(text: str):
    nested = _stdout(text).model_dump_json(exclude_none=True)

    assert json.loads(_shown(nested))["content"] == nested
    if text[0] not in "[{":
        # As a whole result, too; one that opens like JSON and doesn't parse
        # is withheld either way.
        assert sanitize_chat_message(_message(role="tool", content=text)).content == (
            text
        )
