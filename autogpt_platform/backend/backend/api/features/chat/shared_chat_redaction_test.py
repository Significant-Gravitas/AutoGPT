"""What a reader of a shared chat's link receives, end to end.

The rows are stored the way the engines and the approval gate write them, and
only the database is mocked, so the public route, the share read path and the
sanitizer all run.
"""

import json
import re
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI
from prisma.models import ChatMessage as PrismaChatMessage
from prisma.models import ChatSession as PrismaChatSession

from backend.api.features.chat.share import public_router
from backend.copilot.gate.held import HeldCall, _result_row
from backend.copilot.response_model import StreamToolOutputAvailable
from backend.copilot.tools.base import _persist_and_summarize

SHARE_TOKEN = "550e8400-e29b-41d4-a716-446655440000"
SESSION_ID = "c0ffee00-1111-2222-3333-444444444444"
CREATED = datetime(2026, 11, 9, 12, tzinfo=UTC)

HEADER_SECRET = "hdr-canary-3a9c"
ACCESS_SECRET = "access-canary-6e21"
REFRESH_SECRET = "refresh-canary-d407"
SECRETS = (HEADER_SECRET, ACCESS_SECRET, REFRESH_SECRET)

TOKEN_REQUEST = {
    "url": "https://auth.example.com/oauth/token",
    "headers": {"Authorization": f"Bearer {HEADER_SECRET}"},
}
# A large result as the share shows it: its size, not its preview.
WITHHELD_PREVIEW = re.compile(
    r"<tool-output-truncated total_chars=\d+>\n\[redacted\]\n</tool-output-truncated>"
)

app = FastAPI()
app.include_router(public_router, prefix="/api/public/shared/chats")


def _token_response(items: int) -> dict[str, Any]:
    """A block's output: a token pair around a long list."""
    return {
        "type": "block_output",
        "outputs": {
            "response": [
                {
                    "access_token": ACCESS_SECRET,
                    "items": [f"item {n:05d} " + "x" * 90 for n in range(items)],
                    "refresh_token": REFRESH_SECRET,
                }
            ]
        },
    }


async def _stored_preview(output: dict[str, Any], *, digest: bool) -> str:
    """What copilot/tools/base.py stores for a result too large to keep inline."""
    db = AsyncMock()
    db.get_or_create_workspace = AsyncMock(return_value=MagicMock(id="ws-1"))
    with (
        patch("backend.copilot.tools.base.workspace_db", return_value=db),
        patch("backend.copilot.tools.base.WorkspaceManager", return_value=AsyncMock()),
    ):
        return await _persist_and_summarize(
            json.dumps(output), "owner-1", SESSION_ID, "call", digest
        )


def _call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    # Both engines store a call's arguments as a JSON string.
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _row(sequence: int, role: str, **fields: Any) -> PrismaChatMessage:
    return PrismaChatMessage.model_construct(
        id=f"msg-{sequence}",
        sessionId=SESSION_ID,
        createdAt=CREATED,
        sequence=sequence,
        role=role,
        **fields,
    )


async def _stored_rows() -> list[PrismaChatMessage]:
    cut_to_size = StreamToolOutputAvailable(
        toolCallId="call-3",
        toolName="fetch_items",
        output=json.dumps(_token_response(2_000)),
    ).output
    held = HeldCall(
        review_id="copilot-node-gate-run_block:5b7e",
        tool_name="run_block",
        tool_call_id="call-4",
        args={},
    )
    late_result = _result_row(held, json.dumps(_token_response(3)), "approved")
    return [
        _row(0, "user", content="Get a token and list the items."),
        _row(
            1,
            "assistant",
            content="",
            toolCalls=[
                _call(
                    "call-1", "run_block", {"block_id": "http", "input": TOKEN_REQUEST}
                )
            ],
        ),
        _row(
            2,
            "tool",
            toolCallId="call-1",
            content=await _stored_preview(_token_response(400), digest=True),
        ),
        _row(3, "assistant", content="", toolCalls=[_call("call-2", "fetch", {})]),
        _row(
            4,
            "tool",
            toolCallId="call-2",
            content=await _stored_preview(_token_response(1_000), digest=False),
        ),
        _row(5, "assistant", content="", toolCalls=[_call("call-3", "fetch", {})]),
        _row(6, "tool", toolCallId="call-3", content=cut_to_size),
        _row(7, "assistant", content="Here are your items."),
        # An approved call's result, stored by the gate as a user row.
        _row(8, "user", content=late_result.content, metadata=late_result.metadata),
    ]


def _shared_session(rows: list[PrismaChatMessage]) -> PrismaChatSession:
    return PrismaChatSession.model_construct(
        id=SESSION_ID,
        userId="owner-1",
        createdAt=CREATED,
        updatedAt=CREATED,
        credentials={},
        successfulAgentRuns={},
        successfulAgentSchedules={},
        metadata={},
        totalPromptTokens=0,
        totalCompletionTokens=0,
        chatStatus="idle",
        isPinned=False,
        isShared=True,
        shareToken=SHARE_TOKEN,
        sharedAt=CREATED,
        # Newest first, the order the page query reads them in.
        Messages=list(reversed(rows)),
    )


@pytest.mark.asyncio
async def test_a_shared_chat_shows_its_tool_activity_without_the_secrets():
    rows = await _stored_rows()
    assert HEADER_SECRET in json.dumps(rows[1].toolCalls)
    for row in (rows[2], rows[4], rows[6], rows[8]):
        assert row.content is not None
        assert ACCESS_SECRET in row.content and REFRESH_SECRET in row.content

    with patch.object(PrismaChatSession, "prisma") as prisma:
        prisma.return_value.find_first = AsyncMock(return_value=_shared_session(rows))
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://s"
        ) as client:
            response = await client.get(
                f"/api/public/shared/chats/{SHARE_TOKEN}/messages"
            )

    assert response.status_code == 200
    for secret in SECRETS:
        assert secret not in response.text
    messages = response.json()["messages"]
    contents = [message["content"] for message in messages]
    assert contents[0] == "Get a token and list the items."
    assert WITHHELD_PREVIEW.fullmatch(contents[2])
    assert WITHHELD_PREVIEW.fullmatch(contents[4])
    assert contents[6] == "[redacted]"
    assert contents[7] == "Here are your items."
    late_output = contents[8].split(">\n", 1)[1].removesuffix("\n</held_call_result>")
    assert json.loads(late_output)["outputs"]["response"][0]["access_token"] == (
        "[redacted]"
    )
    arguments = json.loads(messages[1]["tool_calls"][0]["function"]["arguments"])
    assert arguments == {
        "block_id": "http",
        "input": {
            "url": TOKEN_REQUEST["url"],
            "headers": {"Authorization": "[redacted]"},
        },
    }
