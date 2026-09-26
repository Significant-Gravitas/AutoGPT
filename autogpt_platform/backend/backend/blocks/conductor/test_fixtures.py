"""Shared fakes and fixtures for the Conductor block tests.

Transcript fixtures mirror the live envelope shapes (sanitized): the API
wraps every row as ``{id, sessionIndex, type, content, receivedAt}`` where
``type`` is ``userMessage`` or ``agent`` and agent rows carry the harness's
raw event in ``content.rawPayload`` plus ``turnId``/``userMessageId``. The
receipt returned by *send message* is the user row's ``content.id``, not its
row ``id``, so it is not a valid ``after`` cursor.

Named ``test_*`` so the block loader skips it; it defines no tests itself.
"""

import inspect
from typing import Any
from unittest import mock

from pydantic import SecretStr

from backend.blocks.conductor._api import PAGE_SIZE, ConductorClient
from backend.data.execution import ExecutionContext
from backend.sdk import APIKeyCredentials

TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="conductor",
    api_key=SecretStr("mock-conductor-api-key"),
    title="Mock Conductor API key",
    expires_at=None,
)
TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}
RECEIPT = "acb3061e-0000-4000-8000-000000000001"
OLD_TURN = "11111111-0000-4000-8000-000000000000"


class FakeResponse:
    def __init__(self, status: int, body: Any):
        self.status = status
        self.ok = status < 400
        self._body = body

    def json(self) -> Any:
        if isinstance(self._body, Exception):
            raise self._body
        return self._body

    def text(self) -> str:
        return str(self._body)


def mock_block(block, mocks: dict):
    for name, mock_fn in mocks.items():
        original = getattr(block, name)
        if inspect.iscoroutinefunction(original):

            async def async_mock(*args, _fn=mock_fn, **kwargs):
                return _fn(*args, **kwargs)

            setattr(block, name, async_mock)
        else:
            setattr(block, name, mock_fn)


async def collect(block, input_data: dict) -> dict:
    outputs = {}
    async for name, value in block.execute(
        input_data,
        credentials=TEST_CREDENTIALS,
        execution_context=ExecutionContext(),
    ):
        outputs[name] = value
    return outputs


def client_with(response: FakeResponse) -> tuple[ConductorClient, mock.AsyncMock]:
    client = ConductorClient(TEST_CREDENTIALS)
    request = mock.AsyncMock(return_value=response)
    client.requests.request = request
    return client, request


# --- transcript fixtures -----------------------------------------------------


def user_row(row_id: str, receipt: str, index: int, turn: str = "") -> dict:
    return {
        "id": row_id,
        "sessionId": "s1",
        "sessionIndex": index,
        "type": "userMessage",
        "content": {
            "type": "userMessage",
            "id": receipt,
            "turnId": turn or receipt,
            "message": "Run the tests",
            "state": "sent",
        },
        "receivedAt": "2026-09-26T00:00:00Z",
    }


def agent_row(row_id: str, turn: str, index: int, raw: dict) -> dict:
    return {
        "id": row_id,
        "sessionId": "s1",
        "sessionIndex": index,
        "type": "agent",
        "content": {
            "type": "agent",
            "rawPayload": raw,
            "userMessageId": turn,
            "turnId": turn,
        },
        "receivedAt": "2026-09-26T00:00:00Z",
    }


def claude_text(text: str) -> dict:
    return {
        "type": "assistant",
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
    }


CLAUDE_THINKING = {
    "type": "assistant",
    "message": {
        "role": "assistant",
        "content": [{"type": "thinking", "thinking": "private reasoning"}],
    },
}
CLAUDE_TOOL_USE = {
    "type": "assistant",
    "message": {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {}}],
    },
}
CLAUDE_TOOL_RESULT = {
    "type": "user",
    "message": {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}],
    },
}
CLAUDE_SYSTEM = {"type": "system", "subtype": "init", "session_id": "x"}
CLAUDE_LIFECYCLE = {"type": "command_lifecycle", "state": "started"}
CLAUDE_TOOL_PROGRESS = {"type": "tool_progress", "tool_name": "Bash"}


def claude_result(text: str) -> dict:
    return {"type": "result", "subtype": "success", "is_error": False, "result": text}


def codex_event(kind: str, item: dict | None = None) -> dict:
    event: dict[str, Any] = {"type": kind}
    if item is not None:
        event["item"] = item
    return {"event": event, "thread_id": "thread_1"}


def claude_turn(receipt: str, start_index: int, answer: str) -> list[dict]:
    """A realistic Claude turn: lifecycle, interim text, a tool call, then the
    answer followed by trailing status/result events."""
    return [
        agent_row("r-sys", receipt, start_index + 1, CLAUDE_SYSTEM),
        agent_row("r-life", receipt, start_index + 2, CLAUDE_LIFECYCLE),
        agent_row("r-think", receipt, start_index + 3, CLAUDE_THINKING),
        agent_row("r-interim", receipt, start_index + 4, claude_text("Reading.")),
        agent_row("r-tool", receipt, start_index + 5, CLAUDE_TOOL_USE),
        agent_row("r-progress", receipt, start_index + 6, CLAUDE_TOOL_PROGRESS),
        agent_row("r-result", receipt, start_index + 7, CLAUDE_TOOL_RESULT),
        agent_row("r-answer", receipt, start_index + 8, claude_text(answer)),
        agent_row("r-final", receipt, start_index + 9, claude_result(answer)),
        agent_row("r-idle", receipt, start_index + 10, CLAUDE_LIFECYCLE),
    ]


class FakeTranscript:
    """In-memory stand-in for GET /sessions/{id}/messages.

    Reproduces the live semantics: ascending order, a hard page cap, a tiny
    default page, `after` as an exclusive row-id cursor that 404s for
    unknown ids, and `after` + `offset` rejected together.
    """

    def __init__(self, rows: list[dict], default_limit: int = 5):
        self.rows = rows
        self.default_limit = default_limit
        self.calls: list[dict[str, Any]] = []

    async def list_messages(
        self,
        session_id: str,
        after: str = "",
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        self.calls.append({"after": after, "limit": limit, "offset": offset})
        if after and offset is not None:
            raise ValueError("Conductor API error (HTTP 400): after and offset")
        size = min(limit or self.default_limit, PAGE_SIZE)
        if after:
            index = next((i for i, r in enumerate(self.rows) if r["id"] == after), None)
            if index is None:
                raise ValueError(
                    "Conductor API error (HTTP 404): Cursor message not found "
                    "in this session"
                )
            start = index + 1
        else:
            start = offset or 0
        data = self.rows[start : start + size]
        return {"data": data, "offset": start, "hasMore": start + size < len(self.rows)}


def wait_client(transcript: FakeTranscript, statuses: list[dict]) -> ConductorClient:
    client = ConductorClient(TEST_CREDENTIALS)
    remaining = list(statuses)

    async def status(session_id: str) -> dict[str, Any]:
        return remaining.pop(0) if len(remaining) > 1 else remaining[0]

    client.session_status = status
    client.list_messages = transcript.list_messages
    return client


def fake_clock(step: float = 1.0):
    """monotonic() that advances `step` per call and sleep() that jumps it."""
    now = [0.0]

    def monotonic():
        now[0] += step
        return now[0]

    async def sleep(seconds):
        now[0] += seconds

    return monotonic, sleep, now
