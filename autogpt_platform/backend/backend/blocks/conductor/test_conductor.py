"""Unit tests for the Conductor blocks.

Complements the ``test_mock`` harness in each block (exercised by
``test_available_blocks``) with direct coverage of the client's URL and
error handling, the wait-for-reply loop and the blocks' input validation.
"""

import inspect
from typing import Any
from unittest import mock

import pytest
from pydantic import SecretStr

from backend.blocks.conductor._api import (
    API_V0,
    CONDUCTOR_API_URL,
    ConductorClient,
    clean,
)
from backend.blocks.conductor._transcript import (
    reply_text,
    wait_for_idle,
    wait_for_reply,
)
from backend.blocks.conductor.create_workspace import ConductorCreateWorkspaceBlock
from backend.blocks.conductor.manage_workspace import ConductorManageWorkspaceBlock
from backend.blocks.conductor.send_message import ConductorSendMessageBlock
from backend.data.execution import ExecutionContext
from backend.sdk import APIKeyCredentials
from backend.util.exceptions import BlockExecutionError, BlockInputError

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


def _mock_block(block, mocks: dict):
    for name, mock_fn in mocks.items():
        original = getattr(block, name)
        if inspect.iscoroutinefunction(original):

            async def async_mock(*args, _fn=mock_fn, **kwargs):
                return _fn(*args, **kwargs)

            setattr(block, name, async_mock)
        else:
            setattr(block, name, mock_fn)


async def _collect(block, input_data: dict) -> dict:
    outputs = {}
    async for name, value in block.execute(
        input_data,
        credentials=TEST_CREDENTIALS,
        execution_context=ExecutionContext(),
    ):
        outputs[name] = value
    return outputs


def _client_with(response: FakeResponse) -> tuple[ConductorClient, mock.AsyncMock]:
    client = ConductorClient(TEST_CREDENTIALS)
    request = mock.AsyncMock(return_value=response)
    client.requests.request = request
    return client, request


# --- client ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_me_is_served_from_the_api_root():
    client, request = _client_with(FakeResponse(200, {"userId": "u1"}))
    assert await client.get_me() == {"userId": "u1"}
    assert request.call_args.args == ("GET", f"{CONDUCTOR_API_URL}/me")


@pytest.mark.asyncio
async def test_list_workspaces_sends_repeated_state_params():
    client, request = _client_with(FakeResponse(200, {"data": [], "hasMore": False}))
    await client.list_workspaces(
        {"state": ["ready", "sleeping"], "includeArchived": True, "name": ""}
    )
    assert request.call_args.args == ("GET", f"{API_V0}/workspaces")
    assert request.call_args.kwargs["params"] == [
        ("state", "ready"),
        ("state", "sleeping"),
        ("includeArchived", "true"),
    ]


@pytest.mark.asyncio
async def test_api_error_surfaces_user_message():
    client, _ = _client_with(
        FakeResponse(401, {"userMessage": "requests require an authenticated user"})
    )
    with pytest.raises(ValueError, match="HTTP 401.*authenticated user"):
        await client.sql("SELECT 1")


@pytest.mark.asyncio
async def test_api_error_without_json_body_reports_status():
    client, _ = _client_with(FakeResponse(502, ValueError("not json")))
    with pytest.raises(ValueError, match="HTTP 502"):
        await client.get_workspace("ws_1")


def test_clean_drops_blank_values_and_unwraps_enums():
    from backend.blocks.conductor._api import ConductorAgent, ConductorEffort

    assert clean(
        {
            "projectId": "p1",
            "branch": "",
            "model": None,
            "agent": ConductorAgent.CODEX,
            "effort": ConductorEffort.DEFAULT,
        }
    ) == {"projectId": "p1", "agent": "codex"}


# --- waiting -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_wait_for_idle_stops_when_the_agent_finishes():
    client = ConductorClient(TEST_CREDENTIALS)
    statuses = iter([{"status": "working"}, {"status": "working"}, {"status": "idle"}])
    client.session_status = mock.AsyncMock(side_effect=lambda _: next(statuses))
    with mock.patch("backend.blocks.conductor._transcript.asyncio.sleep"):
        status, timed_out = await wait_for_idle(client, "s1", 60, 1)
    assert status == {"status": "idle"}
    assert timed_out is False
    assert client.session_status.await_count == 3


@pytest.mark.asyncio
async def test_wait_for_idle_reports_timeout():
    client = ConductorClient(TEST_CREDENTIALS)
    client.session_status = mock.AsyncMock(return_value={"status": "working"})
    clock = iter([0.0, 5.0, 10.0])
    with (
        mock.patch("backend.blocks.conductor._transcript.asyncio.sleep"),
        mock.patch(
            "backend.blocks.conductor._transcript.time.monotonic",
            side_effect=lambda: next(clock),
        ),
    ):
        status, timed_out = await wait_for_idle(client, "s1", 8, 1)
    assert status == {"status": "working"}
    assert timed_out is True


@pytest.mark.asyncio
async def test_wait_for_reply_returns_agent_text_after_the_prompt():
    client = ConductorClient(TEST_CREDENTIALS)
    client.session_status = mock.AsyncMock(
        return_value={"status": "error", "lastError": "boom"}
    )
    client.list_messages = mock.AsyncMock(
        return_value={
            "data": [
                {"id": "m2", "type": "assistant", "content": [{"text": "Hi"}]},
                {"id": "m3", "type": "user", "content": "ignored"},
                {"id": "m4", "type": "assistant", "content": {"text": "Done"}},
            ]
        }
    )
    with mock.patch("backend.blocks.conductor._transcript.asyncio.sleep"):
        result = await wait_for_reply(client, "s1", "m1", 10, 1)
    assert result["reply"] == "Hi\n\nDone"
    assert result["session_status"] == "error"
    assert result["error_message"] == "boom"
    assert result["timed_out"] is False
    client.list_messages.assert_awaited_once_with("s1", after="m1")


def test_reply_text_skips_user_messages_and_handles_unknown_shapes():
    messages = [
        {"type": "user", "content": "prompt"},
        {"type": "assistant", "content": {"kind": "tool", "name": "bash"}},
        {"type": "assistant", "content": "plain"},
    ]
    assert reply_text(messages) == '{"kind": "tool", "name": "bash"}\n\nplain'


# --- blocks ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_workspace_requires_exactly_one_target():
    block = ConductorCreateWorkspaceBlock()
    base = {"credentials": TEST_CREDENTIALS_INPUT, "message": "hi"}
    with pytest.raises(BlockInputError, match="exactly one"):
        await _collect(block, base)
    with pytest.raises(BlockInputError, match="exactly one"):
        await _collect(
            block,
            {**base, "project_id": "p1", "repository_url": "https://x/y.git"},
        )


@pytest.mark.asyncio
async def test_create_workspace_builds_a_clean_payload():
    block = ConductorCreateWorkspaceBlock()
    seen: dict[str, Any] = {}

    def create(_creds, payload):
        seen.update(payload)
        return {"workspaceId": "ws_1", "sessionId": "s1", "deepLink": "d"}

    _mock_block(block, {"_create": create})
    outputs = await _collect(
        block,
        {
            "credentials": TEST_CREDENTIALS_INPUT,
            "repository_url": "https://github.com/x/y",
            "branch": "main",
            "agent": "codex",
            "fast_mode": True,
            "env": {"FOO": "bar"},
            "restricted_access": True,
            "wait_for_reply": True,
        },
    )
    assert seen == {
        "repositoryUrl": "https://github.com/x/y",
        "branch": "main",
        "agent": "codex",
        "fastMode": True,
        "env": {"FOO": "bar"},
        "access": {"restricted": True},
    }
    assert outputs["initial_message_id"] == ""
    assert "reply" not in outputs


@pytest.mark.asyncio
async def test_send_message_without_waiting_returns_only_the_receipt():
    block = ConductorSendMessageBlock()
    _mock_block(
        block,
        {
            "_send": lambda *a, **k: {
                "messageId": "m1",
                "state": "sent",
                "deepLink": "d",
            },
            "_wait": lambda *a, **k: pytest.fail("should not wait"),
        },
    )
    outputs = await _collect(
        block,
        {
            "credentials": TEST_CREDENTIALS_INPUT,
            "session_id": "s1",
            "message": "go",
            "wait_for_reply": False,
        },
    )
    assert outputs == {"message_id": "m1", "state": "sent", "deep_link": "d"}


@pytest.mark.asyncio
async def test_send_message_wraps_api_failures():
    block = ConductorSendMessageBlock()

    def fail(*args, **kwargs):
        raise ValueError("Conductor API error (HTTP 404): no such session")

    _mock_block(block, {"_send": fail})
    with pytest.raises(BlockExecutionError, match="HTTP 404"):
        await _collect(
            block,
            {
                "credentials": TEST_CREDENTIALS_INPUT,
                "session_id": "s1",
                "message": "go",
            },
        )


@pytest.mark.asyncio
async def test_manage_workspace_rename_requires_a_name():
    block = ConductorManageWorkspaceBlock()
    with pytest.raises(BlockInputError, match="name is required"):
        await _collect(
            block,
            {
                "credentials": TEST_CREDENTIALS_INPUT,
                "workspace_id": "ws_1",
                "action": "rename",
            },
        )


@pytest.mark.asyncio
async def test_manage_workspace_clears_section_with_null():
    client = ConductorClient(TEST_CREDENTIALS)
    client.requests.request = mock.AsyncMock(
        return_value=FakeResponse(200, {"ok": True})
    )
    await client.set_workspace_section("ws_1", None)
    assert client.requests.request.call_args.kwargs["json"] == {"sectionId": None}
