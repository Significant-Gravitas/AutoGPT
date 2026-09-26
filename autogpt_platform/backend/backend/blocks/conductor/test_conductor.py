"""Unit tests for the Conductor client and the action blocks.

Complements the ``test_mock`` harness in each block (exercised by
``test_available_blocks``) with direct coverage of the client's URL, paging
and error handling, and the blocks' input validation and output wiring.
Transcript parsing and the wait loop live in ``test_transcript.py``; the
read/list blocks in ``test_listing_blocks.py``.
"""

from typing import Any
from unittest import mock

import pytest

from backend.blocks.conductor._api import (
    API_V0,
    CONDUCTOR_API_URL,
    ConductorClient,
    clean,
)
from backend.blocks.conductor.create_workspace import ConductorCreateWorkspaceBlock
from backend.blocks.conductor.manage_workspace import ConductorManageWorkspaceBlock
from backend.blocks.conductor.search import ConductorSearchTranscriptsBlock
from backend.blocks.conductor.send_message import ConductorSendMessageBlock
from backend.blocks.conductor.test_fixtures import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    FakeResponse,
    client_with,
    collect,
    mock_block,
)
from backend.util.exceptions import BlockExecutionError, BlockInputError

# --- client ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_me_is_served_from_the_api_root():
    client, request = client_with(FakeResponse(200, {"userId": "u1"}))
    assert await client.get_me() == {"userId": "u1"}
    assert request.call_args.args == ("GET", f"{CONDUCTOR_API_URL}/me")


@pytest.mark.asyncio
async def test_list_workspaces_sends_repeated_state_params():
    client, request = client_with(FakeResponse(200, {"data": [], "hasMore": False}))
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
async def test_workspace_sessions_passes_pagination():
    client, request = client_with(FakeResponse(200, {"data": [], "hasMore": False}))
    await client.workspace_sessions("ws_1", True, limit=25, offset=50)
    assert request.call_args.args == ("GET", f"{API_V0}/workspaces/ws_1/sessions")
    assert request.call_args.kwargs["params"] == [
        ("includeArchived", "true"),
        ("limit", 25),
        ("offset", 50),
    ]


@pytest.mark.asyncio
async def test_api_error_surfaces_user_message():
    client, _ = client_with(
        FakeResponse(401, {"userMessage": "requests require an authenticated user"})
    )
    with pytest.raises(ValueError, match=r"HTTP 401.*authenticated user"):
        await client.sql("SELECT 1")


@pytest.mark.asyncio
async def test_api_error_without_json_body_reports_status():
    client, _ = client_with(FakeResponse(502, ValueError("not json")))
    with pytest.raises(ValueError, match="HTTP 502"):
        await client.get_workspace("ws_1")


def test_client_retries_are_bounded():
    client = ConductorClient(TEST_CREDENTIALS)
    assert client.requests.retry_max_attempts is not None
    assert client.requests.retry_max_attempts <= 5
    assert client.requests.retry_max_wait <= 30


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


# --- search (R1-08) -----------------------------------------------------------


def test_search_schema_documents_the_transcript_view():
    description = ConductorSearchTranscriptsBlock.Input.model_fields[
        "query"
    ].description
    assert description is not None
    assert "session_transcripts_view" in description
    assert "transcript" in description
    assert "FROM messages" not in description
    assert "LIMIT" in description


# --- blocks ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_workspace_requires_exactly_one_target():
    block = ConductorCreateWorkspaceBlock()
    base = {"credentials": TEST_CREDENTIALS_INPUT, "message": "hi"}
    with pytest.raises(BlockInputError, match="exactly one"):
        await collect(block, base)
    with pytest.raises(BlockInputError, match="exactly one"):
        await collect(
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

    mock_block(block, {"_create": create})
    outputs = await collect(
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
    mock_block(
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
    outputs = await collect(
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
async def test_send_message_surfaces_truncation():
    block = ConductorSendMessageBlock()
    mock_block(
        block,
        {
            "_send": lambda *a, **k: {
                "messageId": "m1",
                "state": "sent",
                "deepLink": "d",
            },
            "_wait": lambda *a, **k: {
                "session_status": "idle",
                "error_message": "",
                "messages": [],
                "reply": "partial",
                "timed_out": False,
                "truncated": True,
            },
        },
    )
    outputs = await collect(
        block,
        {"credentials": TEST_CREDENTIALS_INPUT, "session_id": "s1", "message": "go"},
    )
    assert outputs["truncated"] is True
    assert outputs["reply"] == "partial"


@pytest.mark.asyncio
async def test_send_message_wraps_api_failures():
    block = ConductorSendMessageBlock()

    def fail(*args, **kwargs):
        raise ValueError("Conductor API error (HTTP 404): no such session")

    mock_block(block, {"_send": fail})
    with pytest.raises(BlockExecutionError, match="HTTP 404"):
        await collect(
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
        await collect(
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
