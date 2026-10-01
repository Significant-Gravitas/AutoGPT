"""Tests for the read/list blocks: transcript tail and cursor paging in Get
Session (R1-05), session pagination in Get Workspace (R1-06) and project
filtering in List Workspaces (R1-07).
"""

from typing import Any
from unittest import mock

import pytest

from backend.blocks.conductor.get_session import ConductorGetSessionBlock
from backend.blocks.conductor.get_workspace import ConductorGetWorkspaceBlock
from backend.blocks.conductor.list_workspaces import ConductorListWorkspacesBlock
from backend.blocks.conductor.test_fixtures import (
    CLAUDE_LIFECYCLE,
    CLAUDE_TOOL_USE,
    RECEIPT,
    TEST_CREDENTIALS_INPUT,
    FakeTranscript,
    agent_row,
    claude_result,
    claude_text,
    claude_turn,
    collect,
    user_row,
)
from backend.util.exceptions import BlockInputError

# --- get session (R1-05) ------------------------------------------------------


def _session_client(transcript: FakeTranscript):
    client = mock.Mock()
    client.get_session = mock.AsyncMock(
        return_value={"id": "s1", "deepLink": "conductor://s/1"}
    )
    client.session_status = mock.AsyncMock(return_value={"status": "idle"})
    client.get_message = mock.AsyncMock(return_value={"id": "row-1"})
    client.list_messages = transcript.list_messages
    return client


@pytest.mark.asyncio
async def test_get_session_returns_the_recent_tail_by_default():
    rows = [user_row("row-prompt", RECEIPT, 0)]
    rows += [
        agent_row(f"tool-{i}", RECEIPT, i + 1, CLAUDE_TOOL_USE) for i in range(240)
    ]
    rows += [agent_row("row-answer", RECEIPT, 241, claude_text("The answer"))]
    rows += [agent_row("row-trailing", RECEIPT, 242, claude_result("The answer"))]
    rows += [agent_row("row-idle", RECEIPT, 243, CLAUDE_LIFECYCLE)]
    transcript = FakeTranscript(rows)
    block = ConductorGetSessionBlock()
    with mock.patch(
        "backend.blocks.conductor.get_session.ConductorClient",
        return_value=_session_client(transcript),
    ):
        outputs = await collect(
            block,
            {
                "credentials": TEST_CREDENTIALS_INPUT,
                "session_id": "s1",
                "message_limit": 20,
            },
        )
    ids = [m["id"] for m in outputs["messages"]]
    assert len(ids) == 20
    assert ids[-3:] == ["row-answer", "row-trailing", "row-idle"]
    assert outputs["latest_reply"] == "The answer"
    assert outputs["has_more"] is True
    assert outputs["next_after"] == "row-idle"
    assert len(transcript.calls) < 30
    assert all(c["limit"] is not None for c in transcript.calls)


@pytest.mark.asyncio
async def test_get_session_short_transcript_needs_a_single_request():
    rows = [user_row("row-prompt", RECEIPT, 0), *claude_turn(RECEIPT, 0, "Done")]
    transcript = FakeTranscript(rows)
    block = ConductorGetSessionBlock()
    with mock.patch(
        "backend.blocks.conductor.get_session.ConductorClient",
        return_value=_session_client(transcript),
    ):
        outputs = await collect(
            block, {"credentials": TEST_CREDENTIALS_INPUT, "session_id": "s1"}
        )
    assert [m["id"] for m in outputs["messages"]] == [r["id"] for r in rows]
    assert outputs["latest_reply"] == "Done"
    assert outputs["has_more"] is False
    assert len(transcript.calls) == 1


@pytest.mark.asyncio
async def test_get_session_after_reads_forward_from_the_cursor():
    rows = [user_row("row-prompt", RECEIPT, 0), *claude_turn(RECEIPT, 0, "Done")]
    transcript = FakeTranscript(rows)
    block = ConductorGetSessionBlock()
    with mock.patch(
        "backend.blocks.conductor.get_session.ConductorClient",
        return_value=_session_client(transcript),
    ):
        outputs = await collect(
            block,
            {
                "credentials": TEST_CREDENTIALS_INPUT,
                "session_id": "s1",
                "after": "r-tool",
                "message_limit": 2,
            },
        )
    assert [m["id"] for m in outputs["messages"]] == ["r-progress", "r-result"]
    assert outputs["has_more"] is True
    assert outputs["next_after"] == "r-result"
    assert outputs["latest_reply"] == ""
    assert transcript.calls == [{"after": "r-tool", "limit": 2, "offset": None}]


# --- get workspace (R1-06) ----------------------------------------------------


@pytest.mark.asyncio
async def test_get_workspace_pages_sessions_up_to_the_limit():
    sessions = [{"id": f"sess_{i}", "deepLink": f"conductor://s/{i}"} for i in range(7)]
    calls: list[dict[str, Any]] = []

    async def workspace_sessions(
        workspace_id, include_archived, limit=None, offset=None
    ):
        calls.append(
            {"include_archived": include_archived, "limit": limit, "offset": offset}
        )
        size = min(limit or 5, 3)
        start = offset or 0
        return {
            "data": sessions[start : start + size],
            "offset": start,
            "hasMore": start + size < len(sessions),
        }

    client = mock.Mock()
    client.get_workspace = mock.AsyncMock(return_value={"id": "ws_1", "state": "ready"})
    client.workspace_status = mock.AsyncMock(return_value={"status": "ready"})
    client.get_preview = mock.AsyncMock(return_value={})
    client.workspace_sessions = workspace_sessions
    block = ConductorGetWorkspaceBlock()
    with mock.patch(
        "backend.blocks.conductor.get_workspace.ConductorClient", return_value=client
    ):
        outputs = await collect(
            block,
            {
                "credentials": TEST_CREDENTIALS_INPUT,
                "workspace_id": "ws_1",
                "include_archived_sessions": True,
                "session_limit": 5,
                "session_offset": 1,
            },
        )
    assert [s["id"] for s in outputs["sessions"]] == [f"sess_{i}" for i in range(1, 6)]
    assert outputs["sessions_has_more"] is True
    assert outputs["next_session_offset"] == 6
    assert all(c["include_archived"] is True for c in calls)
    assert [c["offset"] for c in calls] == [1, 4]


# --- list workspaces (R1-07) --------------------------------------------------


def _workspace(**overrides) -> dict:
    base = {
        "id": "ws",
        "projectId": "proj_1",
        "name": "fix-login",
        "state": "ready",
        "repoUrl": "https://github.com/acme/app",
        "createdAt": "2026-09-01T00:00:00Z",
        "deepLink": "conductor://workspace/ws",
        "creatorId": "user_a",
        "lastActivityAt": "2026-09-20T00:00:00Z",
    }
    return {**base, **overrides}


@pytest.mark.asyncio
async def test_list_workspaces_applies_every_filter_to_project_listings():
    page = [
        _workspace(id="match"),
        _workspace(id="other-creator", creatorId="user_b"),
        _workspace(id="too-old", lastActivityAt="2026-08-01T00:00:00Z"),
        _workspace(id="other-repo", repoUrl="https://github.com/acme/other"),
        _workspace(id="wrong-state", state="sleeping"),
        _workspace(id="wrong-name", name="add-feature"),
        _workspace(id="archived", state="archived"),
    ]
    client = mock.Mock()
    client.list_workspaces = mock.AsyncMock(
        return_value={"data": page, "offset": 10, "hasMore": True}
    )
    block = ConductorListWorkspacesBlock()
    with mock.patch(
        "backend.blocks.conductor.list_workspaces.ConductorClient", return_value=client
    ):
        outputs = await collect(
            block,
            {
                "credentials": TEST_CREDENTIALS_INPUT,
                "project_id": "proj_1",
                "state": ["ready"],
                "name": "LOGIN",
                "repo": "acme/app",
                "creator": "user_a",
                "since": "2026-09-15",
                "offset": 10,
                "limit": 7,
            },
        )
    assert [w["id"] for w in outputs["workspaces"]] == ["match"]
    assert outputs["workspace"]["id"] == "match"
    assert outputs["has_more"] is True
    assert outputs["next_offset"] == 17
    params = client.list_workspaces.call_args.args[0]
    assert "includeArchived" not in params
    assert params["limit"] == 7 and params["offset"] == 10
    assert client.list_workspaces.call_args.args[1] == "proj_1"


@pytest.mark.asyncio
async def test_list_workspaces_project_listing_hides_archived_unless_asked():
    page = [_workspace(id="live"), _workspace(id="gone", state="archived")]
    client = mock.Mock()
    client.list_workspaces = mock.AsyncMock(
        return_value={"data": page, "offset": 0, "hasMore": False}
    )
    block = ConductorListWorkspacesBlock()
    with mock.patch(
        "backend.blocks.conductor.list_workspaces.ConductorClient", return_value=client
    ):
        hidden = await collect(
            block, {"credentials": TEST_CREDENTIALS_INPUT, "project_id": "proj_1"}
        )
        shown = await collect(
            block,
            {
                "credentials": TEST_CREDENTIALS_INPUT,
                "project_id": "proj_1",
                "include_archived": True,
            },
        )
        explicit = await collect(
            block,
            {
                "credentials": TEST_CREDENTIALS_INPUT,
                "project_id": "proj_1",
                "state": ["archived"],
            },
        )
    assert [w["id"] for w in hidden["workspaces"]] == ["live"]
    assert [w["id"] for w in shown["workspaces"]] == ["live", "gone"]
    assert [w["id"] for w in explicit["workspaces"]] == ["gone"]


@pytest.mark.asyncio
async def test_list_workspaces_rejects_an_invalid_since_before_requesting():
    client = mock.Mock()
    client.list_workspaces = mock.AsyncMock()
    block = ConductorListWorkspacesBlock()
    with (
        mock.patch(
            "backend.blocks.conductor.list_workspaces.ConductorClient",
            return_value=client,
        ),
        pytest.raises(BlockInputError, match="since"),
    ):
        await collect(
            block,
            {
                "credentials": TEST_CREDENTIALS_INPUT,
                "project_id": "proj_1",
                "since": "last tuesday",
            },
        )
    client.list_workspaces.assert_not_called()
