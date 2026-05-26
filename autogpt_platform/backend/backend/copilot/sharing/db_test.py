"""Unit tests for chat-sharing data-layer cascade logic.

These tests pin down the multi-chat cascade behavior — fixed in
PR #13081 round 1.  Without coverage here, a future refactor could
silently re-introduce the bug where chat A's revoke breaks chat B's
drill-in into a shared execution.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.enums import SharedVia
from prisma.models import AgentGraphExecution, ChatLinkedShare
from prisma.models import ChatMessage as PrismaChatMessage
from prisma.models import ChatSession as PrismaChatSession
from prisma.models import (
    SharedChatFile,
    SharedExecutionFile,
    UserWorkspace,
    UserWorkspaceFile,
)

from backend.blocks._base import BlockType
from backend.copilot.sharing.db import (
    _build_shared_execution_file_allowlist,
    _collect_execution_ids_from_messages,
    _file_referenced_in_session,
    disable_chat_session_share,
    enable_chat_session_share,
    get_chat_share_state,
    link_new_execution_to_chat_share,
)

SESSION_ID = "sess-A"
OTHER_SESSION_ID = "sess-B"
USER_ID = "user-1"
EXECUTION_ID = "exec-1"


def _mock_session(*, auto_share_executions: bool = False) -> PrismaChatSession:
    now = datetime.now(UTC)
    return PrismaChatSession.model_construct(
        id=SESSION_ID,
        createdAt=now,
        updatedAt=now,
        userId=USER_ID,
        credentials={},
        successfulAgentRuns={},
        successfulAgentSchedules={},
        totalPromptTokens=0,
        totalCompletionTokens=0,
        metadata={},
        chatStatus="idle",
        isShared=True,
        shareToken="token-A",
        sharedAt=now,
        autoShareExecutions=auto_share_executions,
    )


def _mock_execution(
    *, shared_via: SharedVia | None = SharedVia.CHAT_LINK
) -> AgentGraphExecution:
    now = datetime.now(UTC)
    return AgentGraphExecution.model_construct(
        id=EXECUTION_ID,
        createdAt=now,
        agentGraphId="graph-1",
        agentGraphVersion=1,
        executionStatus="COMPLETED",
        userId=USER_ID,
        isDeleted=False,
        isShared=True,
        shareToken="token-exec",
        sharedAt=now,
        sharedVia=shared_via,
    )


def _mock_linked_share() -> ChatLinkedShare:
    return ChatLinkedShare.model_construct(
        id="link-1",
        createdAt=datetime.now(UTC),
        sessionId=SESSION_ID,
        executionId=EXECUTION_ID,
        Execution=_mock_execution(),
    )


class _TxStub:
    """Async context-manager that yields a dummy tx token.

    The real Prisma transaction context manager does the same — yields
    a tx handle that gets threaded into ``.prisma(tx)`` calls.  For
    these tests we just need a sentinel object the calls can accept.
    """

    def __init__(self) -> None:
        self.tx = MagicMock(name="tx")

    async def __aenter__(self):
        return self.tx

    async def __aexit__(self, *args):
        return None


@pytest.fixture()
def mock_transaction():
    with patch("backend.copilot.sharing.db.transaction", return_value=_TxStub()) as m:
        yield m


@pytest.fixture()
def mock_prisma_calls():
    """Patch every Prisma model call used by disable_chat_session_share.

    Each model's ``.prisma()`` (and ``.prisma(tx)``) returns the same
    mock so we can assert against a single call surface regardless of
    whether the call was inside or outside the transaction.
    """
    with (
        patch.object(PrismaChatSession, "prisma") as session_prisma,
        patch.object(ChatLinkedShare, "prisma") as linked_prisma,
        patch.object(AgentGraphExecution, "prisma") as exec_prisma,
        patch.object(SharedChatFile, "prisma") as file_prisma,
        patch.object(SharedExecutionFile, "prisma") as exec_file_prisma,
        patch.object(PrismaChatMessage, "prisma") as msg_prisma,
        patch.object(UserWorkspace, "prisma") as workspace_prisma,
        patch.object(UserWorkspaceFile, "prisma") as workspace_file_prisma,
    ):
        msg_prisma.return_value.count = AsyncMock(return_value=0)
        msg_prisma.return_value.find_many = AsyncMock(return_value=[])
        workspace_prisma.return_value.find_unique = AsyncMock(return_value=None)
        workspace_file_prisma.return_value.find_many = AsyncMock(return_value=[])
        # Default no-op for the execution-file allowlist so tests that
        # don't exercise the file-build / cascade-cleanup paths still see
        # an awaitable.  Tests that DO exercise those paths override.
        exec_file_prisma.return_value.delete_many = AsyncMock(return_value=0)
        exec_file_prisma.return_value.create = AsyncMock()
        # ``_build_shared_execution_file_allowlist`` calls find_unique
        # on AgentGraphExecution to pull NodeExecutions.Output for the
        # workspace-file scan.  Default to None so the helper short-
        # circuits unless a test explicitly sets up node-execution data.
        exec_prisma.return_value.find_unique = AsyncMock(return_value=None)
        # The re-share path calls ``_cascade_revoke_chat_linked_executions``
        # which reads ChatLinkedShare.find_many before any other linked
        # ops fire.  Default to empty so the cascade is a no-op for
        # tests that don't exercise the previous-generation cleanup.
        linked_prisma.return_value.find_many = AsyncMock(return_value=[])
        yield {
            "session": session_prisma,
            "linked": linked_prisma,
            "execution": exec_prisma,
            "file": file_prisma,
            "exec_file": exec_file_prisma,
            "message": msg_prisma,
            "workspace": workspace_prisma,
            "workspace_file": workspace_file_prisma,
        }


class TestDisableCascade:
    """Multi-chat cascade rules on disable_chat_session_share."""

    @pytest.mark.asyncio
    async def test_revokes_chat_link_execution_when_no_other_chat_references_it(
        self, mock_prisma_calls, mock_transaction
    ):
        """Execution shared only via this chat → revoke."""
        # Session lookup succeeds.
        mock_prisma_calls["session"].return_value.find_first = AsyncMock(
            return_value=_mock_session()
        )
        # One linked execution for this session.
        mock_prisma_calls["linked"].return_value.find_many = AsyncMock(
            return_value=[_mock_linked_share()]
        )
        # No OTHER chat session has a linkage for this execution.
        mock_prisma_calls["linked"].return_value.find_first = AsyncMock(
            return_value=None
        )
        mock_prisma_calls["linked"].return_value.delete_many = AsyncMock(return_value=1)
        mock_prisma_calls["execution"].return_value.update = AsyncMock()
        mock_prisma_calls["file"].return_value.delete_many = AsyncMock(return_value=0)
        mock_prisma_calls["session"].return_value.update = AsyncMock()

        await disable_chat_session_share(SESSION_ID, USER_ID)

        # Execution was revoked.
        mock_prisma_calls["execution"].return_value.update.assert_called_once()
        update_data = mock_prisma_calls[
            "execution"
        ].return_value.update.call_args.kwargs["data"]
        assert update_data == {
            "isShared": False,
            "shareToken": None,
            "sharedAt": None,
            "sharedVia": None,
        }

    @pytest.mark.asyncio
    async def test_preserves_execution_when_another_chat_still_references_it(
        self, mock_prisma_calls, mock_transaction
    ):
        """Multi-chat reference → leave execution shared.  Regression for the
        bug fixed in PR #13081 round 1: chat A's revoke must not silently
        break chat B's drill-in link to the same execution."""
        mock_prisma_calls["session"].return_value.find_first = AsyncMock(
            return_value=_mock_session()
        )
        mock_prisma_calls["linked"].return_value.find_many = AsyncMock(
            return_value=[_mock_linked_share()]
        )
        # ANOTHER chat session's linkage exists for the same execution.
        other_link = ChatLinkedShare.model_construct(
            id="link-2",
            createdAt=datetime.now(UTC),
            sessionId=OTHER_SESSION_ID,
            executionId=EXECUTION_ID,
        )
        mock_prisma_calls["linked"].return_value.find_first = AsyncMock(
            return_value=other_link
        )
        mock_prisma_calls["linked"].return_value.delete_many = AsyncMock(return_value=1)
        mock_prisma_calls["execution"].return_value.update = AsyncMock()
        mock_prisma_calls["file"].return_value.delete_many = AsyncMock(return_value=0)
        mock_prisma_calls["session"].return_value.update = AsyncMock()

        await disable_chat_session_share(SESSION_ID, USER_ID)

        # CRITICAL: execution was NOT revoked because chat B still depends on it.
        mock_prisma_calls["execution"].return_value.update.assert_not_called()
        mock_prisma_calls["linked"].return_value.find_first.assert_awaited_once_with(
            where={
                "executionId": EXECUTION_ID,
                "sessionId": {"not": SESSION_ID},
            },
        )

    @pytest.mark.asyncio
    async def test_preserves_user_shared_execution_even_with_no_other_links(
        self, mock_prisma_calls, mock_transaction
    ):
        """USER-shared execution: cascade must skip regardless of linkage."""
        user_shared_link = ChatLinkedShare.model_construct(
            id="link-1",
            createdAt=datetime.now(UTC),
            sessionId=SESSION_ID,
            executionId=EXECUTION_ID,
            Execution=_mock_execution(shared_via=SharedVia.USER),
        )
        mock_prisma_calls["session"].return_value.find_first = AsyncMock(
            return_value=_mock_session()
        )
        mock_prisma_calls["linked"].return_value.find_many = AsyncMock(
            return_value=[user_shared_link]
        )
        # Even if no other linkage exists, USER-shared execution stays untouched.
        mock_prisma_calls["linked"].return_value.find_first = AsyncMock(
            return_value=None
        )
        mock_prisma_calls["linked"].return_value.delete_many = AsyncMock(return_value=1)
        mock_prisma_calls["execution"].return_value.update = AsyncMock()
        mock_prisma_calls["file"].return_value.delete_many = AsyncMock(return_value=0)
        mock_prisma_calls["session"].return_value.update = AsyncMock()

        await disable_chat_session_share(SESSION_ID, USER_ID)

        mock_prisma_calls["execution"].return_value.update.assert_not_called()

    @pytest.mark.asyncio
    async def test_raises_when_session_not_owned_by_user(
        self, mock_prisma_calls, mock_transaction
    ):
        """Non-owner attempt → ValueError, no writes."""
        mock_prisma_calls["session"].return_value.find_first = AsyncMock(
            return_value=None
        )

        with pytest.raises(ValueError, match="not found for user"):
            await disable_chat_session_share(SESSION_ID, "different-user")

        mock_prisma_calls["linked"].return_value.delete_many.assert_not_called()
        mock_prisma_calls["session"].return_value.update.assert_not_called()

    @pytest.mark.asyncio
    async def test_cascade_revoke_deletes_execution_file_allowlist(
        self, mock_prisma_calls, mock_transaction
    ):
        """Revoking a chat that flipped an execution to CHAT_LINK must
        also drop the execution's ``SharedExecutionFile`` allowlist.  A
        stale allowlist row would still authorise public file downloads
        for a leaked pre-revoke URL even though the share token field
        is cleared on the execution."""
        mock_prisma_calls["session"].return_value.find_first = AsyncMock(
            return_value=_mock_session()
        )
        mock_prisma_calls["linked"].return_value.find_many = AsyncMock(
            return_value=[_mock_linked_share()]
        )
        mock_prisma_calls["linked"].return_value.find_first = AsyncMock(
            return_value=None
        )
        mock_prisma_calls["linked"].return_value.delete_many = AsyncMock(return_value=1)
        mock_prisma_calls["execution"].return_value.update = AsyncMock()
        mock_prisma_calls["file"].return_value.delete_many = AsyncMock(return_value=0)
        mock_prisma_calls["session"].return_value.update = AsyncMock()
        exec_file_delete = AsyncMock(return_value=2)
        mock_prisma_calls["exec_file"].return_value.delete_many = exec_file_delete

        await disable_chat_session_share(SESSION_ID, USER_ID)

        exec_file_delete.assert_called_once()
        assert exec_file_delete.call_args.kwargs["where"] == {
            "executionId": EXECUTION_ID
        }

    @pytest.mark.asyncio
    async def test_cascade_skips_exec_file_delete_when_another_chat_references_execution(
        self, mock_prisma_calls, mock_transaction
    ):
        """Multi-chat reference → cascade skips the execution entirely
        (including its file allowlist).  Without this the second chat's
        public viewer would break when the first chat is revoked."""
        mock_prisma_calls["session"].return_value.find_first = AsyncMock(
            return_value=_mock_session()
        )
        mock_prisma_calls["linked"].return_value.find_many = AsyncMock(
            return_value=[_mock_linked_share()]
        )
        other_link = ChatLinkedShare.model_construct(
            id="link-2",
            createdAt=datetime.now(UTC),
            sessionId=OTHER_SESSION_ID,
            executionId=EXECUTION_ID,
        )
        mock_prisma_calls["linked"].return_value.find_first = AsyncMock(
            return_value=other_link
        )
        mock_prisma_calls["linked"].return_value.delete_many = AsyncMock(return_value=1)
        mock_prisma_calls["execution"].return_value.update = AsyncMock()
        mock_prisma_calls["file"].return_value.delete_many = AsyncMock(return_value=0)
        mock_prisma_calls["session"].return_value.update = AsyncMock()
        exec_file_delete = AsyncMock(return_value=0)
        mock_prisma_calls["exec_file"].return_value.delete_many = exec_file_delete

        await disable_chat_session_share(SESSION_ID, USER_ID)

        # Same posture as the execution.update assertion: don't touch
        # the allowlist when another chat still references the execution.
        exec_file_delete.assert_not_called()


class TestEnableChatShareState:
    """Smoke tests for enable_chat_session_share + helpers."""

    @pytest.mark.asyncio
    async def test_raises_value_error_when_session_not_owned_by_user(
        self, mock_prisma_calls
    ):
        """A user can't enable sharing on a chat they don't own — both
        a security guarantee AND the shape callers expect for the 404
        path in the share route."""
        mock_prisma_calls["session"].return_value.find_first = AsyncMock(
            return_value=None
        )

        with pytest.raises(ValueError, match="not found for user"):
            await enable_chat_session_share(
                SESSION_ID, "other-user", auto_share_executions=False
            )

    @pytest.mark.asyncio
    async def test_reshare_cascades_old_chat_link_executions_before_rekey(
        self, mock_prisma_calls, mock_transaction
    ):
        """Re-sharing a chat (e.g. toggling auto-share off without
        Stop-sharing first) must cascade-revoke the previous generation
        of CHAT_LINK execution shares.  Without this, the old per-exec
        tokens stay publicly active with no linkage row to reach them,
        leaving orphan public shares.
        """
        prev_link = _mock_linked_share()  # CHAT_LINK execution with token-exec
        mock_prisma_calls["session"].return_value.find_first = AsyncMock(
            return_value=_mock_session()
        )
        # First find_many is the cascade pre-cleanup → return the
        # previous generation's linkage.  Subsequent calls (none in
        # this enable path) would default to the fixture empty.
        mock_prisma_calls["linked"].return_value.find_many = AsyncMock(
            return_value=[prev_link]
        )
        # No other chat still references the execution.
        mock_prisma_calls["linked"].return_value.find_first = AsyncMock(
            return_value=None
        )
        mock_prisma_calls["linked"].return_value.delete_many = AsyncMock(return_value=1)
        exec_update = AsyncMock()
        mock_prisma_calls["execution"].return_value.update = exec_update
        mock_prisma_calls["execution"].return_value.find_many = AsyncMock(
            return_value=[]
        )
        mock_prisma_calls["file"].return_value.delete_many = AsyncMock(return_value=0)
        with patch("backend.copilot.sharing.db.PrismaChatMessage") as msg_mock:
            msg_mock.prisma.return_value.find_many = AsyncMock(return_value=[])
            mock_prisma_calls["session"].return_value.update = AsyncMock()
            await enable_chat_session_share(
                SESSION_ID, USER_ID, auto_share_executions=False
            )

        # Critical: the previous-generation CHAT_LINK execution was
        # revoked BEFORE the join-table delete that would have orphaned
        # it.  Without the cascade the token stays live forever.
        exec_update.assert_called_once()
        update_data = exec_update.call_args.kwargs["data"]
        assert update_data == {
            "isShared": False,
            "shareToken": None,
            "sharedAt": None,
            "sharedVia": None,
        }

    @pytest.mark.asyncio
    async def test_enable_session_update_is_last_inside_transaction(
        self, mock_prisma_calls, mock_transaction
    ):
        """Session update must happen LAST so a crash before it can't
        leave the chat publicly readable with an empty file allowlist."""
        mock_prisma_calls["session"].return_value.find_first = AsyncMock(
            return_value=_mock_session()
        )
        # No executions to validate / link.
        mock_prisma_calls["execution"].return_value.find_many = AsyncMock(
            return_value=[]
        )
        mock_prisma_calls["file"].return_value.delete_many = AsyncMock(return_value=0)
        mock_prisma_calls["linked"].return_value.delete_many = AsyncMock(return_value=0)
        # _build_shared_chat_files needs to find no messages → no files.
        with patch("backend.copilot.sharing.db.PrismaChatMessage") as msg_mock:
            msg_mock.prisma.return_value.find_many = AsyncMock(return_value=[])
            mock_prisma_calls["session"].return_value.update = AsyncMock()

            token = await enable_chat_session_share(
                SESSION_ID, USER_ID, auto_share_executions=False
            )

        # Token returned, session.update called with isShared=True.
        assert isinstance(token, str)
        assert len(token) > 10
        update_call = mock_prisma_calls["session"].return_value.update.call_args
        assert update_call.kwargs["data"]["isShared"] is True
        assert update_call.kwargs["data"]["shareToken"] == token

    @pytest.mark.asyncio
    async def test_get_chat_share_state_returns_default_for_unknown_session(
        self, mock_prisma_calls
    ):
        """Non-existent / non-owned session reports unshared shape uniformly."""
        mock_prisma_calls["session"].return_value.find_first = AsyncMock(
            return_value=None
        )

        state = await get_chat_share_state(SESSION_ID, USER_ID)

        assert state.is_shared is False
        assert state.share_token is None

    @pytest.mark.asyncio
    async def test_get_chat_share_state_returns_token_when_shared(
        self, mock_prisma_calls
    ):
        """Owner of a shared session sees the actual token surfaced."""
        mock_prisma_calls["session"].return_value.find_first = AsyncMock(
            return_value=_mock_session()
        )

        state = await get_chat_share_state(SESSION_ID, USER_ID)

        assert state.is_shared is True
        assert state.share_token == "token-A"


def _mock_tool_msg(content: str):
    """Build a minimal tool-role ChatMessage row for scanner tests."""
    from prisma.models import ChatMessage as PrismaChatMessage

    return PrismaChatMessage.model_construct(
        id="tm-1",
        createdAt=datetime.now(UTC),
        sessionId=SESSION_ID,
        role="tool",
        content=content,
        sequence=1,
        toolCalls=None,
        functionCall=None,
        name=None,
        toolCallId="call-1",
        refusal=None,
    )


class TestFileReferencedInSession:
    @pytest.mark.asyncio
    async def test_uses_content_contains_candidate_query_for_workspace_uri(self):
        file_id = "11111111-2222-3333-4444-555555555555"
        msg = _mock_tool_msg(f"![chart](workspace://{file_id}#image/png)")
        with patch("backend.copilot.sharing.db.PrismaChatMessage") as msg_mock:
            find_many = AsyncMock(return_value=[msg])
            msg_mock.prisma.return_value.find_many = find_many

            found = await _file_referenced_in_session(
                session_id=SESSION_ID, file_id=file_id
            )

        assert found is True
        find_many.assert_awaited_once_with(
            where={"sessionId": SESSION_ID, "content": {"contains": file_id}},
        )

    @pytest.mark.asyncio
    async def test_falls_back_to_full_scan_for_json_fields(self):
        file_id = "11111111-2222-3333-4444-555555555555"
        msg = _mock_tool_msg("")
        msg.toolCalls = {"result": f"workspace://{file_id}#text/plain"}
        with patch("backend.copilot.sharing.db.PrismaChatMessage") as msg_mock:
            find_many = AsyncMock(side_effect=[[], [msg]])
            msg_mock.prisma.return_value.find_many = find_many

            found = await _file_referenced_in_session(
                session_id=SESSION_ID, file_id=file_id
            )

        assert found is True
        assert find_many.await_args_list[0].kwargs == {
            "where": {"sessionId": SESSION_ID, "content": {"contains": file_id}},
        }
        assert find_many.await_args_list[1].kwargs == {
            "where": {"sessionId": SESSION_ID},
        }


class TestCollectExecutionIdsFromMessages:
    """Sub-agent discovery: which executions did this chat spawn?"""

    @pytest.mark.asyncio
    async def test_extracts_execution_id_from_execution_started_tool_response(
        self,
    ):
        """Tool responses with ``type=execution_started`` carry the run id."""
        msg = _mock_tool_msg(
            '{"type":"execution_started","execution_id":"exec-A","graph_id":"g1"}'
        )
        with patch("backend.copilot.sharing.db.PrismaChatMessage") as msg_mock:
            msg_mock.prisma.return_value.find_many = AsyncMock(return_value=[msg])
            ids = await _collect_execution_ids_from_messages(session_id=SESSION_ID)
        assert ids == {"exec-A"}

    @pytest.mark.asyncio
    async def test_dedupes_executions_across_multiple_tool_messages(self):
        """The same execution referenced twice → single entry."""
        m1 = _mock_tool_msg('{"type":"execution_started","execution_id":"exec-A"}')
        m2 = _mock_tool_msg('{"type":"execution_started","execution_id":"exec-A"}')
        m3 = _mock_tool_msg('{"type":"execution_started","execution_id":"exec-B"}')
        with patch("backend.copilot.sharing.db.PrismaChatMessage") as msg_mock:
            msg_mock.prisma.return_value.find_many = AsyncMock(
                return_value=[m1, m2, m3]
            )
            ids = await _collect_execution_ids_from_messages(session_id=SESSION_ID)
        assert ids == {"exec-A", "exec-B"}

    @pytest.mark.asyncio
    async def test_ignores_non_execution_started_tool_responses(self):
        """Other tool types must not contribute to the linked-execs set."""
        msg = _mock_tool_msg(
            '{"type":"web_fetch","url":"https://example.com","status_code":200}'
        )
        with patch("backend.copilot.sharing.db.PrismaChatMessage") as msg_mock:
            msg_mock.prisma.return_value.find_many = AsyncMock(return_value=[msg])
            ids = await _collect_execution_ids_from_messages(session_id=SESSION_ID)
        assert ids == set()

    @pytest.mark.asyncio
    async def test_ignores_malformed_json_tool_content(self):
        """Tool content that isn't valid JSON shouldn't crash the scan."""
        msg = _mock_tool_msg("not-json-at-all {{")
        with patch("backend.copilot.sharing.db.PrismaChatMessage") as msg_mock:
            msg_mock.prisma.return_value.find_many = AsyncMock(return_value=[msg])
            ids = await _collect_execution_ids_from_messages(session_id=SESSION_ID)
        assert ids == set()

    @pytest.mark.asyncio
    async def test_ignores_execution_started_with_missing_id(self):
        """Defense-in-depth — a malformed payload without execution_id."""
        msg = _mock_tool_msg('{"type":"execution_started"}')
        with patch("backend.copilot.sharing.db.PrismaChatMessage") as msg_mock:
            msg_mock.prisma.return_value.find_many = AsyncMock(return_value=[msg])
            ids = await _collect_execution_ids_from_messages(session_id=SESSION_ID)
        assert ids == set()

    @pytest.mark.asyncio
    async def test_extracts_execution_id_from_agent_output_response(self):
        """``wait_for_result`` sync-complete runs persist an AgentOutputResponse
        with ``execution_id`` nested under ``execution`` rather than at the
        top level.  Without this branch, sync-complete runs would never be
        auto-linked when the owner re-shares the chat — the exact bug we
        debugged on the QuickThemedGreeter chat."""
        msg = _mock_tool_msg(
            '{"type":"agent_output",'
            '"message":"Agent X completed.",'
            '"execution":{"execution_id":"exec-sync","status":"COMPLETED"}}'
        )
        with patch("backend.copilot.sharing.db.PrismaChatMessage") as msg_mock:
            msg_mock.prisma.return_value.find_many = AsyncMock(return_value=[msg])
            ids = await _collect_execution_ids_from_messages(session_id=SESSION_ID)
        assert ids == {"exec-sync"}

    @pytest.mark.asyncio
    async def test_extracts_execution_id_from_error_response(self):
        """Failed / terminated real executions can be recovered on re-share
        when the runtime hook missed the original link attempt."""
        msg = _mock_tool_msg('{"type":"error","execution_id":"exec-failed"}')
        with patch("backend.copilot.sharing.db.PrismaChatMessage") as msg_mock:
            msg_mock.prisma.return_value.find_many = AsyncMock(return_value=[msg])
            ids = await _collect_execution_ids_from_messages(session_id=SESSION_ID)
        assert ids == {"exec-failed"}

    @pytest.mark.asyncio
    async def test_ignores_scheduled_execution_started_response(self):
        """Schedule setup responses carry a schedule id, not a graph execution id."""
        msg = _mock_tool_msg(
            '{"type":"execution_started","execution_id":"schedule-id","status":"SCHEDULED"}'
        )
        with patch("backend.copilot.sharing.db.PrismaChatMessage") as msg_mock:
            msg_mock.prisma.return_value.find_many = AsyncMock(return_value=[msg])
            ids = await _collect_execution_ids_from_messages(session_id=SESSION_ID)
        assert ids == set()

    @pytest.mark.asyncio
    async def test_ignores_agent_output_with_missing_execution_block(self):
        """An agent_output without a nested execution block is a no-op."""
        msg = _mock_tool_msg('{"type":"agent_output","message":"done"}')
        with patch("backend.copilot.sharing.db.PrismaChatMessage") as msg_mock:
            msg_mock.prisma.return_value.find_many = AsyncMock(return_value=[msg])
            ids = await _collect_execution_ids_from_messages(session_id=SESSION_ID)
        assert ids == set()


class TestSharedExecutionFileAllowlist:
    @pytest.mark.asyncio
    async def test_uses_public_output_blocks_not_all_node_outputs(
        self, mock_prisma_calls
    ):
        """CHAT_LINK execution shares should expose only files visible in
        the public execution viewer, matching normal execution shares."""
        hidden_file = "11111111-2222-3333-4444-555555555555"
        public_file = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        hidden_node = SimpleNamespace(
            Node=SimpleNamespace(agentBlockId="standard-block"),
            executionData={
                "result": f"workspace://{hidden_file}#text/csv",
            },
            Input=None,
        )
        output_node = SimpleNamespace(
            Node=SimpleNamespace(agentBlockId="output-block"),
            executionData={
                "name": "Final report",
                "value": f"workspace://{public_file}#text/markdown",
            },
            Input=None,
        )
        execution = SimpleNamespace(NodeExecutions=[hidden_node, output_node])

        mock_prisma_calls["execution"].return_value.find_unique = AsyncMock(
            return_value=execution
        )
        mock_prisma_calls["workspace"].return_value.find_unique = AsyncMock(
            return_value=SimpleNamespace(id="workspace-1")
        )
        mock_prisma_calls["workspace_file"].return_value.find_many = AsyncMock(
            return_value=[SimpleNamespace(id=public_file)]
        )
        mock_prisma_calls["exec_file"].return_value.create = AsyncMock()

        def get_block(block_id: str):
            block_type = (
                BlockType.OUTPUT if block_id == "output-block" else BlockType.STANDARD
            )
            return SimpleNamespace(block_type=block_type)

        with patch("backend.copilot.sharing.db.get_block", side_effect=get_block):
            created = await _build_shared_execution_file_allowlist(
                execution_id=EXECUTION_ID,
                share_token="share-token",
                user_id=USER_ID,
            )

        assert created == 1
        find_many_call = mock_prisma_calls[
            "workspace_file"
        ].return_value.find_many.call_args.kwargs
        assert find_many_call["where"]["id"] == {"in": [public_file]}
        mock_prisma_calls["exec_file"].return_value.create.assert_awaited_once_with(
            data={
                "executionId": EXECUTION_ID,
                "fileId": public_file,
                "shareToken": "share-token",
            }
        )


class TestLinkNewExecutionToChatShare:
    """Runtime hook: new run_agent → auto-link if chat has autoShareExecutions=True."""

    @pytest.mark.asyncio
    async def test_no_op_when_chat_not_shared(self, mock_prisma_calls):
        """An execution in an unshared chat is never auto-linked."""
        unshared = _mock_session()
        unshared.isShared = False
        unshared.autoShareExecutions = False
        mock_prisma_calls["session"].return_value.find_unique = AsyncMock(
            return_value=unshared
        )
        link_create = AsyncMock()
        mock_prisma_calls["linked"].return_value.create = link_create

        await link_new_execution_to_chat_share(
            session_id=SESSION_ID, execution_id=EXECUTION_ID
        )

        link_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_op_when_auto_share_executions_off(self, mock_prisma_calls):
        """Chat shared but the owner opted out of auto-sharing runs."""
        opted_out = _mock_session(auto_share_executions=False)
        mock_prisma_calls["session"].return_value.find_unique = AsyncMock(
            return_value=opted_out
        )
        link_create = AsyncMock()
        mock_prisma_calls["linked"].return_value.create = link_create

        await link_new_execution_to_chat_share(
            session_id=SESSION_ID, execution_id=EXECUTION_ID
        )

        link_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_idempotent_when_link_already_exists(self, mock_prisma_calls):
        """Repeated calls for the same (session, execution) pair don't double-link."""
        mock_prisma_calls["session"].return_value.find_unique = AsyncMock(
            return_value=_mock_session(auto_share_executions=True)
        )
        mock_prisma_calls["execution"].return_value.find_first = AsyncMock(
            return_value=_mock_execution()
        )
        # Existing ChatLinkedShare row for this (session, execution).
        mock_prisma_calls["linked"].return_value.find_first = AsyncMock(
            return_value=ChatLinkedShare.model_construct(
                sessionId=SESSION_ID, executionId=EXECUTION_ID
            )
        )
        link_create = AsyncMock()
        mock_prisma_calls["linked"].return_value.create = link_create

        await link_new_execution_to_chat_share(
            session_id=SESSION_ID, execution_id=EXECUTION_ID
        )

        link_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_swallows_prisma_failures_without_crashing_caller(
        self, mock_prisma_calls
    ):
        """The hook runs inside the CoPilotExecutor worker; if Prisma is
        unconnected there, the underlying find_unique raises
        ClientNotConnectedError.  The hook must NEVER let that escape —
        crashing run_agent surfaces as "The model returned an empty
        response" via the orphan tool_use SDK path.  Best-effort: a
        sharing failure must not break the underlying tool.
        """
        from prisma.errors import ClientNotConnectedError

        mock_prisma_calls["session"].return_value.find_unique = AsyncMock(
            side_effect=ClientNotConnectedError()
        )
        # Should NOT raise — the hook catches everything.
        await link_new_execution_to_chat_share(
            session_id=SESSION_ID, execution_id=EXECUTION_ID
        )

    @pytest.mark.asyncio
    async def test_links_new_execution_when_auto_share_on(
        self, mock_prisma_calls, mock_transaction
    ):
        """Chat shared with autoShareExecutions=True → new run gets linked."""
        mock_prisma_calls["session"].return_value.find_unique = AsyncMock(
            return_value=_mock_session(auto_share_executions=True)
        )
        mock_prisma_calls["execution"].return_value.find_first = AsyncMock(
            return_value=_mock_execution()
        )
        mock_prisma_calls["linked"].return_value.find_first = AsyncMock(
            return_value=None
        )
        link_create = AsyncMock()
        mock_prisma_calls["linked"].return_value.create = link_create
        # ``_link_executions_to_share`` uses ``update_many`` with an
        # ``isShared: False`` guard.  Mock its return as 0 because the
        # fixture execution is already shared — pinning the DB-conditional
        # behaviour.  See ``test_link_skips_already_shared_execution``
        # below for the inverse.
        update_many = AsyncMock(return_value=0)
        mock_prisma_calls["execution"].return_value.update_many = update_many

        await link_new_execution_to_chat_share(
            session_id=SESSION_ID, execution_id=EXECUTION_ID
        )

        link_create.assert_called_once()
        call_data = link_create.call_args.kwargs["data"]
        assert call_data["sessionId"] == SESSION_ID
        assert call_data["executionId"] == EXECUTION_ID
        # Conditional update guards against stale in-memory ``isShared``
        # — the where clause must include ``isShared: False`` so the
        # update is a no-op when the execution was independently shared.
        update_many.assert_called_once()
        where = update_many.call_args.kwargs["where"]
        assert where["id"] == EXECUTION_ID
        assert where["isShared"] is False

    @pytest.mark.asyncio
    async def test_in_tx_recheck_aborts_when_concurrent_disable_unshared_session(
        self, mock_prisma_calls, mock_transaction
    ):
        """Race defense: outside-tx ``session.find_unique`` saw
        ``isShared=True``, but by the time we open the cascade transaction
        a concurrent ``disable_chat_session_share`` has flipped it to
        ``isShared=False``.  The in-tx re-read must catch that and bail
        BEFORE we create a ``ChatLinkedShare`` row that would point at a
        no-longer-shared parent.
        """
        shared_outside = _mock_session(auto_share_executions=True)
        unshared_in_tx = _mock_session(auto_share_executions=True)
        unshared_in_tx.isShared = False

        # First call (outside the tx) returns the shared snapshot;
        # second call (inside the tx) returns the unshared snapshot.
        mock_prisma_calls["session"].return_value.find_unique = AsyncMock(
            side_effect=[shared_outside, unshared_in_tx]
        )
        link_create = AsyncMock()
        mock_prisma_calls["linked"].return_value.create = link_create
        update_many = AsyncMock(return_value=0)
        mock_prisma_calls["execution"].return_value.update_many = update_many

        await link_new_execution_to_chat_share(
            session_id=SESSION_ID, execution_id=EXECUTION_ID
        )

        # Critical: no linkage row was created because the in-tx
        # re-check fired the early-return branch.
        link_create.assert_not_called()
        update_many.assert_not_called()

    @pytest.mark.asyncio
    async def test_link_flips_unshared_execution_to_chat_link_provenance(
        self, mock_prisma_calls, mock_transaction
    ):
        """Brand-new execution (isShared=False at insert) → conditional
        update flips it to CHAT_LINK with a freshly minted token.  The
        DB-level conditional (``isShared: False`` in the where clause)
        protects against the race where another transaction has just
        shared the execution between the prefetch and the update."""
        unshared_exec = _mock_execution()
        unshared_exec.isShared = False
        unshared_exec.shareToken = None
        unshared_exec.sharedAt = None
        unshared_exec.sharedVia = None

        mock_prisma_calls["session"].return_value.find_unique = AsyncMock(
            return_value=_mock_session(auto_share_executions=True)
        )
        mock_prisma_calls["execution"].return_value.find_first = AsyncMock(
            return_value=unshared_exec
        )
        mock_prisma_calls["linked"].return_value.find_first = AsyncMock(
            return_value=None
        )
        mock_prisma_calls["linked"].return_value.create = AsyncMock()
        # Conditional update succeeds — the DB still reports isShared=False.
        update_many = AsyncMock(return_value=1)
        mock_prisma_calls["execution"].return_value.update_many = update_many

        await link_new_execution_to_chat_share(
            session_id=SESSION_ID, execution_id=EXECUTION_ID
        )

        update_many.assert_called_once()
        call = update_many.call_args.kwargs
        assert call["where"] == {"id": EXECUTION_ID, "isShared": False}
        assert call["data"]["isShared"] is True
        assert call["data"]["sharedVia"] == SharedVia.CHAT_LINK
        assert call["data"]["shareToken"] is not None
        assert call["data"]["sharedAt"] is not None
