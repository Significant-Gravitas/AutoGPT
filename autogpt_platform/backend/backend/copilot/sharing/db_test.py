"""Unit tests for chat-sharing data-layer cascade logic.

These tests pin down the multi-chat cascade behavior — fixed in
PR #13081 round 1.  Without coverage here, a future refactor could
silently re-introduce the bug where chat A's revoke breaks chat B's
drill-in into a shared execution.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.enums import SharedVia
from prisma.models import AgentGraphExecution, ChatLinkedShare
from prisma.models import ChatSession as PrismaChatSession
from prisma.models import SharedChatFile

from backend.copilot.sharing.db import (
    _collect_execution_ids_from_messages,
    disable_chat_session_share,
    enable_chat_session_share,
    find_linked_executions_in_session,
    get_chat_share_state,
)

SESSION_ID = "sess-A"
OTHER_SESSION_ID = "sess-B"
USER_ID = "user-1"
EXECUTION_ID = "exec-1"


def _mock_session() -> PrismaChatSession:
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
    ):
        yield {
            "session": session_prisma,
            "linked": linked_prisma,
            "execution": exec_prisma,
            "file": file_prisma,
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


class TestEnableChatShareState:
    """Smoke tests for enable_chat_session_share + helpers."""

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

            token = await enable_chat_session_share(SESSION_ID, USER_ID, [])

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


def _mock_tool_msg(content: str) -> "PrismaChatMessage":
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
        with patch(
            "backend.copilot.sharing.db.PrismaChatMessage"
        ) as msg_mock:
            msg_mock.prisma.return_value.find_many = AsyncMock(return_value=[msg])
            ids = await _collect_execution_ids_from_messages(session_id=SESSION_ID)
        assert ids == {"exec-A"}

    @pytest.mark.asyncio
    async def test_dedupes_executions_across_multiple_tool_messages(self):
        """The same execution referenced twice → single entry."""
        m1 = _mock_tool_msg(
            '{"type":"execution_started","execution_id":"exec-A"}'
        )
        m2 = _mock_tool_msg(
            '{"type":"execution_started","execution_id":"exec-A"}'
        )
        m3 = _mock_tool_msg(
            '{"type":"execution_started","execution_id":"exec-B"}'
        )
        with patch(
            "backend.copilot.sharing.db.PrismaChatMessage"
        ) as msg_mock:
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
        with patch(
            "backend.copilot.sharing.db.PrismaChatMessage"
        ) as msg_mock:
            msg_mock.prisma.return_value.find_many = AsyncMock(return_value=[msg])
            ids = await _collect_execution_ids_from_messages(session_id=SESSION_ID)
        assert ids == set()

    @pytest.mark.asyncio
    async def test_ignores_malformed_json_tool_content(self):
        """Tool content that isn't valid JSON shouldn't crash the scan."""
        msg = _mock_tool_msg("not-json-at-all {{")
        with patch(
            "backend.copilot.sharing.db.PrismaChatMessage"
        ) as msg_mock:
            msg_mock.prisma.return_value.find_many = AsyncMock(return_value=[msg])
            ids = await _collect_execution_ids_from_messages(session_id=SESSION_ID)
        assert ids == set()

    @pytest.mark.asyncio
    async def test_ignores_execution_started_with_missing_id(self):
        """Defense-in-depth — a malformed payload without execution_id."""
        msg = _mock_tool_msg('{"type":"execution_started"}')
        with patch(
            "backend.copilot.sharing.db.PrismaChatMessage"
        ) as msg_mock:
            msg_mock.prisma.return_value.find_many = AsyncMock(return_value=[msg])
            ids = await _collect_execution_ids_from_messages(session_id=SESSION_ID)
        assert ids == set()


class TestFindLinkedExecutionsInSession:
    """Higher-level: from session → SharedChatLinkedExecution list."""

    @pytest.mark.asyncio
    async def test_returns_empty_for_non_owner(
        self, mock_prisma_calls
    ):
        """Non-owner / unknown session → empty list (uniform shape)."""
        mock_prisma_calls["session"].return_value.find_first = AsyncMock(
            return_value=None
        )

        result = await find_linked_executions_in_session(
            session_id=SESSION_ID, user_id="other-user"
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_tool_executions_in_messages(
        self, mock_prisma_calls
    ):
        """Owner with chat but no run_agent calls → empty list."""
        mock_prisma_calls["session"].return_value.find_first = AsyncMock(
            return_value=_mock_session()
        )
        with patch("backend.copilot.sharing.db.PrismaChatMessage") as msg_mock:
            msg_mock.prisma.return_value.find_many = AsyncMock(return_value=[])
            result = await find_linked_executions_in_session(
                session_id=SESSION_ID, user_id=USER_ID
            )
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_linked_executions_with_share_token(
        self, mock_prisma_calls
    ):
        """Chat that ran a SHARED execution → drill-in link surfaces."""
        from prisma.models import AgentGraph

        mock_prisma_calls["session"].return_value.find_first = AsyncMock(
            return_value=_mock_session()
        )
        shared_exec = _mock_execution()
        shared_exec.AgentGraph = AgentGraph.model_construct(
            id="graph-1", version=1, name="My Agent", userId=USER_ID
        )
        mock_prisma_calls["execution"].return_value.find_many = AsyncMock(
            return_value=[shared_exec]
        )

        with patch("backend.copilot.sharing.db.PrismaChatMessage") as msg_mock:
            msg_mock.prisma.return_value.find_many = AsyncMock(
                return_value=[
                    _mock_tool_msg(
                        f'{{"type":"execution_started","execution_id":"{EXECUTION_ID}"}}'
                    )
                ]
            )
            result = await find_linked_executions_in_session(
                session_id=SESSION_ID, user_id=USER_ID
            )

        assert len(result) == 1
        assert result[0].execution_id == EXECUTION_ID
        assert result[0].graph_name == "My Agent"
        assert result[0].share_token == "token-exec"  # was isShared=True

    @pytest.mark.asyncio
    async def test_returns_unshared_executions_with_none_token(
        self, mock_prisma_calls
    ):
        """Unshared execution shows up in the modal as opt-in candidate."""
        from prisma.enums import AgentExecutionStatus

        mock_prisma_calls["session"].return_value.find_first = AsyncMock(
            return_value=_mock_session()
        )
        # Build an unshared execution.
        from prisma.models import AgentGraphExecution as Exec

        unshared = Exec.model_construct(
            id=EXECUTION_ID,
            createdAt=datetime.now(UTC),
            agentGraphId="graph-1",
            agentGraphVersion=1,
            executionStatus=AgentExecutionStatus.COMPLETED,
            userId=USER_ID,
            isDeleted=False,
            isShared=False,
            shareToken=None,
            sharedAt=None,
            sharedVia=None,
            AgentGraph=None,
        )
        mock_prisma_calls["execution"].return_value.find_many = AsyncMock(
            return_value=[unshared]
        )

        with patch("backend.copilot.sharing.db.PrismaChatMessage") as msg_mock:
            msg_mock.prisma.return_value.find_many = AsyncMock(
                return_value=[
                    _mock_tool_msg(
                        f'{{"type":"execution_started","execution_id":"{EXECUTION_ID}"}}'
                    )
                ]
            )
            result = await find_linked_executions_in_session(
                session_id=SESSION_ID, user_id=USER_ID
            )

        assert len(result) == 1
        # Not yet shared → share_token=None so the modal renders the
        # "include in share" toggle for it.
        assert result[0].share_token is None
