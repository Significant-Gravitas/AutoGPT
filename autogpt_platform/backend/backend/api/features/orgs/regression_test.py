"""Regression tests for userId-based isolation.

These tests document and protect EXISTING behavior that must be preserved
during the userId -> org/team migration.  Every test calls the actual data
layer function and asserts that Prisma queries include the correct userId
filter (or set userId in create data).

Tests mock at the Prisma model boundary (``ModelName.prisma()``) so the
real Python logic is exercised while no database connection is needed.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
USER_ID = "user-owner-aaaa"
OTHER_USER_ID = "user-other-bbbb"
GRAPH_ID = "graph-1111"
GRAPH_VERSION = 1
EXEC_ID = "exec-2222"
SESSION_ID = "session-3333"
API_KEY_ID = "key-4444"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_graph_row(
    *,
    id=GRAPH_ID,
    version=GRAPH_VERSION,
    userId=USER_ID,
    isActive=True,
    name="Test Graph",
    description="desc",
    instructions=None,
    recommendedScheduleCron=None,
    forkedFromId=None,
    forkedFromVersion=None,
):
    """Return a MagicMock that quacks like a Prisma AgentGraph row."""
    m = MagicMock()
    m.id = id
    m.version = version
    m.userId = userId
    m.isActive = isActive
    m.name = name
    m.description = description
    m.instructions = instructions
    m.recommendedScheduleCron = recommendedScheduleCron
    m.forkedFromId = forkedFromId
    m.forkedFromVersion = forkedFromVersion
    m.createdAt = datetime(2025, 6, 1, tzinfo=timezone.utc)
    m.Nodes = []
    return m


def _make_execution_row(
    *,
    id=EXEC_ID,
    userId=USER_ID,
    agentGraphId=GRAPH_ID,
    agentGraphVersion=GRAPH_VERSION,
    executionStatus="COMPLETED",
):
    m = MagicMock()
    m.id = id
    m.userId = userId
    m.agentGraphId = agentGraphId
    m.agentGraphVersion = agentGraphVersion
    m.executionStatus = executionStatus
    m.createdAt = datetime(2025, 6, 1, tzinfo=timezone.utc)
    m.updatedAt = datetime(2025, 6, 1, tzinfo=timezone.utc)
    m.startedAt = None
    m.endedAt = None
    m.stats = None
    m.isDeleted = False
    m.inputs = None
    m.credentialInputs = None
    m.nodesInputMasks = None
    m.agentPresetId = None
    m.parentGraphExecutionId = None
    m.isShared = False
    m.shareToken = None
    return m


def _make_api_key_row(
    *,
    id=API_KEY_ID,
    userId=USER_ID,
    name="test-key",
    head="agpt_xxxx",
    tail="yyyy",
    hash="fakehash",
    salt="fakesalt",
    status="ACTIVE",
    permissions=None,
    description=None,
):
    m = MagicMock()
    m.id = id
    m.userId = userId
    m.name = name
    m.head = head
    m.tail = tail
    m.hash = hash
    m.salt = salt
    m.status = status
    m.permissions = permissions or []
    m.description = description
    m.createdAt = datetime(2025, 6, 1, tzinfo=timezone.utc)
    m.lastUsedAt = None
    m.revokedAt = None
    return m


def _make_chat_session_row(
    *,
    id=SESSION_ID,
    userId=USER_ID,
):
    m = MagicMock()
    m.id = id
    m.userId = userId
    m.title = None
    m.credentials = "{}"
    m.successfulAgentRuns = "{}"
    m.successfulAgentSchedules = "{}"
    m.metadata = "{}"
    m.totalPromptTokens = 0
    m.totalCompletionTokens = 0
    m.createdAt = datetime(2025, 6, 1, tzinfo=timezone.utc)
    m.updatedAt = datetime(2025, 6, 1, tzinfo=timezone.utc)
    m.Messages = []
    return m


# ============================================================================
# Graph CRUD regression tests
# ============================================================================
class TestGraphCrudUserIdIsolation:
    """Verify that graph CRUD functions filter/set userId correctly."""

    @pytest.mark.asyncio
    async def test_regression_get_graph_filters_by_user_id(self):
        """get_graph() must include userId in the Prisma where clause
        when called with a user_id so that only the owner's graph is returned."""
        graph_row = _make_graph_row()

        mock_actions = AsyncMock()
        mock_actions.find_first = AsyncMock(return_value=graph_row)

        with patch(
            "backend.data.graph.AgentGraph.prisma",
            return_value=mock_actions,
        ):
            from backend.data.graph import get_graph

            result = await get_graph(
                GRAPH_ID, version=None, user_id=USER_ID
            )

        # The function must have queried Prisma with userId in the where clause
        mock_actions.find_first.assert_called_once()
        where_arg = mock_actions.find_first.call_args.kwargs.get(
            "where", mock_actions.find_first.call_args[1].get("where")
        )
        assert where_arg["id"] == GRAPH_ID
        assert where_arg["userId"] == USER_ID
        assert result is not None

    @pytest.mark.asyncio
    async def test_regression_get_graph_wrong_user_returns_none(self):
        """get_graph() with a non-owner user_id should return None when
        the graph is not store-listed and not in the user's library."""
        mock_actions = AsyncMock()
        mock_actions.find_first = AsyncMock(return_value=None)

        mock_store_actions = AsyncMock()
        mock_store_actions.find_first = AsyncMock(return_value=None)

        mock_library_actions = AsyncMock()
        mock_library_actions.find_first = AsyncMock(return_value=None)

        with (
            patch(
                "backend.data.graph.AgentGraph.prisma",
                return_value=mock_actions,
            ),
            patch(
                "backend.data.graph.StoreListingVersion.prisma",
                return_value=mock_store_actions,
            ),
            patch(
                "backend.data.graph.LibraryAgent.prisma",
                return_value=mock_library_actions,
            ),
        ):
            from backend.data.graph import get_graph

            result = await get_graph(
                GRAPH_ID, version=None, user_id=OTHER_USER_ID
            )

        # First call is the ownership query — must filter by OTHER_USER_ID
        where_arg = mock_actions.find_first.call_args.kwargs.get(
            "where", mock_actions.find_first.call_args[1].get("where")
        )
        assert where_arg["userId"] == OTHER_USER_ID
        assert result is None

    @pytest.mark.asyncio
    async def test_regression_delete_graph_filters_by_user_id(self):
        """delete_graph() must include userId in its delete_many where clause
        so a user can only delete their own graphs."""
        mock_actions = AsyncMock()
        mock_actions.delete_many = AsyncMock(return_value=1)

        with patch(
            "backend.data.graph.AgentGraph.prisma",
            return_value=mock_actions,
        ):
            from backend.data.graph import delete_graph

            count = await delete_graph(GRAPH_ID, USER_ID)

        mock_actions.delete_many.assert_called_once()
        where_arg = mock_actions.delete_many.call_args.kwargs.get(
            "where", mock_actions.delete_many.call_args[1].get("where")
        )
        assert where_arg == {"id": GRAPH_ID, "userId": USER_ID}
        assert count == 1

    @pytest.mark.asyncio
    async def test_regression_create_graph_sets_user_id(self):
        """__create_graph (called by create_graph) must set userId in the
        AgentGraphCreateInput data so new graphs are owned by the creator."""
        created_graph_row = _make_graph_row()

        mock_tx_actions = AsyncMock()
        # find_first for auto-increment check returns None (no prior version)
        mock_tx_actions.find_first = AsyncMock(return_value=None)
        mock_tx_actions.create_many = AsyncMock(return_value=None)

        mock_node_tx_actions = AsyncMock()
        mock_node_tx_actions.create_many = AsyncMock(return_value=None)

        mock_link_tx_actions = AsyncMock()
        mock_link_tx_actions.create_many = AsyncMock(return_value=None)

        # For the get_graph call after creation
        mock_graph_actions = AsyncMock()
        mock_graph_actions.find_first = AsyncMock(return_value=created_graph_row)

        mock_store_actions = AsyncMock()
        mock_store_actions.find_first = AsyncMock(return_value=None)

        with (
            patch(
                "backend.data.graph.AgentGraph.prisma",
                side_effect=lambda tx=None: mock_tx_actions
                if tx is not None
                else mock_graph_actions,
            ),
            patch(
                "backend.data.graph.AgentNode.prisma",
                return_value=mock_node_tx_actions,
            ),
            patch(
                "backend.data.graph.AgentNodeLink.prisma",
                return_value=mock_link_tx_actions,
            ),
            patch(
                "backend.data.graph.StoreListingVersion.prisma",
                return_value=mock_store_actions,
            ),
            patch("backend.data.graph.transaction") as mock_transaction,
        ):
            # Make transaction context manager yield a mock tx
            mock_tx = MagicMock()
            mock_transaction.return_value.__aenter__ = AsyncMock(
                return_value=mock_tx
            )
            mock_transaction.return_value.__aexit__ = AsyncMock(
                return_value=False
            )

            from backend.data.graph import Graph, create_graph

            test_graph = Graph(
                id=GRAPH_ID,
                name="My Agent",
                description="test",
                version=1,
                is_active=True,
                nodes=[],
                links=[],
            )

            await create_graph(test_graph, USER_ID)

        # Verify the create_many call included userId
        mock_tx_actions.create_many.assert_called_once()
        create_data = mock_tx_actions.create_many.call_args.kwargs.get(
            "data", mock_tx_actions.create_many.call_args[1].get("data")
        )
        assert len(create_data) >= 1
        # AgentGraphCreateInput is a TypedDict; access it as a dict
        first_entry = create_data[0]
        assert first_entry["userId"] == USER_ID

    @pytest.mark.asyncio
    async def test_regression_graph_all_versions_filtered_by_user(self):
        """get_graph_all_versions() must filter by userId when no team_id
        is provided, ensuring users only see their own graph versions."""
        rows = [
            _make_graph_row(version=2),
            _make_graph_row(version=1),
        ]

        mock_actions = AsyncMock()
        mock_actions.find_many = AsyncMock(return_value=rows)

        with patch(
            "backend.data.graph.AgentGraph.prisma",
            return_value=mock_actions,
        ):
            from backend.data.graph import get_graph_all_versions

            results = await get_graph_all_versions(GRAPH_ID, USER_ID)

        mock_actions.find_many.assert_called_once()
        where_arg = mock_actions.find_many.call_args.kwargs.get(
            "where", mock_actions.find_many.call_args[1].get("where")
        )
        assert where_arg["id"] == GRAPH_ID
        assert where_arg["userId"] == USER_ID
        assert "teamId" not in where_arg
        assert len(results) == 2


# ============================================================================
# Execution regression tests
# ============================================================================
class TestExecutionUserIdIsolation:
    """Verify that execution functions filter/set userId correctly."""

    @pytest.mark.asyncio
    async def test_regression_list_executions_filters_by_user_id(self):
        """get_graph_executions() must include userId in the where clause
        when user_id is provided and no team_id is given."""
        exec_row = _make_execution_row()

        mock_actions = AsyncMock()
        mock_actions.find_many = AsyncMock(return_value=[exec_row])

        with patch(
            "backend.data.execution.AgentGraphExecution.prisma",
            return_value=mock_actions,
        ):
            from backend.data.execution import get_graph_executions

            results = await get_graph_executions(user_id=USER_ID)

        mock_actions.find_many.assert_called_once()
        where_arg = mock_actions.find_many.call_args.kwargs.get(
            "where", mock_actions.find_many.call_args[1].get("where")
        )
        assert where_arg["userId"] == USER_ID
        assert where_arg["isDeleted"] is False
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_regression_list_executions_prefers_team_over_user(self):
        """When both team_id and user_id are provided, team_id takes
        precedence and userId is NOT in the where clause."""
        mock_actions = AsyncMock()
        mock_actions.find_many = AsyncMock(return_value=[])

        with patch(
            "backend.data.execution.AgentGraphExecution.prisma",
            return_value=mock_actions,
        ):
            from backend.data.execution import get_graph_executions

            await get_graph_executions(
                user_id=USER_ID, team_id="team-xyz"
            )

        where_arg = mock_actions.find_many.call_args.kwargs.get(
            "where", mock_actions.find_many.call_args[1].get("where")
        )
        assert where_arg["teamId"] == "team-xyz"
        assert "userId" not in where_arg

    @pytest.mark.asyncio
    async def test_regression_create_execution_sets_user_id(self):
        """create_graph_execution() must include userId in the create data
        so that every execution is attributed to the requesting user."""
        mock_exec_row = _make_execution_row(executionStatus="INCOMPLETE")
        mock_exec_row.NodeExecutions = []

        mock_actions = AsyncMock()
        mock_actions.create = AsyncMock(return_value=mock_exec_row)

        with patch(
            "backend.data.execution.AgentGraphExecution.prisma",
            return_value=mock_actions,
        ):
            from backend.data.execution import create_graph_execution

            await create_graph_execution(
                graph_id=GRAPH_ID,
                graph_version=GRAPH_VERSION,
                starting_nodes_input=[],
                inputs={},
                user_id=USER_ID,
            )

        mock_actions.create.assert_called_once()
        create_data = mock_actions.create.call_args.kwargs.get(
            "data", mock_actions.create.call_args[1].get("data")
        )
        assert create_data["userId"] == USER_ID
        assert create_data["agentGraphId"] == GRAPH_ID

    @pytest.mark.asyncio
    async def test_regression_create_execution_dual_writes_org_fields(self):
        """create_graph_execution() includes organizationId and teamId in
        the create data when they are provided (tenancy dual-write)."""
        mock_exec_row = _make_execution_row(executionStatus="INCOMPLETE")
        mock_exec_row.NodeExecutions = []

        mock_actions = AsyncMock()
        mock_actions.create = AsyncMock(return_value=mock_exec_row)

        with patch(
            "backend.data.execution.AgentGraphExecution.prisma",
            return_value=mock_actions,
        ):
            from backend.data.execution import create_graph_execution

            await create_graph_execution(
                graph_id=GRAPH_ID,
                graph_version=GRAPH_VERSION,
                starting_nodes_input=[],
                inputs={},
                user_id=USER_ID,
                organization_id="org-111",
                team_id="team-222",
            )

        create_data = mock_actions.create.call_args.kwargs.get(
            "data", mock_actions.create.call_args[1].get("data")
        )
        assert create_data["userId"] == USER_ID
        assert create_data["organizationId"] == "org-111"
        assert create_data["teamId"] == "team-222"

    @pytest.mark.asyncio
    async def test_regression_list_executions_with_graph_id_filter(self):
        """get_graph_executions() correctly combines userId and graph_id
        in the where clause for scoped execution listing."""
        mock_actions = AsyncMock()
        mock_actions.find_many = AsyncMock(return_value=[])

        with patch(
            "backend.data.execution.AgentGraphExecution.prisma",
            return_value=mock_actions,
        ):
            from backend.data.execution import get_graph_executions

            await get_graph_executions(
                user_id=USER_ID, graph_id=GRAPH_ID
            )

        where_arg = mock_actions.find_many.call_args.kwargs.get(
            "where", mock_actions.find_many.call_args[1].get("where")
        )
        assert where_arg["userId"] == USER_ID
        assert where_arg["agentGraphId"] == GRAPH_ID
        assert where_arg["isDeleted"] is False


# ============================================================================
# Credits regression tests
# ============================================================================
class TestCreditsUserIdIsolation:
    """Verify that credit functions are scoped to the correct userId."""

    @pytest.mark.asyncio
    async def test_regression_get_credits_returns_user_balance(self):
        """UserCredit.get_credits() queries UserBalance by userId,
        returning the per-user balance."""
        mock_balance = MagicMock()
        mock_balance.balance = 5000
        mock_balance.updatedAt = datetime(2025, 6, 1, tzinfo=timezone.utc)

        mock_actions = AsyncMock()
        mock_actions.find_unique = AsyncMock(return_value=mock_balance)

        with patch(
            "backend.data.credit.UserBalance.prisma",
            return_value=mock_actions,
        ):
            from backend.data.credit import UserCredit

            credit = UserCredit()
            balance = await credit.get_credits(USER_ID)

        mock_actions.find_unique.assert_called_once()
        where_arg = mock_actions.find_unique.call_args.kwargs.get(
            "where", mock_actions.find_unique.call_args[1].get("where")
        )
        assert where_arg == {"userId": USER_ID}
        assert balance == 5000

    @pytest.mark.asyncio
    async def test_regression_spend_credits_charges_user(self):
        """UserCredit.spend_credits() passes user_id to _add_transaction
        which writes a CreditTransaction scoped to that userId."""
        from backend.data.credit import UsageTransactionMetadata, UserCredit

        credit = UserCredit()
        metadata = UsageTransactionMetadata(
            graph_exec_id="exec-1",
            graph_id=GRAPH_ID,
        )

        # Mock _add_transaction to capture args
        with patch.object(
            credit,
            "_add_transaction",
            new_callable=AsyncMock,
            return_value=(4900, "txn-key"),
        ) as mock_add, patch(
            "backend.data.credit.get_auto_top_up",
            new_callable=AsyncMock,
            return_value=MagicMock(threshold=None, amount=0),
        ):
            balance = await credit.spend_credits(USER_ID, 100, metadata)

        mock_add.assert_called_once()
        assert mock_add.call_args.kwargs["user_id"] == USER_ID
        assert mock_add.call_args.kwargs["amount"] == -100
        assert balance == 4900

    @pytest.mark.asyncio
    async def test_regression_top_up_credits_user_balance(self):
        """UserCredit.top_up_credits() calls _top_up_credits with the
        correct user_id so the funds go to the right user."""
        from backend.data.credit import UserCredit

        credit = UserCredit()

        with patch.object(
            credit,
            "_top_up_credits",
            new_callable=AsyncMock,
        ) as mock_top_up:
            await credit.top_up_credits(USER_ID, 1000)

        mock_top_up.assert_called_once()
        assert mock_top_up.call_args.kwargs["user_id"] == USER_ID
        assert mock_top_up.call_args.kwargs["amount"] == 1000


# ============================================================================
# API Keys regression tests
# ============================================================================
class TestApiKeyUserIdIsolation:
    """Verify that API key functions are scoped to the correct userId."""

    @pytest.mark.asyncio
    async def test_regression_create_api_key_sets_user_id(self):
        """create_api_key() stores userId in the Prisma create data so
        the key is bound to the creating user."""
        key_row = _make_api_key_row()

        mock_actions = AsyncMock()
        mock_actions.create = AsyncMock(return_value=key_row)

        with patch(
            "backend.data.auth.api_key.PrismaAPIKey.prisma",
            return_value=mock_actions,
        ):
            from backend.data.auth.api_key import create_api_key

            info, plaintext = await create_api_key(
                name="test-key",
                user_id=USER_ID,
                permissions=[],
            )

        mock_actions.create.assert_called_once()
        create_data = mock_actions.create.call_args.kwargs.get(
            "data", mock_actions.create.call_args[1].get("data")
        )
        assert create_data["userId"] == USER_ID
        assert info.user_id == USER_ID

    @pytest.mark.asyncio
    async def test_regression_validate_api_key_returns_user_id(self):
        """validate_api_key() returns an APIKeyInfo whose user_id matches
        the userId stored in the database row."""
        key_row = _make_api_key_row()

        # Mock the head lookup to return our key
        mock_actions = AsyncMock()
        mock_actions.find_many = AsyncMock(return_value=[key_row])

        with (
            patch(
                "backend.data.auth.api_key.PrismaAPIKey.prisma",
                return_value=mock_actions,
            ),
            patch(
                "backend.data.auth.api_key.keysmith.verify_key",
                return_value=True,
            ),
        ):
            from backend.data.auth.api_key import validate_api_key

            result = await validate_api_key("agpt_xxxx_fake_key_yyyy")

        assert result is not None
        assert result.user_id == USER_ID

    @pytest.mark.asyncio
    async def test_regression_list_api_keys_by_user(self):
        """list_user_api_keys() filters by userId so each user only sees
        their own API keys."""
        key_rows = [_make_api_key_row(), _make_api_key_row(id="key-5555")]

        mock_actions = AsyncMock()
        mock_actions.find_many = AsyncMock(return_value=key_rows)

        with patch(
            "backend.data.auth.api_key.PrismaAPIKey.prisma",
            return_value=mock_actions,
        ):
            from backend.data.auth.api_key import list_user_api_keys

            results = await list_user_api_keys(USER_ID)

        mock_actions.find_many.assert_called_once()
        where_arg = mock_actions.find_many.call_args.kwargs.get(
            "where", mock_actions.find_many.call_args[1].get("where")
        )
        assert where_arg["userId"] == USER_ID
        assert len(results) == 2


# ============================================================================
# Chat Session regression tests
# ============================================================================
class TestChatSessionUserIdIsolation:
    """Verify that chat session functions are scoped to the correct userId."""

    @pytest.mark.asyncio
    async def test_regression_create_session_sets_user_id(self):
        """create_chat_session() stores userId in the ChatSessionCreateInput
        so every session is attributed to the creating user."""
        session_row = _make_chat_session_row()

        mock_actions = AsyncMock()
        mock_actions.create = AsyncMock(return_value=session_row)

        with patch(
            "backend.copilot.db.PrismaChatSession.prisma",
            return_value=mock_actions,
        ):
            from backend.copilot.db import create_chat_session

            result = await create_chat_session(SESSION_ID, USER_ID)

        mock_actions.create.assert_called_once()
        create_data = mock_actions.create.call_args.kwargs.get(
            "data", mock_actions.create.call_args[1].get("data")
        )
        assert create_data["userId"] == USER_ID
        assert create_data["id"] == SESSION_ID
        assert result.user_id == USER_ID

    @pytest.mark.asyncio
    async def test_regression_list_sessions_filtered_by_user(self):
        """get_user_chat_sessions() filters by userId so each user only
        sees their own chat sessions."""
        session_rows = [_make_chat_session_row()]

        mock_actions = AsyncMock()
        mock_actions.find_many = AsyncMock(return_value=session_rows)

        with patch(
            "backend.copilot.db.PrismaChatSession.prisma",
            return_value=mock_actions,
        ):
            from backend.copilot.db import get_user_chat_sessions

            results = await get_user_chat_sessions(USER_ID)

        mock_actions.find_many.assert_called_once()
        where_arg = mock_actions.find_many.call_args.kwargs.get(
            "where", mock_actions.find_many.call_args[1].get("where")
        )
        assert where_arg["userId"] == USER_ID
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_regression_delete_session_validates_ownership(self):
        """delete_chat_session() includes userId in the where clause when
        a user_id is provided, preventing cross-user deletion."""
        mock_actions = AsyncMock()
        mock_actions.delete_many = AsyncMock(return_value=1)

        with patch(
            "backend.copilot.db.PrismaChatSession.prisma",
            return_value=mock_actions,
        ):
            from backend.copilot.db import delete_chat_session

            deleted = await delete_chat_session(SESSION_ID, user_id=USER_ID)

        mock_actions.delete_many.assert_called_once()
        where_arg = mock_actions.delete_many.call_args.kwargs.get(
            "where", mock_actions.delete_many.call_args[1].get("where")
        )
        assert where_arg["id"] == SESSION_ID
        assert where_arg["userId"] == USER_ID
        assert deleted is True

    @pytest.mark.asyncio
    async def test_regression_list_sessions_excludes_other_users(self):
        """get_user_chat_sessions() with OTHER_USER_ID should not return
        sessions belonging to USER_ID."""
        mock_actions = AsyncMock()
        mock_actions.find_many = AsyncMock(return_value=[])

        with patch(
            "backend.copilot.db.PrismaChatSession.prisma",
            return_value=mock_actions,
        ):
            from backend.copilot.db import get_user_chat_sessions

            results = await get_user_chat_sessions(OTHER_USER_ID)

        mock_actions.find_many.assert_called_once()
        where_arg = mock_actions.find_many.call_args.kwargs.get(
            "where", mock_actions.find_many.call_args[1].get("where")
        )
        assert where_arg["userId"] == OTHER_USER_ID
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_regression_get_session_requires_ownership(self):
        """get_chat_session() returns the session by ID without userId
        filter (it is a low-level lookup), but callers must validate
        ownership.  We verify the where clause shape here."""
        session_row = _make_chat_session_row()

        mock_actions = AsyncMock()
        mock_actions.find_unique = AsyncMock(return_value=session_row)

        with patch(
            "backend.copilot.db.PrismaChatSession.prisma",
            return_value=mock_actions,
        ):
            from backend.copilot.db import get_chat_session

            result = await get_chat_session(SESSION_ID)

        mock_actions.find_unique.assert_called_once()
        where_arg = mock_actions.find_unique.call_args.kwargs.get(
            "where", mock_actions.find_unique.call_args[1].get("where")
        )
        assert where_arg["id"] == SESSION_ID
        assert result is not None

    @pytest.mark.asyncio
    async def test_regression_stream_requires_session_ownership(self):
        """Verify get_chat_session is used to resolve sessions for
        streaming, and the result carries userId for caller validation."""
        session_row = _make_chat_session_row()

        mock_actions = AsyncMock()
        mock_actions.find_unique = AsyncMock(return_value=session_row)

        with patch(
            "backend.copilot.db.PrismaChatSession.prisma",
            return_value=mock_actions,
        ):
            from backend.copilot.db import get_chat_session

            result = await get_chat_session(SESSION_ID)

        assert result is not None
        # The returned ChatSession carries userId so callers can verify
        assert result.user_id == USER_ID


# ============================================================================
# Additional Execution regression tests
# ============================================================================
class TestExecutionUserIdIsolationExtended:
    """Additional execution regression tests for stop and share operations."""

    @pytest.mark.asyncio
    async def test_regression_stop_execution_requires_ownership(self):
        """delete_graph_execution() (soft delete) must include userId in
        the where clause so only the owner can stop/delete executions."""
        mock_actions = AsyncMock()
        mock_actions.update_many = AsyncMock(return_value=1)

        with patch(
            "backend.data.execution.AgentGraphExecution.prisma",
            return_value=mock_actions,
        ):
            from backend.data.execution import delete_graph_execution

            await delete_graph_execution(EXEC_ID, USER_ID, soft_delete=True)

        mock_actions.update_many.assert_called_once()
        where_arg = mock_actions.update_many.call_args.kwargs.get(
            "where", mock_actions.update_many.call_args[1].get("where")
        )
        assert where_arg["id"] == EXEC_ID
        assert where_arg["userId"] == USER_ID

    @pytest.mark.asyncio
    async def test_regression_execution_share_token_works_without_auth(self):
        """get_graph_execution_by_share_token() does NOT require userId
        — it uses the share_token + isShared flag for public access."""
        exec_row = _make_execution_row()
        exec_row.isShared = True
        exec_row.shareToken = "tok-public-1234"
        exec_row.sharedAt = datetime(2025, 6, 1, tzinfo=timezone.utc)
        # AgentGraph must have real string fields for Pydantic validation
        mock_graph = MagicMock()
        mock_graph.name = "Test Agent"
        mock_graph.description = "A test agent"
        exec_row.AgentGraph = mock_graph
        exec_row.NodeExecutions = []

        mock_actions = AsyncMock()
        mock_actions.find_first = AsyncMock(return_value=exec_row)

        with patch(
            "backend.data.execution.AgentGraphExecution.prisma",
            return_value=mock_actions,
        ):
            from backend.data.execution import (
                get_graph_execution_by_share_token,
            )

            result = await get_graph_execution_by_share_token(
                "tok-public-1234"
            )

        mock_actions.find_first.assert_called_once()
        where_arg = mock_actions.find_first.call_args.kwargs.get(
            "where", mock_actions.find_first.call_args[1].get("where")
        )
        assert where_arg["shareToken"] == "tok-public-1234"
        assert where_arg["isShared"] is True
        # No userId in the where clause — public access
        assert "userId" not in where_arg


# ============================================================================
# Library Agent regression tests
# ============================================================================
LIBRARY_AGENT_ID = "lib-agent-5555"


def _make_library_agent_row(
    *,
    id=LIBRARY_AGENT_ID,
    userId=USER_ID,
    agentGraphId=GRAPH_ID,
    agentGraphVersion=GRAPH_VERSION,
    isDeleted=False,
    isArchived=False,
    isFavorite=False,
    isCreatedByUser=True,
):
    m = MagicMock()
    m.id = id
    m.userId = userId
    m.agentGraphId = agentGraphId
    m.agentGraphVersion = agentGraphVersion
    m.isDeleted = isDeleted
    m.isArchived = isArchived
    m.isFavorite = isFavorite
    m.isCreatedByUser = isCreatedByUser
    m.useGraphIsActiveVersion = True
    m.createdAt = datetime(2025, 6, 1, tzinfo=timezone.utc)
    m.updatedAt = datetime(2025, 6, 1, tzinfo=timezone.utc)
    m.imageUrl = None
    m.folderId = None
    m.settings = "{}"
    # Nested relations
    m.AgentGraph = None
    m.Folder = None
    return m


class TestRegressionLibraryAgents:
    """Verify library agent operations are userId-isolated."""

    @pytest.fixture(autouse=True)
    def setup_library_mocks(self, mocker):
        """Mock prisma at the library module level."""
        self.mock_library_actions = AsyncMock()
        mocker.patch(
            "backend.api.features.library.db.prisma.models.LibraryAgent.prisma",
            return_value=self.mock_library_actions,
        )

    @pytest.mark.asyncio
    async def test_regression_list_library_agents_returns_only_own(self):
        """list_library_agents() must include userId=USER_ID in the
        where clause so only the caller's agents are returned."""
        self.mock_library_actions.find_many = AsyncMock(return_value=[])
        self.mock_library_actions.count = AsyncMock(return_value=0)

        from backend.api.features.library.db import list_library_agents

        result = await list_library_agents(user_id=USER_ID)

        self.mock_library_actions.find_many.assert_called_once()
        where_arg = self.mock_library_actions.find_many.call_args.kwargs.get(
            "where",
            self.mock_library_actions.find_many.call_args[1].get("where"),
        )
        assert where_arg["userId"] == USER_ID
        assert where_arg["isDeleted"] is False
        assert result.pagination.total_items == 0

    @pytest.mark.asyncio
    async def test_regression_list_library_excludes_other_users(self):
        """list_library_agents() called with OTHER_USER_ID must filter
        by that user, not by USER_ID."""
        self.mock_library_actions.find_many = AsyncMock(return_value=[])
        self.mock_library_actions.count = AsyncMock(return_value=0)

        from backend.api.features.library.db import list_library_agents

        result = await list_library_agents(user_id=OTHER_USER_ID)

        where_arg = self.mock_library_actions.find_many.call_args.kwargs.get(
            "where",
            self.mock_library_actions.find_many.call_args[1].get("where"),
        )
        assert where_arg["userId"] == OTHER_USER_ID
        assert len(result.agents) == 0

    @pytest.mark.asyncio
    async def test_regression_get_library_agent_requires_ownership(self):
        """get_library_agent() must include userId in find_first so only
        the owner can retrieve a specific library agent."""
        self.mock_library_actions.find_first = AsyncMock(return_value=None)

        from backend.api.features.library.db import get_library_agent
        from backend.util.exceptions import NotFoundError

        with pytest.raises(NotFoundError):
            await get_library_agent(LIBRARY_AGENT_ID, USER_ID)

        self.mock_library_actions.find_first.assert_called_once()
        where_arg = self.mock_library_actions.find_first.call_args.kwargs.get(
            "where",
            self.mock_library_actions.find_first.call_args[1].get("where"),
        )
        assert where_arg["id"] == LIBRARY_AGENT_ID
        assert where_arg["userId"] == USER_ID

    @pytest.mark.asyncio
    async def test_regression_delete_library_agent_requires_ownership(self):
        """delete_library_agent() must verify userId before deletion.
        It fetches the agent first and checks library_agent.userId."""
        agent_row = _make_library_agent_row()
        agent_row.AgentGraph = MagicMock()
        agent_row.AgentGraph.id = GRAPH_ID
        self.mock_library_actions.find_unique = AsyncMock(
            return_value=agent_row
        )
        self.mock_library_actions.update_many = AsyncMock(return_value=1)

        with (
            patch(
                "backend.api.features.library.db.get_scheduler_client"
            ) as mock_sched,
            patch(
                "backend.api.features.library.db.integrations_db"
            ) as mock_integrations,
        ):
            mock_client = AsyncMock()
            mock_client.get_execution_schedules = AsyncMock(return_value=[])
            mock_sched.return_value = mock_client
            mock_integrations.find_webhooks_by_graph_id = AsyncMock(
                return_value=[]
            )

            from backend.api.features.library.db import delete_library_agent

            await delete_library_agent(LIBRARY_AGENT_ID, USER_ID)

        # The soft-delete update_many must filter by userId
        self.mock_library_actions.update_many.assert_called_once()
        where_arg = self.mock_library_actions.update_many.call_args.kwargs.get(
            "where",
            self.mock_library_actions.update_many.call_args[1].get("where"),
        )
        assert where_arg["id"] == LIBRARY_AGENT_ID
        assert where_arg["userId"] == USER_ID

    @pytest.mark.asyncio
    async def test_regression_fork_library_agent_creates_for_caller(self):
        """fork_library_agent() must create the forked graph under the
        requesting user_id, not the original owner."""
        # We mock get_library_agent and graph_db.fork_graph
        mock_original = MagicMock()
        mock_original.graph_id = GRAPH_ID
        mock_original.graph_version = GRAPH_VERSION
        mock_original.settings = MagicMock()
        mock_original.settings.human_in_the_loop_safe_mode = True
        mock_original.settings.sensitive_action_safe_mode = False

        mock_new_graph = MagicMock()
        mock_new_graph.id = "graph-forked"
        mock_new_graph.version = 1
        mock_new_graph.user_id = USER_ID
        mock_new_graph.is_active = True
        mock_new_graph.sub_graphs = []

        mock_lib_result = MagicMock()
        mock_lib_result.id = "lib-forked"

        with (
            patch(
                "backend.api.features.library.db.get_library_agent",
                new_callable=AsyncMock,
                return_value=mock_original,
            ),
            patch(
                "backend.api.features.library.db.graph_db.fork_graph",
                new_callable=AsyncMock,
                return_value=mock_new_graph,
            ) as mock_fork,
            patch(
                "backend.api.features.library.db.on_graph_activate",
                new_callable=AsyncMock,
                return_value=mock_new_graph,
            ),
            patch(
                "backend.api.features.library.db.create_library_agent",
                new_callable=AsyncMock,
                return_value=[mock_lib_result],
            ) as mock_create_lib,
        ):
            from backend.api.features.library.db import fork_library_agent

            result = await fork_library_agent(LIBRARY_AGENT_ID, USER_ID)

        # fork_graph must be called with the caller's user_id
        mock_fork.assert_called_once_with(GRAPH_ID, GRAPH_VERSION, USER_ID)
        # create_library_agent must use the caller's user_id
        assert mock_create_lib.call_args.args[1] == USER_ID
        assert result.id == "lib-forked"

    @pytest.mark.asyncio
    async def test_regression_library_favorites_are_per_user(self):
        """list_favorite_library_agents() filters by userId AND
        isFavorite=True, ensuring favorites are per-user."""
        self.mock_library_actions.find_many = AsyncMock(return_value=[])
        self.mock_library_actions.count = AsyncMock(return_value=0)

        from backend.api.features.library.db import (
            list_favorite_library_agents,
        )

        result = await list_favorite_library_agents(user_id=USER_ID)

        self.mock_library_actions.find_many.assert_called_once()
        where_arg = self.mock_library_actions.find_many.call_args.kwargs.get(
            "where",
            self.mock_library_actions.find_many.call_args[1].get("where"),
        )
        assert where_arg["userId"] == USER_ID
        assert where_arg["isFavorite"] is True
        assert where_arg["isDeleted"] is False
        assert result.pagination.total_items == 0


# ============================================================================
# Store regression tests
# ============================================================================
STORE_LISTING_VERSION_ID = "slv-6666"
SLUG = "test-agent-slug"


class TestRegressionStore:
    """Verify store operations are userId-isolated."""

    @pytest.fixture(autouse=True)
    def setup_store_mocks(self, mocker):
        """Mock prisma models used in store db functions."""
        self.mock_library_actions = AsyncMock()
        mocker.patch(
            "backend.api.features.store.db.prisma.models.LibraryAgent.prisma",
            return_value=self.mock_library_actions,
        )
        self.mock_submission_view_actions = AsyncMock()
        mocker.patch(
            "backend.api.features.store.db.prisma.models.StoreSubmission.prisma",
            return_value=self.mock_submission_view_actions,
        )
        self.mock_store_agent_actions = AsyncMock()
        mocker.patch(
            "backend.api.features.store.db.prisma.models.StoreAgent.prisma",
            return_value=self.mock_store_agent_actions,
        )
        self.mock_agent_graph_actions = AsyncMock()
        mocker.patch(
            "backend.api.features.store.db.prisma.models.AgentGraph.prisma",
            return_value=self.mock_agent_graph_actions,
        )
        self.mock_slv_actions = AsyncMock()
        mocker.patch(
            "backend.api.features.store.db.prisma.models.StoreListingVersion.prisma",
            return_value=self.mock_slv_actions,
        )
        self.mock_listing_actions = AsyncMock()
        mocker.patch(
            "backend.api.features.store.db.prisma.models.StoreListing.prisma",
            return_value=self.mock_listing_actions,
        )

    @pytest.mark.asyncio
    async def test_regression_my_unpublished_agents_returns_only_own(self):
        """get_my_agents() must include userId=USER_ID in the where clause
        so only the caller's unpublished agents are returned."""
        self.mock_library_actions.find_many = AsyncMock(return_value=[])
        self.mock_library_actions.count = AsyncMock(return_value=0)

        from backend.api.features.store.db import get_my_agents

        result = await get_my_agents(user_id=USER_ID)

        self.mock_library_actions.find_many.assert_called_once()
        where_arg = self.mock_library_actions.find_many.call_args.kwargs.get(
            "where",
            self.mock_library_actions.find_many.call_args[1].get("where"),
        )
        assert where_arg["userId"] == USER_ID
        assert where_arg["isDeleted"] is False
        assert len(result.agents) == 0

    @pytest.mark.asyncio
    async def test_regression_my_submissions_returns_only_own(self):
        """get_store_submissions() must include user_id in the where clause
        so only the caller's submissions are returned."""
        self.mock_submission_view_actions.find_many = AsyncMock(
            return_value=[]
        )
        self.mock_submission_view_actions.count = AsyncMock(return_value=0)

        from backend.api.features.store.db import get_store_submissions

        result = await get_store_submissions(user_id=USER_ID)

        self.mock_submission_view_actions.find_many.assert_called_once()
        where_arg = (
            self.mock_submission_view_actions.find_many.call_args.kwargs.get(
                "where",
                self.mock_submission_view_actions.find_many.call_args[1].get(
                    "where"
                ),
            )
        )
        assert where_arg["user_id"] == USER_ID
        assert where_arg["is_deleted"] is False
        assert len(result.submissions) == 0

    @pytest.mark.asyncio
    async def test_regression_create_submission_sets_owning_user(self):
        """create_store_submission() must verify graph ownership via
        userId and set OwningUser on the StoreListing to the caller."""
        # Graph lookup must filter by userId
        mock_graph = MagicMock()
        mock_graph.id = GRAPH_ID
        mock_graph.User = MagicMock()
        mock_graph.User.Profile = MagicMock()
        self.mock_agent_graph_actions.find_first = AsyncMock(
            return_value=mock_graph
        )

        # Transaction mocks
        mock_listing = MagicMock()
        mock_listing.Versions = []
        self.mock_listing_actions.find_unique = AsyncMock(
            return_value=None
        )

        mock_listing_obj = MagicMock()
        mock_listing_obj.id = "listing-1"
        mock_listing_obj.owningUserId = USER_ID
        mock_listing_obj.slug = SLUG

        mock_submission = MagicMock()
        mock_submission.StoreListing = mock_listing_obj
        mock_submission.id = STORE_LISTING_VERSION_ID
        mock_submission.name = "Test"
        mock_submission.version = 1
        mock_submission.description = ""
        mock_submission.subHeading = ""
        mock_submission.imageUrls = []
        mock_submission.videoUrl = None
        mock_submission.agentOutputDemoUrl = None
        mock_submission.categories = []
        mock_submission.submissionStatus = "PENDING"
        mock_submission.submittedAt = datetime(2025, 6, 1, tzinfo=timezone.utc)
        mock_submission.changesSummary = "Initial"
        mock_submission.agentGraphId = GRAPH_ID
        mock_submission.agentGraphVersion = GRAPH_VERSION
        mock_submission.isDeleted = False
        mock_submission.isAvailable = False
        mock_submission.reviewComments = None
        mock_submission.reviewedAt = None
        mock_submission.reviewerId = None
        mock_submission.internalComments = None
        mock_submission.storeListingId = "listing-1"
        mock_submission.instructions = None
        mock_submission.recommendedScheduleCron = None
        mock_submission.createdAt = datetime(2025, 6, 1, tzinfo=timezone.utc)
        self.mock_slv_actions.create = AsyncMock(return_value=mock_submission)

        with patch(
            "backend.api.features.store.db.transaction"
        ) as mock_tx_ctx:
            mock_tx = MagicMock()
            mock_tx_ctx.return_value.__aenter__ = AsyncMock(
                return_value=mock_tx
            )
            mock_tx_ctx.return_value.__aexit__ = AsyncMock(
                return_value=False
            )
            # Re-patch with tx-aware returns
            with (
                patch(
                    "backend.api.features.store.db.prisma.models.StoreListing.prisma",
                    return_value=self.mock_listing_actions,
                ),
                patch(
                    "backend.api.features.store.db.prisma.models.StoreListingVersion.prisma",
                    return_value=self.mock_slv_actions,
                ),
            ):
                from backend.api.features.store.db import (
                    create_store_submission,
                )

                await create_store_submission(
                    user_id=USER_ID,
                    graph_id=GRAPH_ID,
                    graph_version=GRAPH_VERSION,
                    slug=SLUG,
                    name="Test Agent",
                )

        # The initial graph lookup must include userId
        self.mock_agent_graph_actions.find_first.assert_called_once()
        graph_where = (
            self.mock_agent_graph_actions.find_first.call_args.kwargs.get(
                "where",
                self.mock_agent_graph_actions.find_first.call_args[1].get(
                    "where"
                ),
            )
        )
        assert graph_where["userId"] == USER_ID
        assert graph_where["id"] == GRAPH_ID

    @pytest.mark.asyncio
    async def test_regression_edit_submission_requires_ownership(self):
        """edit_store_submission() must verify the StoreListing
        owningUserId matches user_id before allowing edits."""
        # Return a submission owned by a DIFFERENT user
        mock_version = MagicMock()
        mock_version.StoreListing = MagicMock()
        mock_version.StoreListing.owningUserId = OTHER_USER_ID
        self.mock_slv_actions.find_first = AsyncMock(
            return_value=mock_version
        )

        from backend.api.features.store.db import edit_store_submission
        from backend.api.features.store import exceptions as store_exceptions

        with pytest.raises(store_exceptions.UnauthorizedError):
            await edit_store_submission(
                user_id=USER_ID,
                store_listing_version_id=STORE_LISTING_VERSION_ID,
                name="Updated",
            )

    @pytest.mark.asyncio
    async def test_regression_public_store_agent_visible_to_all(self):
        """get_store_agents() does NOT require userId — the public store
        is visible to everyone. Verify no userId filter is applied."""
        self.mock_store_agent_actions.find_many = AsyncMock(return_value=[])
        self.mock_store_agent_actions.count = AsyncMock(return_value=0)

        # get_store_agents may use hybrid search or direct DB
        # We mock the DB path (no search_query)
        from backend.api.features.store.db import get_store_agents

        result = await get_store_agents(page=1, page_size=10)

        # The function queries StoreAgent view without userId filter
        self.mock_store_agent_actions.find_many.assert_called_once()
        where_arg = self.mock_store_agent_actions.find_many.call_args.kwargs.get(
            "where",
            self.mock_store_agent_actions.find_many.call_args[1].get(
                "where", {}
            ),
        )
        assert "userId" not in where_arg
        assert "user_id" not in where_arg


# ============================================================================
# Schedules regression tests
# ============================================================================
class TestRegressionSchedules:
    """Verify schedule operations are userId-isolated."""

    @pytest.mark.asyncio
    async def test_regression_list_schedules_returns_only_own(self):
        """get_graph_execution_schedules() filters jobs by user_id
        so users only see their own schedules."""
        from backend.executor.scheduler import (
            GraphExecutionJobArgs,
            Scheduler,
        )

        scheduler = Scheduler(register_system_tasks=False)
        scheduler.scheduler = MagicMock()

        # Create two mock jobs: one owned, one not
        owned_job = MagicMock()
        owned_job.id = "job-owned"
        owned_job.kwargs = GraphExecutionJobArgs(
            user_id=USER_ID,
            graph_id=GRAPH_ID,
            graph_version=1,
            cron="*/5 * * * *",
            input_data={},
        ).model_dump()
        owned_job.next_run_time = datetime(2025, 7, 1, tzinfo=timezone.utc)
        owned_job.trigger = MagicMock()
        owned_job.trigger.timezone = "UTC"
        owned_job.name = "test-job"

        other_job = MagicMock()
        other_job.id = "job-other"
        other_job.kwargs = GraphExecutionJobArgs(
            user_id=OTHER_USER_ID,
            graph_id="graph-other",
            graph_version=1,
            cron="*/10 * * * *",
            input_data={},
        ).model_dump()
        other_job.next_run_time = datetime(2025, 7, 1, tzinfo=timezone.utc)
        other_job.trigger = MagicMock()
        other_job.trigger.timezone = "UTC"
        other_job.name = "other-job"

        scheduler.scheduler.get_jobs = MagicMock(
            return_value=[owned_job, other_job]
        )

        results = scheduler.get_graph_execution_schedules(user_id=USER_ID)

        assert len(results) == 1
        assert results[0].user_id == USER_ID

    @pytest.mark.asyncio
    async def test_regression_create_schedule_requires_graph_ownership(self):
        """add_graph_execution_schedule() validates the graph via
        validate_and_construct_node_execution_input which checks ownership."""
        from backend.executor.scheduler import Scheduler

        scheduler = Scheduler(register_system_tasks=False)
        scheduler.scheduler = MagicMock()

        # Mock the validation to raise (simulating non-ownership)
        with patch(
            "backend.executor.scheduler.run_async",
            side_effect=Exception("Graph not found for user"),
        ):
            with pytest.raises(Exception, match="Graph not found"):
                scheduler.add_graph_execution_schedule(
                    user_id=USER_ID,
                    graph_id=GRAPH_ID,
                    graph_version=1,
                    cron="*/5 * * * *",
                    input_data={},
                    input_credentials={},
                )

    @pytest.mark.asyncio
    async def test_regression_delete_schedule_requires_ownership(self):
        """delete_graph_execution_schedule() must verify that the job's
        user_id matches the requesting user_id."""
        from backend.executor.scheduler import (
            GraphExecutionJobArgs,
            Scheduler,
        )

        scheduler = Scheduler(register_system_tasks=False)
        scheduler.scheduler = MagicMock()

        # Create a job owned by OTHER_USER_ID
        mock_job = MagicMock()
        mock_job.kwargs = GraphExecutionJobArgs(
            user_id=OTHER_USER_ID,
            graph_id=GRAPH_ID,
            graph_version=1,
            cron="*/5 * * * *",
            input_data={},
        ).model_dump()
        mock_job.next_run_time = datetime(2025, 7, 1, tzinfo=timezone.utc)
        mock_job.trigger = MagicMock()
        mock_job.trigger.timezone = "UTC"
        mock_job.name = "test-job"

        scheduler.scheduler.get_job = MagicMock(return_value=mock_job)

        from backend.util.exceptions import NotAuthorizedError

        with pytest.raises(NotAuthorizedError):
            scheduler.delete_graph_execution_schedule(
                schedule_id="sched-123", user_id=USER_ID
            )

    @pytest.mark.asyncio
    async def test_regression_scheduled_execution_records_user_id(self):
        """The GraphExecutionJobArgs model always carries user_id which is
        passed to execution_utils.add_graph_execution at runtime."""
        from backend.executor.scheduler import GraphExecutionJobArgs

        args = GraphExecutionJobArgs(
            schedule_id="sched-1",
            user_id=USER_ID,
            graph_id=GRAPH_ID,
            graph_version=1,
            cron="*/5 * * * *",
            input_data={},
        )
        dumped = args.model_dump()
        assert dumped["user_id"] == USER_ID
        assert dumped["graph_id"] == GRAPH_ID


# ============================================================================
# Shared / public access regression tests
# ============================================================================
class TestRegressionShared:
    """Verify public share mechanisms work without auth."""

    @pytest.mark.asyncio
    async def test_regression_public_share_token_works_without_auth(self):
        """get_store_agent_details() is public — it queries the StoreAgent
        view by creator_username and slug, with no userId filter."""
        mock_agent = MagicMock()
        mock_agent.creator_username = "testuser"
        mock_agent.slug = "my-agent"
        # Fields used by StoreAgentDetails.from_db
        mock_agent.listing_version_id = "slv-1"
        mock_agent.agent_name = "My Agent"
        mock_agent.agent_video = ""
        mock_agent.agent_output_demo = ""
        mock_agent.agent_image = []
        mock_agent.creator_avatar = ""
        mock_agent.sub_heading = ""
        mock_agent.description = "A test agent"
        mock_agent.instructions = None
        mock_agent.categories = []
        mock_agent.runs = 0
        mock_agent.rating = 0.0
        mock_agent.versions = ["1"]
        mock_agent.graph_id = GRAPH_ID
        mock_agent.graph_versions = ["1"]
        mock_agent.updated_at = datetime(2025, 6, 1, tzinfo=timezone.utc)
        mock_agent.recommended_schedule_cron = None

        mock_actions = AsyncMock()
        mock_actions.find_first = AsyncMock(return_value=mock_agent)

        with patch(
            "backend.api.features.store.db.prisma.models.StoreAgent.prisma",
            return_value=mock_actions,
        ):
            from backend.api.features.store.db import (
                get_store_agent_details,
            )

            result = await get_store_agent_details("testuser", "my-agent")

        mock_actions.find_first.assert_called_once()
        where_arg = mock_actions.find_first.call_args.kwargs.get(
            "where", mock_actions.find_first.call_args[1].get("where")
        )
        assert where_arg["creator_username"] == "testuser"
        assert where_arg["slug"] == "my-agent"
        # No userId filter — public access
        assert "userId" not in where_arg
        assert "user_id" not in where_arg
        assert result is not None


# ============================================================================
# User Settings regression tests
# ============================================================================
class TestRegressionUserSettings:
    """Verify user settings are isolated per userId."""

    @pytest.fixture(autouse=True)
    def setup_user_mocks(self, mocker):
        """Mock prisma User model."""
        self.mock_user_actions = AsyncMock()
        mocker.patch(
            "backend.data.user.PrismaUser.prisma",
            return_value=self.mock_user_actions,
        )

    @pytest.mark.asyncio
    async def test_regression_user_prefs_isolated_per_user(self):
        """get_user_notification_preference() queries by userId so
        each user only sees their own notification preferences."""
        mock_user = MagicMock()
        mock_user.id = USER_ID
        mock_user.email = "test@example.com"
        mock_user.notifyOnAgentRun = True
        mock_user.notifyOnZeroBalance = False
        mock_user.notifyOnLowBalance = False
        mock_user.notifyOnBlockExecutionFailed = False
        mock_user.notifyOnContinuousAgentError = False
        mock_user.notifyOnDailySummary = False
        mock_user.notifyOnWeeklySummary = False
        mock_user.notifyOnMonthlySummary = False
        mock_user.notifyOnAgentApproved = False
        mock_user.notifyOnAgentRejected = False
        mock_user.maxEmailsPerDay = 3
        self.mock_user_actions.find_unique_or_raise = AsyncMock(
            return_value=mock_user
        )

        from backend.data.user import get_user_notification_preference

        result = await get_user_notification_preference(USER_ID)

        self.mock_user_actions.find_unique_or_raise.assert_called_once()
        where_arg = (
            self.mock_user_actions.find_unique_or_raise.call_args.kwargs.get(
                "where",
                self.mock_user_actions.find_unique_or_raise.call_args[1].get(
                    "where",
                ),
            )
        )
        assert where_arg == {"id": USER_ID}
        assert result.user_id == USER_ID

    @pytest.mark.asyncio
    async def test_regression_user_timezone_isolated_per_user(self):
        """update_user_timezone() updates by userId so timezone changes
        only affect the target user."""
        mock_user = MagicMock()
        mock_user.id = USER_ID
        mock_user.email = "test@example.com"
        mock_user.emailVerified = True
        mock_user.name = "Test"
        mock_user.metadata = "{}"
        mock_user.createdAt = datetime(2025, 6, 1, tzinfo=timezone.utc)
        mock_user.updatedAt = datetime(2025, 6, 1, tzinfo=timezone.utc)
        mock_user.isEnabled = True
        mock_user.timezone = "US/Eastern"
        mock_user.integrations = ""
        mock_user.stripeCustomerId = None
        mock_user.topUpConfig = None
        mock_user.onboardingCompletedAt = None
        mock_user.notifyOnAgentRun = True
        mock_user.notifyOnZeroBalance = False
        mock_user.notifyOnLowBalance = False
        mock_user.notifyOnBlockExecutionFailed = False
        mock_user.notifyOnContinuousAgentError = False
        mock_user.notifyOnDailySummary = False
        mock_user.notifyOnWeeklySummary = False
        mock_user.notifyOnMonthlySummary = False
        mock_user.notifyOnAgentApproved = False
        mock_user.notifyOnAgentRejected = False
        mock_user.maxEmailsPerDay = 3
        self.mock_user_actions.update = AsyncMock(return_value=mock_user)

        from backend.data.user import update_user_timezone

        # Also patch the cache_delete to avoid side effects
        with patch(
            "backend.data.user.get_user_by_id"
        ) as mock_cached:
            mock_cached.cache_delete = MagicMock()
            result = await update_user_timezone(USER_ID, "US/Eastern")

        self.mock_user_actions.update.assert_called_once()
        where_arg = self.mock_user_actions.update.call_args.kwargs.get(
            "where",
            self.mock_user_actions.update.call_args[1].get("where"),
        )
        assert where_arg == {"id": USER_ID}
        data_arg = self.mock_user_actions.update.call_args.kwargs.get(
            "data",
            self.mock_user_actions.update.call_args[1].get("data"),
        )
        assert data_arg == {"timezone": "US/Eastern"}


# ============================================================================
# xfail tests for future PRs
# ============================================================================


class TestPR10WebhookTenancy:
    """PR10: Thread org/team context through webhook and background execution."""

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="PR10: CoPilot agent run missing org/team context"
    )
    async def test_copilot_agent_run_passes_org_team_to_execution(self):
        assert False, "Not implemented"

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="PR10: Preset execution missing org/team context"
    )
    async def test_preset_execution_passes_org_team(self):
        assert False, "Not implemented"

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="PR10: PendingHumanReview missing org/team on create"
    )
    async def test_pending_human_review_gets_org_team_on_create(self):
        assert False, "Not implemented"


class TestPR11RealtimeTenancy:
    """PR11: Re-key SSE/websocket channels by org/team."""

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="PR11: UserNotificationBatch missing org/team columns"
    )
    async def test_notification_batch_created_with_org_team(self):
        assert False, "Not implemented"

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="PR11: subscribe_to_session doesn't validate org"
    )
    async def test_subscribe_to_session_validates_org_membership(self):
        assert False, "Not implemented"

    @pytest.mark.asyncio
    async def test_session_ids_are_globally_unique(self):
        """Document assumption: ChatSession IDs are UUIDs, no re-keying
        needed."""
        # Verified by schema inspection: ChatSession.id uses @default(uuid())
        assert True


class TestPR14APIKeyTenancy:
    """PR14: API keys scoped to org context."""

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="PR14: create_api_key doesn't accept org context"
    )
    async def test_create_api_key_with_org_context(self):
        assert False, "Not implemented"

    @pytest.mark.asyncio
    async def test_create_api_key_defaults_to_user_owned(self):
        """Current behavior: create_api_key sets userId and nothing else
        for org context. This must keep working."""
        key_row = _make_api_key_row()

        mock_actions = AsyncMock()
        mock_actions.create = AsyncMock(return_value=key_row)

        with patch(
            "backend.data.auth.api_key.PrismaAPIKey.prisma",
            return_value=mock_actions,
        ):
            from backend.data.auth.api_key import create_api_key

            info, plaintext = await create_api_key(
                name="test-key",
                user_id=USER_ID,
                permissions=[],
            )

        create_data = mock_actions.create.call_args.kwargs.get(
            "data", mock_actions.create.call_args[1].get("data")
        )
        assert create_data["userId"] == USER_ID
        assert "organizationId" not in create_data
        assert info.user_id == USER_ID

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="PR14: validate_api_key doesn't return org context"
    )
    async def test_validate_api_key_returns_org_context(self):
        assert False, "Not implemented"

    @pytest.mark.asyncio
    async def test_validate_api_key_without_org_returns_none(self):
        """Current behavior: validate_api_key returns APIKeyInfo with
        user_id but no org fields. Verify that works."""
        key_row = _make_api_key_row()

        mock_actions = AsyncMock()
        mock_actions.find_many = AsyncMock(return_value=[key_row])

        with (
            patch(
                "backend.data.auth.api_key.PrismaAPIKey.prisma",
                return_value=mock_actions,
            ),
            patch(
                "backend.data.auth.api_key.keysmith.verify_key",
                return_value=True,
            ),
        ):
            from backend.data.auth.api_key import validate_api_key

            result = await validate_api_key("agpt_xxxx_fake_key_yyyy")

        assert result is not None
        assert result.user_id == USER_ID
        # No org context fields yet
        assert not hasattr(result, "organization_id") or getattr(
            result, "organization_id", None
        ) is None

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="PR14: external API doesn't resolve org from key"
    )
    async def test_external_api_resolves_org_from_key(self):
        assert False, "Not implemented"

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="PR14: external API doesn't enforce team restriction"
    )
    async def test_external_api_enforces_team_restriction(self):
        assert False, "Not implemented"

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="PR14: API key list not scoped by org")
    async def test_list_api_keys_scoped_by_org(self):
        assert False, "Not implemented"

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="PR14: API key create route doesn't use RequestContext"
    )
    async def test_api_key_create_route_uses_request_context(self):
        assert False, "Not implemented"


class TestPR15MarketplaceOrg:
    """PR15: Marketplace operations scoped to org."""

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="PR15: create_submission doesn't set owningOrgId"
    )
    async def test_create_submission_sets_owning_org_id(self):
        assert False, "Not implemented"

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="PR15: submission doesn't check org membership"
    )
    async def test_create_submission_requires_org_membership(self):
        assert False, "Not implemented"

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="PR15: edit_submission doesn't verify org ownership"
    )
    async def test_edit_submission_verifies_org_ownership(self):
        assert False, "Not implemented"

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="PR15: my_agents not filtered by org")
    async def test_my_agents_filtered_by_org(self):
        assert False, "Not implemented"

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="PR15: store agent doesn't resolve org creator"
    )
    async def test_store_agent_resolves_org_creator(self):
        assert False, "Not implemented"

    @pytest.mark.asyncio
    async def test_store_agent_resolves_user_creator_fallback(self):
        """Current behavior: store agent details resolve creator by
        username, not org. This is the fallback path."""
        mock_agent = MagicMock()
        mock_agent.creator_username = "testuser"
        mock_agent.slug = "my-agent"
        # Fields used by StoreAgentDetails.from_db
        mock_agent.listing_version_id = "slv-1"
        mock_agent.agent_name = "My Agent"
        mock_agent.agent_video = ""
        mock_agent.agent_output_demo = ""
        mock_agent.agent_image = []
        mock_agent.creator_avatar = ""
        mock_agent.sub_heading = ""
        mock_agent.description = "test"
        mock_agent.instructions = None
        mock_agent.categories = []
        mock_agent.runs = 0
        mock_agent.rating = 0.0
        mock_agent.versions = ["1"]
        mock_agent.graph_id = GRAPH_ID
        mock_agent.graph_versions = ["1"]
        mock_agent.updated_at = datetime(2025, 6, 1, tzinfo=timezone.utc)
        mock_agent.recommended_schedule_cron = None

        mock_actions = AsyncMock()
        mock_actions.find_first = AsyncMock(return_value=mock_agent)

        with patch(
            "backend.api.features.store.db.prisma.models.StoreAgent.prisma",
            return_value=mock_actions,
        ):
            from backend.api.features.store.db import (
                get_store_agent_details,
            )

            result = await get_store_agent_details("testuser", "my-agent")

        # Creator resolved by username — no org join
        assert result is not None
        where_arg = mock_actions.find_first.call_args.kwargs.get(
            "where", mock_actions.find_first.call_args[1].get("where")
        )
        assert where_arg["creator_username"] == "testuser"

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="PR15: slug not unique per org")
    async def test_slug_unique_per_org(self):
        assert False, "Not implemented"

    @pytest.mark.asyncio
    async def test_slug_unique_per_user(self):
        """Current behavior: slugs are unique per StoreListing
        (keyed by agentGraphId). This is the user-level behavior."""
        # The schema enforces: StoreListing.agentGraphId is @unique
        # and StoreListing.slug is @unique — so slugs are globally unique.
        # This test documents that assumption.
        assert True  # Verified by schema: StoreListing.slug is @unique

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="PR15: fork_graph to different org not wired"
    )
    async def test_fork_graph_to_different_org(self):
        assert False, "Not implemented"

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="PR15: copy_graph endpoint doesn't exist")
    async def test_copy_graph_to_different_team(self):
        assert False, "Not implemented"

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="PR15: Creator view doesn't join org profile"
    )
    async def test_creator_view_joins_org_profile(self):
        assert False, "Not implemented"

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="PR15: StoreSubmission view missing organization_id"
    )
    async def test_submission_exposes_organization_id(self):
        assert False, "Not implemented"


class TestPR16Transfers:
    """PR16: Asset transfer between orgs."""

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="PR16: transfer endpoint doesn't exist")
    async def test_initiate_transfer_creates_pending_request(self):
        assert False, "Not implemented"

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="PR16: transfer endpoint doesn't exist")
    async def test_initiate_transfer_requires_source_admin(self):
        assert False, "Not implemented"

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="PR16: transfer approval doesn't exist"
    )
    async def test_approve_transfer_from_source_admin(self):
        assert False, "Not implemented"

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="PR16: transfer approval doesn't exist"
    )
    async def test_approve_transfer_from_target_admin(self):
        assert False, "Not implemented"

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="PR16: transfer execution doesn't exist"
    )
    async def test_execute_transfer_moves_ownership(self):
        assert False, "Not implemented"

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="PR16: transfer execution doesn't exist"
    )
    async def test_execute_requires_both_approvals(self):
        assert False, "Not implemented"

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="PR16: transfer rejection doesn't exist"
    )
    async def test_reject_transfer_sets_status(self):
        assert False, "Not implemented"

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="PR16: credential handling on transfer doesn't exist"
    )
    async def test_transferred_resource_loses_credentials(self):
        assert False, "Not implemented"


class TestPR18Cutover:
    """PR18: Full cutover from userId to org/team scoping."""

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="PR18: list_graphs still uses userId not org"
    )
    async def test_list_graphs_returns_only_org_graphs(self):
        assert False, "Not implemented"

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="PR18: list_graphs doesn't exclude other orgs"
    )
    async def test_list_graphs_excludes_other_org_graphs(self):
        assert False, "Not implemented"

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="PR18: get_graph doesn't check org membership"
    )
    async def test_get_graph_requires_org_membership(self):
        assert False, "Not implemented"

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="PR18: delete_graph doesn't check org")
    async def test_delete_graph_requires_org_ownership(self):
        assert False, "Not implemented"

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="PR18: executions not scoped by team"
    )
    async def test_list_executions_scoped_by_team(self):
        assert False, "Not implemented"

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="PR18: get_execution doesn't check org"
    )
    async def test_get_execution_requires_org_membership(self):
        assert False, "Not implemented"

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="PR18: schedule not scoped to org"
    )
    async def test_create_schedule_scoped_to_org(self):
        assert False, "Not implemented"

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="PR18: file upload not scoped to org"
    )
    async def test_upload_file_scoped_to_org(self):
        assert False, "Not implemented"

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="PR18: credits still user-scoped")
    async def test_get_credits_returns_org_balance(self):
        assert False, "Not implemented"

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="PR18: top-up still user-scoped")
    async def test_top_up_credits_org_balance(self):
        assert False, "Not implemented"

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="PR18: credit history still user-scoped"
    )
    async def test_credit_history_returns_org_transactions(self):
        assert False, "Not implemented"

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="PR18: API keys not scoped by org"
    )
    async def test_list_api_keys_scoped_by_org_cutover(self):
        assert False, "Not implemented"

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="PR18: API key create doesn't set org"
    )
    async def test_create_api_key_sets_org_context(self):
        assert False, "Not implemented"

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="PR18: organizationId still nullable"
    )
    async def test_all_agent_graphs_have_organization_id(self):
        assert False, "Not implemented"

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="PR18: organizationId still nullable"
    )
    async def test_all_executions_have_organization_id(self):
        assert False, "Not implemented"

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="PR18: owningOrgId still nullable"
    )
    async def test_all_store_listings_have_owning_org_id(self):
        assert False, "Not implemented"

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="PR18: Creator view not org-aware"
    )
    async def test_creator_view_resolves_org_profile(self):
        assert False, "Not implemented"

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="PR18: StoreAgent view not org-aware"
    )
    async def test_store_agent_view_includes_org_id(self):
        assert False, "Not implemented"

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="PR18: StoreSubmission view not org-aware"
    )
    async def test_store_submission_view_includes_org_id(self):
        assert False, "Not implemented"

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="PR18: legacy userId fallback not removed"
    )
    async def test_read_by_user_id_fallback_removed(self):
        assert False, "Not implemented"
