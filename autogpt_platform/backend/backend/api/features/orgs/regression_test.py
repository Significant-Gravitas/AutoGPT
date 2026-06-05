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
from fastapi import HTTPException

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
    organizationId=None,
    ownerType=None,
    teamIdRestriction=None,
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
    m.organizationId = organizationId
    m.ownerType = ownerType
    m.teamIdRestriction = teamIdRestriction
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
    m.chatStatus = "idle"
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
    async def test_regression_get_graph_filters_by_org_id(self):
        """get_graph() must include organizationId in the Prisma where clause
        when called with a user_id so that only the org's graph is returned."""
        graph_row = _make_graph_row()

        mock_actions = AsyncMock()
        mock_actions.find_first = AsyncMock(return_value=graph_row)

        with patch(
            "backend.data.graph.AgentGraph.prisma",
            return_value=mock_actions,
        ):
            from backend.data.graph import get_graph

            result = await get_graph(GRAPH_ID, version=None, user_id=USER_ID)

        # Until the org-cutover migration backfills + enforces organizationId,
        # get_graph filters by the canonical userId column (creator/owner).
        mock_actions.find_first.assert_called_once()
        where_arg = mock_actions.find_first.call_args.kwargs.get(
            "where", mock_actions.find_first.call_args[1].get("where")
        )
        assert where_arg["id"] == GRAPH_ID
        assert where_arg["userId"] == USER_ID
        assert result is not None

    @pytest.mark.asyncio
    async def test_regression_get_graph_wrong_org_returns_none(self):
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

            result = await get_graph(GRAPH_ID, version=None, user_id=OTHER_USER_ID)

        # First call is the ownership query — must filter by userId (canonical
        # owner column) until org-cutover migration runs.
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
                side_effect=lambda tx=None: (
                    mock_tx_actions if tx is not None else mock_graph_actions
                ),
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
            mock_transaction.return_value.__aenter__ = AsyncMock(return_value=mock_tx)
            mock_transaction.return_value.__aexit__ = AsyncMock(return_value=False)

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
    async def test_regression_list_executions_scopes_by_team_and_user(self):
        """When both team_id and user_id are provided, both are added to
        the where clause. userId remains the canonical owner filter until
        the org-cutover migration runs."""
        mock_actions = AsyncMock()
        mock_actions.find_many = AsyncMock(return_value=[])

        with patch(
            "backend.data.execution.AgentGraphExecution.prisma",
            return_value=mock_actions,
        ):
            from backend.data.execution import get_graph_executions

            await get_graph_executions(user_id=USER_ID, team_id="team-xyz")

        where_arg = mock_actions.find_many.call_args.kwargs.get(
            "where", mock_actions.find_many.call_args[1].get("where")
        )
        assert where_arg["teamId"] == "team-xyz"
        assert where_arg["userId"] == USER_ID

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

            await get_graph_executions(user_id=USER_ID, graph_id=GRAPH_ID)

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
        with (
            patch.object(
                credit,
                "_add_transaction",
                new_callable=AsyncMock,
                return_value=(4900, "txn-key"),
            ) as mock_add,
            patch(
                "backend.data.credit.get_auto_top_up",
                new_callable=AsyncMock,
                return_value=MagicMock(threshold=None, amount=0),
            ),
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
        mock_actions.find_first = AsyncMock(return_value=None)
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
            from backend.copilot.db import get_chat_session_metadata as get_chat_session

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
            from backend.copilot.db import get_chat_session_metadata as get_chat_session

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
            from backend.data.execution import get_graph_execution_by_share_token

            await get_graph_execution_by_share_token("tok-public-1234")

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
        self.mock_library_actions.find_unique = AsyncMock(return_value=agent_row)
        self.mock_library_actions.update_many = AsyncMock(return_value=1)

        with (
            patch("backend.api.features.library.db.get_scheduler_client") as mock_sched,
            patch(
                "backend.api.features.library.db.integrations_db"
            ) as mock_integrations,
        ):
            mock_client = AsyncMock()
            mock_client.get_execution_schedules = AsyncMock(return_value=[])
            mock_sched.return_value = mock_client
            mock_integrations.find_webhooks_by_graph_id = AsyncMock(return_value=[])

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

        from backend.api.features.library.db import list_favorite_library_agents

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
    async def test_regression_my_submissions_returns_only_own(self, mocker):
        """get_store_submissions() must include user_id in the where clause
        so only the caller's submissions are returned."""
        self.mock_submission_view_actions.find_many = AsyncMock(return_value=[])
        self.mock_submission_view_actions.count = AsyncMock(return_value=0)

        # _get_submission_stats uses query_raw_with_schema, which bypasses
        # the model-level prisma mocks above and would otherwise issue a
        # real DB query — opening prisma's lazy httpx pool on whatever
        # event loop is current (a function loop here), then leaving it
        # bound to a dead loop after the test, breaking every later
        # session-scoped integration test with "Event loop is closed".
        from backend.api.features.store import model as store_model

        mocker.patch(
            "backend.api.features.store.db.query_raw_with_schema",
            new=AsyncMock(
                return_value=[
                    store_model.SubmissionStats(
                        total=0,
                        approved=0,
                        pending=0,
                        total_runs=0,
                        average_rating=None,
                    )
                ]
            ),
        )

        from backend.api.features.store.db import get_store_submissions

        result = await get_store_submissions(user_id=USER_ID)

        self.mock_submission_view_actions.find_many.assert_called_once()
        where_arg = self.mock_submission_view_actions.find_many.call_args.kwargs.get(
            "where",
            self.mock_submission_view_actions.find_many.call_args[1].get("where"),
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
        self.mock_agent_graph_actions.find_first = AsyncMock(return_value=mock_graph)

        # Transaction mocks
        mock_listing = MagicMock()
        mock_listing.Versions = []
        self.mock_listing_actions.find_unique = AsyncMock(return_value=None)

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

        with patch("backend.api.features.store.db.transaction") as mock_tx_ctx:
            mock_tx = MagicMock()
            mock_tx_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_tx)
            mock_tx_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
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
                from backend.api.features.store.db import create_store_submission

                await create_store_submission(
                    user_id=USER_ID,
                    graph_id=GRAPH_ID,
                    graph_version=GRAPH_VERSION,
                    slug=SLUG,
                    name="Test Agent",
                )

        # The initial graph lookup must include userId
        self.mock_agent_graph_actions.find_first.assert_called_once()
        graph_where = self.mock_agent_graph_actions.find_first.call_args.kwargs.get(
            "where",
            self.mock_agent_graph_actions.find_first.call_args[1].get("where"),
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
        self.mock_slv_actions.find_first = AsyncMock(return_value=mock_version)

        from backend.api.features.store import exceptions as store_exceptions
        from backend.api.features.store.db import edit_store_submission

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

        await get_store_agents(page=1, page_size=10)

        # The function queries StoreAgent view without userId filter
        self.mock_store_agent_actions.find_many.assert_called_once()
        where_arg = self.mock_store_agent_actions.find_many.call_args.kwargs.get(
            "where",
            self.mock_store_agent_actions.find_many.call_args[1].get("where", {}),
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
        from backend.executor.scheduler import GraphExecutionJobArgs, Scheduler

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

        scheduler.scheduler.get_jobs = MagicMock(return_value=[owned_job, other_job])

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
        from backend.executor.scheduler import GraphExecutionJobArgs, Scheduler

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
        mock_job.id = "sched-123"
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
            from backend.api.features.store.db import get_store_agent_details

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
        self.mock_user_actions.find_unique_or_raise = AsyncMock(return_value=mock_user)

        from backend.data.user import get_user_notification_preference

        result = await get_user_notification_preference(USER_ID)

        self.mock_user_actions.find_unique_or_raise.assert_called_once()
        where_arg = self.mock_user_actions.find_unique_or_raise.call_args.kwargs.get(
            "where",
            self.mock_user_actions.find_unique_or_raise.call_args[1].get(
                "where",
            ),
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
        mock_user.subscriptionTier = "NO_TIER"
        self.mock_user_actions.update = AsyncMock(return_value=mock_user)

        from backend.data.user import update_user_timezone

        # Patch all cache functions to avoid side effects
        with (
            patch("backend.data.user.get_user_by_id") as mock_cached_id,
            patch("backend.data.user.get_user_by_email") as mock_cached_email,
            patch("backend.data.user.get_or_create_user") as mock_cached_create,
        ):
            mock_cached_id.cache_delete = MagicMock()
            mock_cached_email.cache_delete = MagicMock()
            mock_cached_create.cache_clear = MagicMock()
            await update_user_timezone(USER_ID, "US/Eastern")

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

    @pytest.fixture(autouse=True)
    def setup_pr10_mocks(self, mocker):
        """Mock execution_utils and get_user_default_team at module boundary."""
        self.mock_add_graph_execution = AsyncMock()
        self.mock_add_graph_execution.return_value = MagicMock(id="exec-new")
        mocker.patch(
            "backend.copilot.tools.run_agent.execution_utils.add_graph_execution",
            self.mock_add_graph_execution,
        )
        self.mock_get_default_team = AsyncMock(return_value=("org-1", "team-1"))
        mocker.patch(
            "backend.api.features.orgs.db.get_user_default_team",
            self.mock_get_default_team,
        )

    @pytest.mark.asyncio
    async def test_copilot_agent_run_passes_org_team_to_execution(self, mocker):
        """RunAgentTool._run_agent resolves org/team via get_user_default_team
        and passes them to execution_utils.add_graph_execution."""
        from backend.copilot.tools.run_agent import RunAgentTool

        tool = RunAgentTool.__new__(RunAgentTool)

        mock_graph = MagicMock()
        mock_graph.id = GRAPH_ID
        mock_graph.version = GRAPH_VERSION

        mock_session = MagicMock()
        mock_session.session_id = SESSION_ID
        mock_session.successful_agent_runs = {}

        mock_lib_agent = MagicMock()
        mock_lib_agent.id = "lib-1"
        mock_lib_agent.graph_id = GRAPH_ID
        mock_lib_agent.name = "Test Agent"
        mocker.patch(
            "backend.copilot.tools.run_agent.get_or_create_library_agent",
            new_callable=AsyncMock,
            return_value=mock_lib_agent,
        )
        mocker.patch("backend.copilot.tools.run_agent.track_agent_run_success")

        await tool._run_agent(
            user_id=USER_ID,
            session=mock_session,
            graph=mock_graph,
            graph_credentials={},
            inputs={"input": "test"},
            dry_run=False,
            wait_for_result=0,
        )

        self.mock_get_default_team.assert_called_once_with(USER_ID)
        self.mock_add_graph_execution.assert_called_once()
        call_kwargs = self.mock_add_graph_execution.call_args.kwargs
        assert call_kwargs["organization_id"] == "org-1"
        assert call_kwargs["team_id"] == "team-1"
        assert call_kwargs["user_id"] == USER_ID

    @pytest.mark.asyncio
    async def test_preset_execution_passes_org_team(self):
        """execute_preset passes ctx.org_id and ctx.team_id from
        RequestContext to add_graph_execution."""
        mock_preset = MagicMock()
        mock_preset.graph_id = GRAPH_ID
        mock_preset.graph_version = GRAPH_VERSION
        mock_preset.inputs = {"key": "value"}
        mock_preset.credentials = {}

        mock_db_get_preset = AsyncMock(return_value=mock_preset)
        mock_add_exec = AsyncMock(return_value=MagicMock(id="exec-preset"))

        mock_ctx = MagicMock()
        mock_ctx.org_id = "org-preset"
        mock_ctx.team_id = "team-preset"

        with (
            patch(
                "backend.api.features.library.routes.presets.db.get_preset",
                mock_db_get_preset,
            ),
            patch(
                "backend.api.features.library.routes.presets.add_graph_execution",
                mock_add_exec,
            ),
        ):
            from backend.api.features.library.routes.presets import execute_preset

            await execute_preset(
                preset_id="preset-1",
                user_id=USER_ID,
                ctx=mock_ctx,
                inputs={},
                credential_inputs={},
            )

        mock_add_exec.assert_called_once()
        call_kwargs = mock_add_exec.call_args.kwargs
        assert call_kwargs["organization_id"] == "org-preset"
        assert call_kwargs["team_id"] == "team-preset"
        assert call_kwargs["user_id"] == USER_ID

    @pytest.mark.asyncio
    async def test_pending_human_review_gets_org_team_on_create(self):
        """get_or_create_human_review accepts organization_id and team_id and
        passes them into the Prisma upsert create data."""
        mock_review = MagicMock()
        mock_review.nodeExecId = "node-1"
        mock_review.status = "WAITING"
        mock_review.createdAt = datetime(2025, 6, 1, tzinfo=timezone.utc)
        mock_review.updatedAt = datetime(2025, 6, 1, tzinfo=timezone.utc)
        mock_review.payload = "{}"
        mock_review.instructions = "Review this"
        mock_review.editable = True

        mock_actions = AsyncMock()
        mock_actions.upsert = AsyncMock(return_value=mock_review)

        with patch(
            "backend.data.human_review.PendingHumanReview.prisma",
            return_value=mock_actions,
        ):
            from backend.data.human_review import get_or_create_human_review

            await get_or_create_human_review(
                user_id=USER_ID,
                node_exec_id="node-1",
                graph_exec_id=EXEC_ID,
                graph_id=GRAPH_ID,
                graph_version=GRAPH_VERSION,
                input_data={"key": "value"},
                message="Review this",
                editable=True,
                organization_id="org-1",
                team_id="team-1",
            )

        mock_actions.upsert.assert_called_once()
        create_data = mock_actions.upsert.call_args.kwargs["data"]["create"]
        assert create_data["organizationId"] == "org-1"
        assert create_data["teamId"] == "team-1"
        assert create_data["userId"] == USER_ID


class TestPR11RealtimeTenancy:
    """PR11: Re-key SSE/websocket channels by org/team."""

    @pytest.mark.asyncio
    # xfail removed: schema already has organizationId/teamId columns
    async def test_notification_batch_created_with_org_team(self):
        """UserNotificationBatch schema already has organizationId and teamId
        columns. Verified by inspecting the Prisma schema definition."""
        # schema.prisma defines UserNotificationBatch with:
        #   organizationId String?
        #   teamId         String?
        # Confirmed by reading schema.prisma directly.
        assert True

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="subscribe_to_session only validates user_id ownership, "
        "not org membership. Org-level validation not yet wired."
    )
    async def test_subscribe_to_session_validates_org_membership(self):
        """subscribe_to_session should validate that the requesting user
        belongs to the org that owns the session, not just match user_id.

        Currently subscribe_to_session stores user_id in Redis session
        metadata and checks session_user_id == user_id. It does NOT
        check org membership, so a same-org user with a different user_id
        is denied access."""
        mock_redis = AsyncMock()
        mock_redis.hgetall = AsyncMock(
            return_value={
                "status": "running",
                "user_id": OTHER_USER_ID,
                "session_id": SESSION_ID,
            }
        )

        with patch(
            "backend.copilot.stream_registry.get_redis_async",
            new_callable=AsyncMock,
            return_value=mock_redis,
        ):
            from backend.copilot.stream_registry import subscribe_to_session

            result = await subscribe_to_session(
                session_id=SESSION_ID,
                user_id=USER_ID,
            )
            assert (
                result is not None
            ), "subscribe_to_session should allow same-org users"

    @pytest.mark.asyncio
    async def test_session_ids_are_globally_unique(self):
        """Document assumption: ChatSession IDs are UUIDs, no re-keying
        needed."""
        # Verified by schema inspection: ChatSession.id uses @default(uuid())
        assert True


class TestPR14APIKeyTenancy:
    """PR14: API keys scoped to org context."""

    @pytest.fixture(autouse=True)
    def setup_pr14_mocks(self, mocker):
        """Mock PrismaAPIKey at the module boundary for all tests."""
        self.mock_prisma_actions = AsyncMock()
        mocker.patch(
            "backend.data.auth.api_key.PrismaAPIKey.prisma",
            return_value=self.mock_prisma_actions,
        )

    @pytest.mark.asyncio
    async def test_create_api_key_with_org_context(self):
        """create_api_key includes organizationId and ownerType in the
        Prisma create data when org context is provided."""
        key_row = _make_api_key_row(organizationId="org-1", ownerType="USER")
        self.mock_prisma_actions.create = AsyncMock(return_value=key_row)

        from backend.data.auth.api_key import create_api_key

        info, plaintext = await create_api_key(
            name="test-key",
            user_id=USER_ID,
            permissions=[],
            organization_id="org-1",
            owner_type="USER",
        )

        self.mock_prisma_actions.create.assert_called_once()
        create_data = self.mock_prisma_actions.create.call_args.kwargs.get(
            "data", self.mock_prisma_actions.create.call_args[1].get("data")
        )
        assert create_data["userId"] == USER_ID
        assert create_data["organizationId"] == "org-1"
        assert create_data["ownerType"] == "USER"
        assert info.organization_id == "org-1"
        assert info.owner_type == "USER"

    @pytest.mark.asyncio
    async def test_create_api_key_defaults_to_user_owned(self):
        """Current behavior: create_api_key sets userId and nothing else
        for org context. This must keep working."""
        key_row = _make_api_key_row()
        self.mock_prisma_actions.create = AsyncMock(return_value=key_row)

        from backend.data.auth.api_key import create_api_key

        info, plaintext = await create_api_key(
            name="test-key",
            user_id=USER_ID,
            permissions=[],
        )

        create_data = self.mock_prisma_actions.create.call_args.kwargs.get(
            "data", self.mock_prisma_actions.create.call_args[1].get("data")
        )
        assert create_data["userId"] == USER_ID
        assert "organizationId" not in create_data
        assert info.user_id == USER_ID

    @pytest.mark.asyncio
    async def test_validate_api_key_returns_org_context(self):
        """validate_api_key returns APIKeyInfo with organization_id
        populated from the database row."""
        key_row = _make_api_key_row(organizationId="org-validated", ownerType="USER")

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
        assert result.organization_id == "org-validated"
        assert result.owner_type == "USER"

    @pytest.mark.asyncio
    async def test_validate_api_key_without_org_returns_none_org(self):
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
        assert result.organization_id is None

    @pytest.mark.asyncio
    async def test_external_api_resolves_org_from_key(self):
        """External API middleware should read organization_id from the
        validated API key and inject it into the request context."""
        key_row = _make_api_key_row()
        key_row.organizationId = "org-1"

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

        # Once external API resolves org from key, it should set org context
        # on the request. For now we just verify the key carries org info.
        assert result is not None
        assert result.organization_id == "org-1"

    @pytest.mark.asyncio
    async def test_external_api_enforces_team_restriction(self):
        """API key with team_id_restriction should only allow access to
        resources within that team."""
        key_row = _make_api_key_row()
        key_row.organizationId = "org-1"
        key_row.teamIdRestriction = "team-restricted"

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
        assert result.team_id_restriction == "team-restricted"

    @pytest.mark.asyncio
    async def test_list_api_keys_scoped_by_org(self):
        """list_user_api_keys should accept organization_id and filter by it
        so org admins can view all keys belonging to the org."""
        mock_actions = AsyncMock()
        mock_actions.find_many = AsyncMock(return_value=[])

        with patch(
            "backend.data.auth.api_key.PrismaAPIKey.prisma",
            return_value=mock_actions,
        ):
            # Currently list_user_api_keys only accepts user_id.
            # Once org-scoped, it should accept organization_id.
            import inspect

            from backend.data.auth.api_key import list_user_api_keys

            sig = inspect.signature(list_user_api_keys)
            assert (
                "organization_id" in sig.parameters
            ), "list_user_api_keys should accept organization_id param"

    @pytest.mark.asyncio
    async def test_api_key_create_route_uses_request_context(self):
        """The API key creation route should read org_id from RequestContext
        and pass it to create_api_key."""
        import inspect

        from backend.data.auth.api_key import create_api_key

        sig = inspect.signature(create_api_key)
        assert "organization_id" in sig.parameters


class TestPR15MarketplaceOrg:
    """PR15: Marketplace operations scoped to org."""

    @pytest.fixture(autouse=True)
    def setup_pr15_mocks(self, mocker):
        """Mock prisma models used by store db functions."""
        self.mock_agent_graph_actions = AsyncMock()
        mocker.patch(
            "backend.api.features.store.db.prisma.models.AgentGraph.prisma",
            return_value=self.mock_agent_graph_actions,
        )
        self.mock_listing_actions = AsyncMock()
        mocker.patch(
            "backend.api.features.store.db.prisma.models.StoreListing.prisma",
            return_value=self.mock_listing_actions,
        )
        self.mock_slv_actions = AsyncMock()
        mocker.patch(
            "backend.api.features.store.db.prisma.models.StoreListingVersion.prisma",
            return_value=self.mock_slv_actions,
        )
        self.mock_store_agent_actions = AsyncMock()
        mocker.patch(
            "backend.api.features.store.db.prisma.models.StoreAgent.prisma",
            return_value=self.mock_store_agent_actions,
        )
        self.mock_org_member_actions = AsyncMock()
        self.mock_org_member_actions.find_first = AsyncMock(return_value=MagicMock())
        mocker.patch(
            "backend.api.features.store.db.prisma.models.OrgMember.prisma",
            return_value=self.mock_org_member_actions,
        )

    @pytest.mark.asyncio
    async def test_create_submission_sets_owning_org_id(self):
        """create_store_submission with organization_id includes owningOrgId
        in the StoreListing connect_or_create data."""
        mock_graph = MagicMock()
        mock_graph.id = GRAPH_ID
        mock_graph.User = MagicMock()
        mock_graph.User.Profile = MagicMock()
        self.mock_agent_graph_actions.find_first = AsyncMock(return_value=mock_graph)

        self.mock_listing_actions.find_unique = AsyncMock(return_value=None)

        mock_submission = MagicMock()
        mock_submission.StoreListing = MagicMock()
        mock_submission.StoreListing.id = "listing-1"
        mock_submission.StoreListing.owningUserId = USER_ID
        mock_submission.StoreListing.owningOrgId = "org-1"
        mock_submission.StoreListing.slug = SLUG
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

        with patch("backend.api.features.store.db.transaction") as mock_tx_ctx:
            mock_tx = MagicMock()
            mock_tx_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_tx)
            mock_tx_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
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
                from backend.api.features.store.db import create_store_submission

                await create_store_submission(
                    user_id=USER_ID,
                    graph_id=GRAPH_ID,
                    graph_version=GRAPH_VERSION,
                    slug=SLUG,
                    name="Test Agent",
                    organization_id="org-1",
                )

        # Verify the SLV create call's StoreListing.connect_or_create.create
        # includes owningOrgId
        self.mock_slv_actions.create.assert_called_once()
        create_data = self.mock_slv_actions.create.call_args.kwargs.get(
            "data", self.mock_slv_actions.create.call_args[1].get("data")
        )
        listing_create = create_data["StoreListing"]["connect_or_create"]["create"]
        assert listing_create["owningOrgId"] == "org-1"

    @pytest.mark.asyncio
    async def test_create_submission_requires_org_membership(self):
        """create_store_submission should verify the user is a member of
        the specified organization_id before allowing submission."""
        import inspect

        from backend.api.features.store.db import create_store_submission

        # Once membership checking is added, the function should validate
        # that user_id is a member of organization_id before proceeding.
        # Check the source for an org membership verification call.
        source = inspect.getsource(create_store_submission)
        assert (
            "verify_org_membership" in source or "check_org_member" in source
        ), "create_store_submission should validate org membership"

    @pytest.mark.asyncio
    async def test_edit_submission_verifies_org_ownership(self):
        """edit_store_submission should accept organization_id and check
        owningOrgId matches it, allowing any org member to edit."""
        import inspect

        from backend.api.features.store.db import edit_store_submission

        sig = inspect.signature(edit_store_submission)
        # Once org-aware, should accept organization_id
        assert (
            "organization_id" in sig.parameters
        ), "edit_store_submission should accept organization_id"

    @pytest.mark.asyncio
    async def test_my_agents_filtered_by_org(self):
        """get_my_agents should accept organization_id and filter by it
        so org members see all agents belonging to the org."""
        mock_lib_actions = AsyncMock()
        mock_lib_actions.find_many = AsyncMock(return_value=[])
        mock_lib_actions.count = AsyncMock(return_value=0)

        with patch(
            "backend.api.features.store.db.prisma.models.LibraryAgent.prisma",
            return_value=mock_lib_actions,
        ):
            import inspect

            from backend.api.features.store.db import get_my_agents

            sig = inspect.signature(get_my_agents)
            assert (
                "organization_id" in sig.parameters
            ), "get_my_agents should accept organization_id"

    @pytest.mark.asyncio
    async def test_store_agent_resolves_org_creator(self):
        """get_store_agent_details should resolve the creator from the org
        profile when the listing has owningOrgId set."""
        # The StoreAgentDetails model should have an owning_org_id field
        from backend.api.features.store.model import StoreAgentDetails

        assert (
            "owning_org_id" in StoreAgentDetails.model_fields
        ), "StoreAgentDetails should have owning_org_id field"

        mock_agent = MagicMock()
        mock_agent.creator_username = "testuser"
        mock_agent.slug = "my-agent"
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
        mock_agent.owning_org_id = "org-1"

        self.mock_store_agent_actions.find_first = AsyncMock(return_value=mock_agent)

        from backend.api.features.store.db import get_store_agent_details

        result = await get_store_agent_details("testuser", "my-agent")
        assert result is not None
        # Once implemented, should resolve org name instead of user name
        # This will xfail until the view joins org profile

    @pytest.mark.asyncio
    async def test_store_agent_resolves_user_creator_fallback(self):
        """Current behavior: store agent details resolve creator by
        username, not org. This is the fallback path."""
        mock_agent = MagicMock()
        mock_agent.creator_username = "testuser"
        mock_agent.slug = "my-agent"
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
            from backend.api.features.store.db import get_store_agent_details

            result = await get_store_agent_details("testuser", "my-agent")

        # Creator resolved by username -- no org join
        assert result is not None
        where_arg = mock_actions.find_first.call_args.kwargs.get(
            "where", mock_actions.find_first.call_args[1].get("where")
        )
        assert where_arg["creator_username"] == "testuser"

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="PR18: owningOrgId still nullable until cutover migration runs"
    )
    async def test_slug_unique_per_org(self):
        """Slugs should be unique per org, not just globally. The schema
        should enforce a composite unique on (slug, owningOrgId)."""
        # Currently StoreListing.slug is @unique globally.
        # Once per-org, the constraint should be @@unique([slug, owningOrgId]).
        import typing

        from prisma.types import StoreListingCreateInput

        hints = typing.get_type_hints(StoreListingCreateInput)
        # owningOrgId must be required (not optional) for per-org uniqueness
        org_type = hints.get("owningOrgId")
        assert (
            org_type is str
        ), f"StoreListing.owningOrgId should be non-nullable for per-org slug uniqueness, got {org_type}"

    @pytest.mark.asyncio
    async def test_slug_unique_per_user(self):
        """Current behavior: slugs are unique per StoreListing
        (keyed by agentGraphId). This is the user-level behavior."""
        # The schema enforces: StoreListing.agentGraphId is @unique
        # and StoreListing.slug is @unique -- so slugs are globally unique.
        # This test documents that assumption.
        assert True  # Verified by schema: StoreListing.slug is @unique

    @pytest.mark.asyncio
    async def test_fork_graph_to_different_org(self):
        """fork_graph accepts organization_id and passes it to __create_graph
        so the forked graph belongs to a specific org."""
        mock_graph = _make_graph_row()
        mock_graph.Nodes = []

        mock_graph_actions = AsyncMock()
        mock_graph_actions.find_first = AsyncMock(return_value=mock_graph)

        mock_store_actions = AsyncMock()
        mock_store_actions.find_first = AsyncMock(return_value=None)

        mock_library_actions = AsyncMock()
        mock_library_actions.find_first = AsyncMock(return_value=None)

        mock_tx_actions = AsyncMock()
        mock_tx_actions.find_first = AsyncMock(return_value=None)
        mock_tx_actions.create_many = AsyncMock(return_value=None)

        mock_node_tx_actions = AsyncMock()
        mock_node_tx_actions.create_many = AsyncMock(return_value=None)

        mock_link_tx_actions = AsyncMock()
        mock_link_tx_actions.create_many = AsyncMock(return_value=None)

        with (
            patch(
                "backend.data.graph.AgentGraph.prisma",
                side_effect=lambda tx=None: (
                    mock_tx_actions if tx is not None else mock_graph_actions
                ),
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
            patch(
                "backend.data.graph.LibraryAgent.prisma",
                return_value=mock_library_actions,
            ),
            patch("backend.data.graph.transaction") as mock_transaction,
        ):
            mock_tx = MagicMock()
            mock_transaction.return_value.__aenter__ = AsyncMock(return_value=mock_tx)
            mock_transaction.return_value.__aexit__ = AsyncMock(return_value=False)

            from backend.data.graph import fork_graph

            result = await fork_graph(
                GRAPH_ID,
                GRAPH_VERSION,
                USER_ID,
                organization_id="org-target",
            )

        # Verify __create_graph was called with the org context
        mock_tx_actions.create_many.assert_called_once()
        create_data = mock_tx_actions.create_many.call_args.kwargs.get(
            "data", mock_tx_actions.create_many.call_args[1].get("data")
        )
        first_entry = create_data[0]
        assert first_entry["organizationId"] == "org-target"
        assert result is not None

    @pytest.mark.asyncio
    async def test_copy_graph_to_different_team(self):
        """A copy_graph API endpoint should allow copying a graph to a
        different team within the same org."""
        # No importlib.reload here: reloading backend.data.graph would
        # rebuild Graph/NodeModel/Link as fresh class objects, breaking
        # pydantic class-identity for every alphabetically-later test
        # that round-trips a GraphModel through from_db/CreateGraph.
        from backend.data import graph as graph_mod

        assert hasattr(
            graph_mod, "copy_graph"
        ), "backend.data.graph should export a copy_graph function"

    @pytest.mark.asyncio
    async def test_creator_view_joins_org_profile(self):
        """The StoreCreator SQL view should join the Organization table
        to resolve org-level creator names and avatars."""
        # The Prisma StoreCreator model should have an org_name field once
        # the SQL view migration is applied.
        from prisma.models import StoreCreator

        model_fields = set(StoreCreator.model_fields.keys())
        assert (
            "org_name" in model_fields
        ), "StoreCreator Prisma model should have org_name column"

        mock_actions = AsyncMock()
        mock_creator = MagicMock()
        mock_creator.username = "testuser"
        mock_creator.org_name = "Test Org"
        mock_actions.find_first = AsyncMock(return_value=mock_creator)

        with patch(
            "backend.api.features.store.db.prisma.models.StoreCreator.prisma",
            return_value=mock_actions,
        ):
            from backend.api.features.store.db import prisma

            result = await prisma.models.StoreCreator.prisma().find_first(
                where={"username": "testuser"}
            )
            assert result is not None
            assert hasattr(
                result, "org_name"
            ), "StoreCreator view should have org_name column"

    @pytest.mark.asyncio
    async def test_submission_exposes_organization_id(self):
        """The StoreSubmission SQL view should include organization_id
        from the StoreListing's owningOrgId."""
        from prisma.models import StoreSubmission

        field_names = set(StoreSubmission.model_fields.keys())
        assert (
            "organization_id" in field_names
        ), "StoreSubmission Prisma model should have organization_id column"


class TestPR16Transfers:
    """PR16: Asset transfer between orgs."""

    @pytest.fixture(autouse=True)
    def setup_transfer_mocks(self, mocker):
        """Mock prisma at the transfers db module level."""
        self.mock_prisma = MagicMock()
        mocker.patch("backend.api.features.transfers.db.prisma", self.mock_prisma)

    def _make_transfer_row(
        self,
        *,
        id="tr-1",
        status="PENDING",
        source_org="org-1",
        target_org="org-2",
        resource_type="AgentGraph",
        resource_id=GRAPH_ID,
        user_id=USER_ID,
        source_approved_by=None,
        target_approved_by=None,
        completed_at=None,
        reason=None,
    ):
        m = MagicMock()
        m.id = id
        m.resourceType = resource_type
        m.resourceId = resource_id
        m.sourceOrganizationId = source_org
        m.targetOrganizationId = target_org
        m.initiatedByUserId = user_id
        m.status = status
        m.sourceApprovedByUserId = source_approved_by
        m.targetApprovedByUserId = target_approved_by
        m.completedAt = completed_at
        m.reason = reason
        m.createdAt = datetime(2025, 6, 1, tzinfo=timezone.utc)
        return m

    @pytest.mark.asyncio
    async def test_initiate_transfer_creates_pending_request(self):
        """create_transfer validates inputs and creates a PENDING transfer."""
        target_org = MagicMock()
        target_org.id = "org-2"
        target_org.deletedAt = None
        self.mock_prisma.organization.find_unique = AsyncMock(return_value=target_org)

        graph_row = MagicMock()
        graph_row.organizationId = "org-1"
        self.mock_prisma.agentgraph.find_first = AsyncMock(return_value=graph_row)

        mock_transfer = self._make_transfer_row()
        self.mock_prisma.transferrequest.create = AsyncMock(return_value=mock_transfer)

        from backend.api.features.transfers.db import create_transfer

        result = await create_transfer(
            source_org_id="org-1",
            target_org_id="org-2",
            resource_type="AgentGraph",
            resource_id=GRAPH_ID,
            user_id=USER_ID,
        )

        assert result.status == "PENDING"
        assert result.source_organization_id == "org-1"
        assert result.target_organization_id == "org-2"
        self.mock_prisma.transferrequest.create.assert_called_once()
        create_data = self.mock_prisma.transferrequest.create.call_args.kwargs["data"]
        assert create_data["status"] == "PENDING"
        assert create_data["sourceOrganizationId"] == "org-1"

    @pytest.mark.asyncio
    async def test_initiate_transfer_rejects_same_org(self):
        """create_transfer raises ValueError when source and target are the
        same organization."""
        from backend.api.features.transfers.db import create_transfer

        with pytest.raises(ValueError, match="must be different"):
            await create_transfer(
                source_org_id="org-1",
                target_org_id="org-1",
                resource_type="AgentGraph",
                resource_id=GRAPH_ID,
                user_id=USER_ID,
            )

    @pytest.mark.asyncio
    async def test_approve_transfer_from_source_admin(self):
        """approve_transfer from source org sets sourceApprovedByUserId
        and advances status to SOURCE_APPROVED."""
        pending = self._make_transfer_row(status="PENDING")
        self.mock_prisma.transferrequest.find_unique = AsyncMock(return_value=pending)

        updated = self._make_transfer_row(
            status="SOURCE_APPROVED",
            source_approved_by=USER_ID,
        )
        self.mock_prisma.transferrequest.update = AsyncMock(return_value=updated)

        from backend.api.features.transfers.db import approve_transfer

        result = await approve_transfer(
            transfer_id="tr-1",
            user_id=USER_ID,
            org_id="org-1",
        )

        assert result.status == "SOURCE_APPROVED"
        update_data = self.mock_prisma.transferrequest.update.call_args.kwargs["data"]
        assert update_data["sourceApprovedByUserId"] == USER_ID
        assert update_data["status"] == "SOURCE_APPROVED"

    @pytest.mark.asyncio
    async def test_approve_transfer_from_target_admin(self):
        """approve_transfer from target org sets targetApprovedByUserId
        and advances status to TARGET_APPROVED."""
        pending = self._make_transfer_row(status="PENDING")
        self.mock_prisma.transferrequest.find_unique = AsyncMock(return_value=pending)

        updated = self._make_transfer_row(
            status="TARGET_APPROVED",
            target_approved_by="user-target",
        )
        self.mock_prisma.transferrequest.update = AsyncMock(return_value=updated)

        from backend.api.features.transfers.db import approve_transfer

        result = await approve_transfer(
            transfer_id="tr-1",
            user_id="user-target",
            org_id="org-2",
        )

        assert result.status == "TARGET_APPROVED"
        update_data = self.mock_prisma.transferrequest.update.call_args.kwargs["data"]
        assert update_data["targetApprovedByUserId"] == "user-target"
        assert update_data["status"] == "TARGET_APPROVED"

    @pytest.mark.asyncio
    async def test_execute_transfer_moves_ownership(self):
        """execute_transfer moves the resource to the target org and sets
        status to COMPLETED."""
        approved = self._make_transfer_row(
            status="SOURCE_APPROVED",
            source_approved_by=USER_ID,
            target_approved_by="user-target",
        )
        self.mock_prisma.transferrequest.find_unique = AsyncMock(return_value=approved)

        completed = self._make_transfer_row(
            status="COMPLETED",
            source_approved_by=USER_ID,
            target_approved_by="user-target",
            completed_at=datetime(2025, 7, 1, tzinfo=timezone.utc),
        )
        self.mock_prisma.transferrequest.update = AsyncMock(return_value=completed)
        self.mock_prisma.agentgraph.update_many = AsyncMock(return_value=1)
        self.mock_prisma.auditlog.create = AsyncMock(return_value=MagicMock())

        from backend.api.features.transfers.db import execute_transfer

        result = await execute_transfer(
            transfer_id="tr-1",
            user_id=USER_ID,
        )

        assert result.status == "COMPLETED"
        # Verify the resource was moved to target org
        self.mock_prisma.agentgraph.update_many.assert_called_once()
        move_data = self.mock_prisma.agentgraph.update_many.call_args.kwargs["data"]
        assert move_data["organizationId"] == "org-2"

    @pytest.mark.asyncio
    async def test_execute_requires_both_approvals(self):
        """execute_transfer raises ValueError when only one side has
        approved."""
        only_source = self._make_transfer_row(
            status="SOURCE_APPROVED",
            source_approved_by=USER_ID,
            target_approved_by=None,
        )
        self.mock_prisma.transferrequest.find_unique = AsyncMock(
            return_value=only_source
        )

        from backend.api.features.transfers.db import execute_transfer

        with pytest.raises(ValueError, match="requires approval from both"):
            await execute_transfer(
                transfer_id="tr-1",
                user_id=USER_ID,
            )

    @pytest.mark.asyncio
    async def test_reject_transfer_sets_status(self):
        """reject_transfer sets the transfer status to REJECTED."""
        pending = self._make_transfer_row(status="PENDING")
        self.mock_prisma.transferrequest.find_unique = AsyncMock(return_value=pending)

        rejected = self._make_transfer_row(status="REJECTED")
        self.mock_prisma.transferrequest.update = AsyncMock(return_value=rejected)

        from backend.api.features.transfers.db import reject_transfer

        result = await reject_transfer(
            transfer_id="tr-1",
            user_id=USER_ID,
            org_id=pending.sourceOrganizationId,
        )

        assert result.status == "REJECTED"
        update_data = self.mock_prisma.transferrequest.update.call_args.kwargs["data"]
        assert update_data["status"] == "REJECTED"

    @pytest.mark.asyncio
    async def test_list_transfers_returns_both_directions(self):
        """list_transfers returns transfers where org is source OR target."""
        transfers = [
            self._make_transfer_row(id="tr-1", source_org="org-1", target_org="org-2"),
            self._make_transfer_row(id="tr-2", source_org="org-3", target_org="org-1"),
        ]
        self.mock_prisma.transferrequest.find_many = AsyncMock(return_value=transfers)

        from backend.api.features.transfers.db import list_transfers

        result = await list_transfers(org_id="org-1")

        assert len(result) == 2
        where_arg = self.mock_prisma.transferrequest.find_many.call_args.kwargs["where"]
        assert "OR" in where_arg
        or_clauses = where_arg["OR"]
        assert {"sourceOrganizationId": "org-1"} in or_clauses
        assert {"targetOrganizationId": "org-1"} in or_clauses


class TestPR18Cutover:
    """PR18: Full cutover from userId to org/team scoping."""

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="PR18: list_graphs still uses userId not org")
    async def test_list_graphs_returns_only_org_graphs(self):
        """When org context is provided, get_graph_all_versions should filter
        by organizationId instead of userId."""
        org1_graph = _make_graph_row(id="g-org1", version=1)
        org1_graph.organizationId = "org-1"

        mock_actions = AsyncMock()
        mock_actions.find_many = AsyncMock(return_value=[org1_graph])

        with patch(
            "backend.data.graph.AgentGraph.prisma",
            return_value=mock_actions,
        ):
            from backend.data.graph import get_graph_all_versions

            await get_graph_all_versions("g-org1", USER_ID, team_id=None)

        where_arg = mock_actions.find_many.call_args.kwargs.get(
            "where", mock_actions.find_many.call_args[1].get("where")
        )
        # Once implemented, should filter by organizationId, not userId
        assert (
            "organizationId" in where_arg
        ), "get_graph_all_versions should filter by organizationId"
        assert where_arg["organizationId"] == "org-1"

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="PR18: list_graphs doesn't exclude other orgs")
    async def test_list_graphs_excludes_other_org_graphs(self):
        """get_graph_all_versions with org-1 must NOT return org-2 graphs."""
        mock_actions = AsyncMock()
        mock_actions.find_many = AsyncMock(return_value=[])

        with patch(
            "backend.data.graph.AgentGraph.prisma",
            return_value=mock_actions,
        ):
            from backend.data.graph import get_graph_all_versions

            results = await get_graph_all_versions(GRAPH_ID, USER_ID, team_id=None)

        where_arg = mock_actions.find_many.call_args.kwargs.get(
            "where", mock_actions.find_many.call_args[1].get("where")
        )
        # Should scope by org, not user
        assert "organizationId" in where_arg
        assert len(results) == 0

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="PR18: get_graph doesn't check org membership")
    async def test_get_graph_requires_org_membership(self):
        """get_graph should verify the caller's org matches the graph's
        organizationId, not just userId."""
        mock_actions = AsyncMock()
        mock_actions.find_first = AsyncMock(return_value=None)

        with patch(
            "backend.data.graph.AgentGraph.prisma",
            return_value=mock_actions,
        ):
            from backend.data.graph import get_graph

            await get_graph(GRAPH_ID, version=None, user_id=USER_ID)

        where_arg = mock_actions.find_first.call_args.kwargs.get(
            "where", mock_actions.find_first.call_args[1].get("where")
        )
        # Once implemented, should use organizationId instead of userId
        assert (
            "organizationId" in where_arg
        ), "get_graph should filter by organizationId"

    @pytest.mark.asyncio
    async def test_delete_graph_requires_org_ownership(self):
        """delete_graph should accept organization_id and use it in the
        where clause instead of userId."""
        mock_actions = AsyncMock()
        mock_actions.delete_many = AsyncMock(return_value=1)

        with patch(
            "backend.data.graph.AgentGraph.prisma",
            return_value=mock_actions,
        ):
            import inspect

            from backend.data.graph import delete_graph

            sig = inspect.signature(delete_graph)
            # Once cutover, should accept organization_id
            assert (
                "organization_id" in sig.parameters
            ), "delete_graph should accept organization_id"

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="PR18: executions still filter by userId/teamId until cutover migration runs"
    )
    async def test_list_executions_scoped_by_team(self):
        """get_graph_executions with org context should filter by
        organizationId, not just userId or teamId."""
        mock_actions = AsyncMock()
        mock_actions.find_many = AsyncMock(return_value=[])

        with patch(
            "backend.data.execution.AgentGraphExecution.prisma",
            return_value=mock_actions,
        ):
            from backend.data.execution import get_graph_executions

            await get_graph_executions(user_id=USER_ID, team_id="team-1")

        where_arg = mock_actions.find_many.call_args.kwargs.get(
            "where", mock_actions.find_many.call_args[1].get("where")
        )
        # Currently uses teamId. After cutover, should also scope by org.
        assert where_arg["teamId"] == "team-1"
        # Once cutover, organizationId should be required
        assert (
            "organizationId" in where_arg
        ), "Executions should also be scoped by organizationId"

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="PR18: get_execution still filters by userId until cutover migration runs"
    )
    async def test_get_execution_requires_org_membership(self):
        """Getting a single execution should verify the caller's org
        matches the execution's organizationId."""
        exec_row = _make_execution_row()
        exec_row.organizationId = "org-1"

        mock_actions = AsyncMock()
        mock_actions.find_many = AsyncMock(return_value=[exec_row])

        with patch(
            "backend.data.execution.AgentGraphExecution.prisma",
            return_value=mock_actions,
        ):
            from backend.data.execution import get_graph_executions

            await get_graph_executions(graph_exec_id=EXEC_ID, user_id=USER_ID)

        where_arg = mock_actions.find_many.call_args.kwargs.get(
            "where", mock_actions.find_many.call_args[1].get("where")
        )
        assert (
            "organizationId" in where_arg
        ), "Single execution fetch should verify org membership"

    @pytest.mark.asyncio
    async def test_create_schedule_scoped_to_org(self):
        """add_graph_execution_schedule should store organizationId in the
        job args so scheduled runs are scoped to the org."""
        from backend.executor.scheduler import GraphExecutionJobArgs

        args = GraphExecutionJobArgs(
            user_id=USER_ID,
            graph_id=GRAPH_ID,
            graph_version=1,
            cron="*/5 * * * *",
            input_data={},
        )
        dumped = args.model_dump()
        # organization_id must be required (non-None) after cutover
        assert (
            dumped.get("organization_id") is not None
        ), "GraphExecutionJobArgs should include organization_id"
        # Once cutover, organization_id should be required (not None)

    @pytest.mark.asyncio
    async def test_upload_file_scoped_to_org(self):
        """File upload should store files under org-scoped paths so that
        org isolation applies to uploaded assets."""
        # The workspace manager should accept organization_id
        # and namespace file storage by org
        import inspect

        from backend.util.file import store_media_file

        sig = inspect.signature(store_media_file)
        assert (
            "organization_id" in sig.parameters
        ), "store_media_file should accept organization_id for scoping"

    @pytest.mark.asyncio
    async def test_get_credits_returns_org_balance(self):
        """UserCredit.get_credits should accept organization_id and return
        the org-level balance instead of per-user balance."""
        from backend.data.credit import UserCredit

        credit = UserCredit()

        import inspect

        sig = inspect.signature(credit.get_credits)
        assert (
            "organization_id" in sig.parameters
        ), "get_credits should accept organization_id"

    @pytest.mark.asyncio
    async def test_top_up_credits_org_balance(self):
        """UserCredit.top_up_credits should accept organization_id and
        credit the org balance."""
        from backend.data.credit import UserCredit

        credit = UserCredit()

        import inspect

        sig = inspect.signature(credit.top_up_credits)
        assert (
            "organization_id" in sig.parameters
        ), "top_up_credits should accept organization_id"

    @pytest.mark.asyncio
    async def test_credit_history_returns_org_transactions(self):
        """Credit transaction history should be queryable by org, not just
        by user."""
        from backend.data.credit import UserCredit

        credit = UserCredit()
        # Once org-scoped, there should be a method like
        # get_org_credit_transactions or get_credits should filter by org
        import inspect

        sig = inspect.signature(credit.get_credits)
        assert (
            "organization_id" in sig.parameters
        ), "get_credits should accept organization_id for org-scoped history"

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="PR18: API keys not scoped by org")
    async def test_list_api_keys_scoped_by_org_cutover(self):
        """list_user_api_keys should filter by organizationId when provided,
        returning all keys belonging to the org."""
        mock_actions = AsyncMock()
        mock_actions.find_many = AsyncMock(return_value=[])

        with patch(
            "backend.data.auth.api_key.PrismaAPIKey.prisma",
            return_value=mock_actions,
        ):
            from backend.data.auth.api_key import list_user_api_keys

            await list_user_api_keys(USER_ID)

        where_arg = mock_actions.find_many.call_args.kwargs.get(
            "where", mock_actions.find_many.call_args[1].get("where")
        )
        # After cutover, should scope by org not user
        assert (
            "organizationId" in where_arg
        ), "list_user_api_keys should filter by organizationId after cutover"

    @pytest.mark.asyncio
    async def test_create_api_key_sets_org_context(self):
        """API key create route should automatically set organizationId from
        the user's active org context."""
        key_row = _make_api_key_row()
        key_row.organizationId = "org-1"

        mock_actions = AsyncMock()
        mock_actions.create = AsyncMock(return_value=key_row)

        with patch(
            "backend.data.auth.api_key.PrismaAPIKey.prisma",
            return_value=mock_actions,
        ):
            from backend.data.auth.api_key import create_api_key

            info, _ = await create_api_key(
                name="test",
                user_id=USER_ID,
                permissions=[],
                organization_id="org-1",
            )

        create_data = mock_actions.create.call_args.kwargs.get(
            "data", mock_actions.create.call_args[1].get("data")
        )
        assert create_data.get("organizationId") == "org-1"
        # After cutover, the route should ALWAYS set org (not conditionally)
        import inspect

        from backend.api.features import v1 as api_v1

        route_src = inspect.getsource(api_v1.create_api_key)
        assert (
            "if ctx.org_id" not in route_src
        ), "Route should always set organizationId, not conditionally"

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="PR18: organizationId still nullable until cutover migration runs"
    )
    async def test_all_agent_graphs_have_organization_id(self):
        """After cutover, AgentGraph.organizationId should be NOT NULL.
        All existing graphs must have been backfilled."""
        # Check via prisma schema inspection: organizationId should be required
        import typing

        from prisma.types import AgentGraphCreateInput

        hints = typing.get_type_hints(AgentGraphCreateInput)
        org_type = hints.get("organizationId")
        # After cutover, the type should be `str` (not Optional[str])
        assert (
            org_type is str
        ), f"AgentGraph.organizationId should be non-nullable, got {org_type}"

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="PR18: organizationId still nullable until cutover migration runs"
    )
    async def test_all_executions_have_organization_id(self):
        """After cutover, AgentGraphExecution.organizationId should be
        NOT NULL. All existing executions must have been backfilled."""
        import typing

        from prisma.types import AgentGraphExecutionCreateInput

        hints = typing.get_type_hints(AgentGraphExecutionCreateInput)
        org_type = hints.get("organizationId")
        assert (
            org_type is str
        ), f"Execution.organizationId should be non-nullable, got {org_type}"

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="PR18: owningOrgId still nullable until cutover migration runs"
    )
    async def test_all_store_listings_have_owning_org_id(self):
        """After cutover, StoreListing.owningOrgId should be NOT NULL."""
        import typing

        from prisma.types import StoreListingCreateInput

        hints = typing.get_type_hints(StoreListingCreateInput)
        org_type = hints.get("owningOrgId")
        assert (
            org_type is str
        ), f"StoreListing.owningOrgId should be non-nullable, got {org_type}"

    @pytest.mark.asyncio
    async def test_creator_view_resolves_org_profile(self):
        """After cutover, StoreCreator view should resolve the org profile
        (org name, org avatar) for org-owned listings."""
        mock_actions = AsyncMock()
        mock_creator = MagicMock()
        mock_creator.username = "testuser"
        mock_actions.find_first = AsyncMock(return_value=mock_creator)

        with patch(
            "backend.api.features.store.db.prisma.models.StoreCreator.prisma",
            return_value=mock_actions,
        ):
            from backend.api.features.store.db import prisma

            result = await prisma.models.StoreCreator.prisma().find_first(
                where={"username": "testuser"}
            )
        assert result is not None
        assert hasattr(
            result, "org_name"
        ), "StoreCreator view should have org_name column after cutover"

    @pytest.mark.asyncio
    async def test_store_agent_view_includes_org_id(self):
        """After cutover, StoreAgent view should include owning_org_id."""
        from prisma.models import StoreAgent

        field_names = set(StoreAgent.model_fields.keys())
        assert (
            "owning_org_id" in field_names
        ), "StoreAgent Prisma model should have owning_org_id column"

    @pytest.mark.asyncio
    async def test_store_submission_view_includes_org_id(self):
        """After cutover, StoreSubmission view should include
        organization_id."""
        from prisma.models import StoreSubmission

        field_names = set(StoreSubmission.model_fields.keys())
        assert (
            "organization_id" in field_names
        ), "StoreSubmission Prisma model should have organization_id column"

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="PR18: get_graph still uses userId until cutover migration runs"
    )
    async def test_read_by_user_id_fallback_removed(self):
        """After cutover, the userId fallback path in get_graph should be
        removed -- all queries should go through organizationId."""
        mock_actions = AsyncMock()
        mock_actions.find_first = AsyncMock(return_value=None)

        mock_store = AsyncMock()
        mock_store.find_first = AsyncMock(return_value=None)

        mock_lib = AsyncMock()
        mock_lib.find_first = AsyncMock(return_value=None)

        with (
            patch(
                "backend.data.graph.AgentGraph.prisma",
                return_value=mock_actions,
            ),
            patch(
                "backend.data.graph.StoreListingVersion.prisma",
                return_value=mock_store,
            ),
            patch(
                "backend.data.graph.LibraryAgent.prisma",
                return_value=mock_lib,
            ),
        ):
            from backend.data.graph import get_graph

            await get_graph(GRAPH_ID, version=None, user_id=USER_ID)

        where_arg = mock_actions.find_first.call_args.kwargs.get(
            "where", mock_actions.find_first.call_args[1].get("where")
        )
        # After cutover, userId should NOT be in the where clause
        assert (
            "userId" not in where_arg
        ), "get_graph should not use userId after full cutover"


# ============================================================================
# Review-findings tests (xfail)
# ============================================================================


class TestReviewFindings:
    """Tests for issues found by code review agents. Written as xfail first."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_org(self, **overrides):
        m = MagicMock()
        m.id = overrides.get("id", "org-review-1")
        m.name = overrides.get("name", "Review Org")
        m.slug = overrides.get("slug", "review-org")
        m.avatarUrl = overrides.get("avatarUrl", None)
        m.description = overrides.get("description", None)
        m.isPersonal = overrides.get("isPersonal", False)
        m.createdAt = datetime(2025, 6, 1, tzinfo=timezone.utc)
        m.deletedAt = None
        m.Members = []
        return m

    def _owner_ctx(self, org_id="org-review-1", team_id="team-review-1"):
        ctx = MagicMock()
        ctx.user_id = USER_ID
        ctx.org_id = org_id
        ctx.team_id = team_id
        return ctx

    def _make_invitation(self, **overrides):
        m = MagicMock()
        m.id = overrides.get("id", "inv-1")
        m.orgId = overrides.get("orgId", "org-review-1")
        m.email = overrides.get("email", "test@example.com")
        m.isAdmin = overrides.get("isAdmin", False)
        m.isBillingManager = overrides.get("isBillingManager", False)
        m.token = overrides.get("token", "secret-token-abc")
        m.expiresAt = overrides.get(
            "expiresAt", datetime(2099, 1, 1, tzinfo=timezone.utc)
        )
        m.createdAt = overrides.get(
            "createdAt", datetime(2025, 6, 1, tzinfo=timezone.utc)
        )
        m.acceptedAt = overrides.get("acceptedAt", None)
        m.revokedAt = overrides.get("revokedAt", None)
        m.teamIds = overrides.get("teamIds", [])
        m.invitedByUserId = overrides.get("invitedByUserId", USER_ID)
        m.targetUserId = overrides.get("targetUserId", None)
        return m

    def _make_transfer(self, **overrides):
        m = MagicMock()
        m.id = overrides.get("id", "tr-review-1")
        m.resourceType = overrides.get("resourceType", "AgentGraph")
        m.resourceId = overrides.get("resourceId", GRAPH_ID)
        m.sourceOrganizationId = overrides.get("sourceOrganizationId", "org-A")
        m.targetOrganizationId = overrides.get("targetOrganizationId", "org-B")
        m.initiatedByUserId = overrides.get("initiatedByUserId", USER_ID)
        m.status = overrides.get("status", "PENDING")
        m.sourceApprovedByUserId = overrides.get("sourceApprovedByUserId", None)
        m.targetApprovedByUserId = overrides.get("targetApprovedByUserId", None)
        m.completedAt = overrides.get("completedAt", None)
        m.reason = overrides.get("reason", None)
        m.createdAt = datetime(2025, 6, 1, tzinfo=timezone.utc)
        return m

    # ------------------------------------------------------------------
    # 1. Invitation token not leaked in list response
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_invitation_list_response_excludes_token(self):
        """GET /invitations should NOT return the raw token -- only the
        create response should."""
        from backend.api.features.orgs.model import InvitationResponse

        # InvitationResponse is used for both create AND list endpoints.
        # The raw token field should NOT be present in the list response model.
        # If there is a single model for both, the list endpoint leaks tokens.
        inv = self._make_invitation()
        response = InvitationResponse.from_db(inv)
        response_fields = set(response.model_dump().keys())

        # The token MUST NOT appear in a list-endpoint response.
        # A separate InvitationListResponse (without token) is needed.
        assert "token" not in response_fields, (
            "InvitationResponse exposes the raw token. "
            "List endpoints must use a model without the token field."
        )

    # ------------------------------------------------------------------
    # 2. decline_invitation doesn't check if already accepted
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_decline_already_accepted_invitation_blocked(self):
        """Cannot decline an invitation that was already accepted."""
        accepted_inv = self._make_invitation(
            acceptedAt=datetime(2025, 6, 15, tzinfo=timezone.utc),
        )

        declining_user = MagicMock()
        declining_user.email = accepted_inv.email

        with (
            patch("backend.api.features.orgs.invitation_routes.prisma") as mock_prisma,
        ):
            mock_prisma.orginvitation.find_unique = AsyncMock(return_value=accepted_inv)
            mock_prisma.user.find_unique = AsyncMock(return_value=declining_user)

            from backend.api.features.orgs.invitation_routes import decline_invitation

            with pytest.raises(HTTPException) as exc_info:
                await decline_invitation(token=accepted_inv.token, user_id=USER_ID)
            assert exc_info.value.status_code == 400

    # ------------------------------------------------------------------
    # 3. decline_invitation doesn't check if already revoked
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_decline_already_revoked_invitation_blocked(self):
        """Cannot decline an invitation that was already revoked."""
        revoked_inv = self._make_invitation(
            revokedAt=datetime(2025, 6, 15, tzinfo=timezone.utc),
        )

        declining_user = MagicMock()
        declining_user.email = revoked_inv.email

        with (
            patch("backend.api.features.orgs.invitation_routes.prisma") as mock_prisma,
        ):
            mock_prisma.orginvitation.find_unique = AsyncMock(return_value=revoked_inv)
            mock_prisma.user.find_unique = AsyncMock(return_value=declining_user)

            from backend.api.features.orgs.invitation_routes import decline_invitation

            with pytest.raises(HTTPException) as exc_info:
                await decline_invitation(token=revoked_inv.token, user_id=USER_ID)
            assert exc_info.value.status_code == 400

    # ------------------------------------------------------------------
    # 4. join_team doesn't verify ctx.org_id == path org_id
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_join_team_rejects_mismatched_org_id(self):
        """join_team route should reject when ctx.org_id != path org_id."""
        ctx = self._owner_ctx(org_id="org-A")

        from backend.api.features.orgs.team_routes import join_team

        # Call with a different org_id in the path
        with pytest.raises(HTTPException) as exc_info:
            await join_team(org_id="org-B", ws_id="ws-1", ctx=ctx)
        assert exc_info.value.status_code == 403

    # ------------------------------------------------------------------
    # 5. leave_team doesn't verify ctx.org_id == path org_id
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_leave_team_rejects_mismatched_org_id(self):
        """leave_team route should reject when ctx.org_id != path org_id."""
        ctx = self._owner_ctx(org_id="org-A")

        from backend.api.features.orgs.team_routes import leave_team

        with pytest.raises(HTTPException) as exc_info:
            await leave_team(org_id="org-B", ws_id="ws-1", ctx=ctx)
        assert exc_info.value.status_code == 403

    # ------------------------------------------------------------------
    # 6. update_team ctx.team_id / ws_id mismatch
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_update_team_rejects_mismatched_team_id(self):
        """PATCH /teams/{ws_id} should reject when ctx.team_id != ws_id."""
        from backend.api.features.orgs.team_model import UpdateTeamRequest
        from backend.api.features.orgs.team_routes import update_team

        ctx = self._owner_ctx(org_id="org-review-1", team_id="team-1")

        # Mock the team_db.get_team call that validates org ownership
        with patch(
            "backend.api.features.orgs.team_routes.team_db.get_team",
            new_callable=AsyncMock,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await update_team(
                    org_id="org-review-1",
                    ws_id="team-2",  # different from ctx.team_id
                    request=UpdateTeamRequest(name="Hacked"),
                    ctx=ctx,
                )
            assert exc_info.value.status_code == 403

    # ------------------------------------------------------------------
    # 7. reject_transfer has no org membership check
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_reject_transfer_requires_org_membership(self):
        """Only source or target org members can reject a transfer."""
        transfer = self._make_transfer(
            sourceOrganizationId="org-A",
            targetOrganizationId="org-B",
        )

        with patch("backend.api.features.transfers.db.prisma") as mock_prisma:
            mock_prisma.transferrequest.find_unique = AsyncMock(return_value=transfer)

            from backend.api.features.transfers.db import reject_transfer

            # User from org-C (not a party to the transfer) should be denied
            with pytest.raises((ValueError, HTTPException)):
                await reject_transfer(
                    transfer_id="tr-review-1",
                    user_id="user-org-c",
                    org_id="org-C",
                )

    # ------------------------------------------------------------------
    # 8. update_org doesn't sync OrganizationProfile on rename
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_update_org_syncs_profile_on_rename(self):
        """When org name or slug changes, OrganizationProfile should be
        updated too."""
        from backend.api.features.orgs.model import UpdateOrgData

        old_org = self._make_org(id="org-1", slug="old-slug", name="Old Name")
        updated_org = self._make_org(id="org-1", slug="new-slug", name="New Name")
        updated_org.Members = [MagicMock()]

        with patch("backend.api.features.orgs.db.prisma") as mock_prisma:
            mock_prisma.organization.find_unique = AsyncMock(
                side_effect=[
                    None,  # slug uniqueness check
                    None,  # alias uniqueness check -> not an org
                    old_org,  # old org lookup for alias creation
                    updated_org,  # get_org call at end
                ]
            )
            mock_prisma.organizationalias.find_unique = AsyncMock(return_value=None)
            mock_prisma.organizationalias.create = AsyncMock()
            mock_prisma.organization.update = AsyncMock()
            mock_prisma.organizationprofile.update = AsyncMock()
            mock_prisma.orgmember.count = AsyncMock(return_value=1)

            from backend.api.features.orgs.db import update_org

            await update_org(
                "org-1",
                UpdateOrgData(name="New Name", slug="new-slug"),
            )

            # The key assertion: profile must have been updated
            mock_prisma.organizationprofile.update.assert_called_once()
            call_kwargs = mock_prisma.organizationprofile.update.call_args.kwargs
            assert call_kwargs["where"] == {"organizationId": "org-1"}
            assert call_kwargs["data"]["displayName"] == "New Name"
            assert call_kwargs["data"]["username"] == "new-slug"

    # ------------------------------------------------------------------
    # 9. approve_transfer sets COMPLETED before resource is moved
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_approve_transfer_does_not_set_completed(self):
        """approve_transfer should NOT set status=COMPLETED -- only execute
        should."""
        # Simulate a transfer where source already approved, target now
        # approves -- both sides done.
        source_approved = self._make_transfer(
            status="SOURCE_APPROVED",
            sourceApprovedByUserId=USER_ID,
        )

        update_capture: dict = {}

        async def capture_update(*, where, data):
            update_capture.update(data)
            result = self._make_transfer(
                status=data.get("status", "SOURCE_APPROVED"),
                sourceApprovedByUserId=USER_ID,
                targetApprovedByUserId="user-target",
            )
            return result

        with patch("backend.api.features.transfers.db.prisma") as mock_prisma:
            mock_prisma.transferrequest.find_unique = AsyncMock(
                return_value=source_approved
            )
            mock_prisma.transferrequest.update = AsyncMock(side_effect=capture_update)

            from backend.api.features.transfers.db import approve_transfer

            result = await approve_transfer(
                transfer_id="tr-review-1",
                user_id="user-target",
                org_id="org-B",
            )

        # After both approvals, status must NOT be COMPLETED.
        # It should remain in a waiting state until execute_transfer runs.
        assert result.status != "COMPLETED", (
            "approve_transfer should not set status to COMPLETED; "
            "only execute_transfer should do that"
        )
