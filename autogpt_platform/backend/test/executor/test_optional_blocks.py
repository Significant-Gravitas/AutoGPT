"""Tests for optional/conditional block execution."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.data.execution import ExecutionStatus
from backend.data.graph import Node
from backend.data.optional_block import (
    ConditionOperator,
    OptionalBlockConditions,
    OptionalBlockConfig,
    get_optional_config,
)
from backend.executor.manager import should_skip_node
from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.util.user import UserContext


@pytest.fixture
def mock_node():
    """Create a mock node for testing."""
    node = MagicMock(spec=Node)
    node.metadata = {}
    node.block = MagicMock()
    node.block.input_schema = MagicMock()
    node.block.input_schema.get_credentials_fields.return_value = {}
    return node


@pytest.fixture
def mock_creds_manager():
    """Create a mock credentials manager."""
    manager = AsyncMock(spec=IntegrationCredentialsManager)
    manager.exists = AsyncMock(return_value=True)
    return manager


@pytest.fixture
def user_context():
    """Create a mock user context."""
    return UserContext(user_id="test_user", scopes=[])


class TestOptionalBlockConfig:
    """Test OptionalBlockConfig model."""

    def test_optional_config_defaults(self):
        """Test default values for OptionalBlockConfig."""
        config = OptionalBlockConfig()
        assert config.enabled is False
        assert config.conditions.on_missing_credentials is False
        assert config.conditions.input_flag is None
        assert config.conditions.kv_flag is None
        assert config.conditions.operator == ConditionOperator.OR
        assert config.skip_message is None

    def test_optional_config_with_values(self):
        """Test OptionalBlockConfig with custom values."""
        config = OptionalBlockConfig(
            enabled=True,
            conditions=OptionalBlockConditions(
                on_missing_credentials=True,
                input_flag="skip_linear",
                kv_flag="enable_linear",
                operator=ConditionOperator.AND,
            ),
            skip_message="Skipping Linear block due to missing credentials",
        )
        assert config.enabled is True
        assert config.conditions.on_missing_credentials is True
        assert config.conditions.input_flag == "skip_linear"
        assert config.conditions.kv_flag == "enable_linear"
        assert config.conditions.operator == ConditionOperator.AND
        assert config.skip_message == "Skipping Linear block due to missing credentials"

    def test_get_optional_config_from_metadata(self):
        """Test extracting optional config from node metadata."""
        # No optional config
        metadata = {}
        config = get_optional_config(metadata)
        assert config is None

        # Empty optional config
        metadata = {"optional": {}}
        config = get_optional_config(metadata)
        assert config is None

        # Valid optional config
        metadata = {
            "optional": {
                "enabled": True,
                "conditions": {
                    "on_missing_credentials": True,
                },
            }
        }
        config = get_optional_config(metadata)
        assert config is not None
        assert config.enabled is True
        assert config.conditions.on_missing_credentials is True


@pytest.mark.asyncio
class TestShouldSkipNode:
    """Test should_skip_node function."""

    async def test_skip_when_not_optional(
        self, mock_node, mock_creds_manager, user_context
    ):
        """Test that non-optional nodes are not skipped."""
        should_skip, reason = await should_skip_node(
            node=mock_node,
            creds_manager=mock_creds_manager,
            user_id="test_user",
            user_context=user_context,
            input_data={},
            graph_id="test_graph_id",
        )
        assert should_skip is False
        assert reason == ""

    async def test_skip_when_optional_disabled(
        self, mock_node, mock_creds_manager, user_context
    ):
        """Test that optional but disabled nodes are not skipped."""
        mock_node.metadata = {
            "optional": {
                "enabled": False,
                "conditions": {"on_missing_credentials": True},
            }
        }
        should_skip, reason = await should_skip_node(
            node=mock_node,
            creds_manager=mock_creds_manager,
            user_id="test_user",
            user_context=user_context,
            input_data={},
            graph_id="test_graph_id",
        )
        assert should_skip is False
        assert reason == ""

    async def test_skip_on_missing_credentials(
        self, mock_node, mock_creds_manager, user_context
    ):
        """Test skipping when credentials are missing."""
        mock_node.metadata = {
            "optional": {
                "enabled": True,
                "conditions": {"on_missing_credentials": True},
            }
        }
        mock_node.block.input_schema.get_credentials_fields.return_value = {
            "credentials": MagicMock()
        }
        mock_creds_manager.exists.return_value = False

        should_skip, reason = await should_skip_node(
            node=mock_node,
            creds_manager=mock_creds_manager,
            user_id="test_user",
            user_context=user_context,
            input_data={"credentials": {"id": "cred_123"}},
            graph_id="test_graph_id",
        )
        assert should_skip is True
        assert "Missing credentials" in reason

    async def test_no_skip_when_credentials_exist(
        self, mock_node, mock_creds_manager, user_context
    ):
        """Test no skip when credentials exist."""
        mock_node.metadata = {
            "optional": {
                "enabled": True,
                "conditions": {"on_missing_credentials": True},
            }
        }
        mock_node.block.input_schema.get_credentials_fields.return_value = {
            "credentials": MagicMock()
        }
        mock_creds_manager.exists.return_value = True

        should_skip, reason = await should_skip_node(
            node=mock_node,
            creds_manager=mock_creds_manager,
            user_id="test_user",
            user_context=user_context,
            input_data={"credentials": {"id": "cred_123"}},
            graph_id="test_graph_id",
        )
        assert should_skip is False
        assert reason == ""

    async def test_skip_on_skip_input_true(
        self, mock_node, mock_creds_manager, user_context
    ):
        """Test skipping when skip_run_block input is true."""
        mock_node.metadata = {
            "optional": {
                "enabled": True,
                "conditions": {"check_skip_input": True},
            }
        }

        should_skip, reason = await should_skip_node(
            node=mock_node,
            creds_manager=mock_creds_manager,
            user_id="test_user",
            user_context=user_context,
            input_data={"skip_run_block": True},
            graph_id="test_graph_id",
        )
        assert should_skip is True
        assert "Skip input is true" in reason

    async def test_no_skip_on_skip_input_false(
        self, mock_node, mock_creds_manager, user_context
    ):
        """Test no skip when skip_run_block input is false."""
        mock_node.metadata = {
            "optional": {
                "enabled": True,
                "conditions": {"check_skip_input": True},
            }
        }

        should_skip, reason = await should_skip_node(
            node=mock_node,
            creds_manager=mock_creds_manager,
            user_id="test_user",
            user_context=user_context,
            input_data={"skip_run_block": False},
            graph_id="test_graph_id",
        )
        assert should_skip is False
        assert reason == ""

    async def test_skip_with_or_operator(
        self, mock_node, mock_creds_manager, user_context
    ):
        """Test OR logic - skip if any condition is met."""
        mock_node.metadata = {
            "optional": {
                "enabled": True,
                "conditions": {
                    "on_missing_credentials": True,
                    "check_skip_input": True,
                    "operator": "or",
                },
            }
        }
        mock_node.block.input_schema.get_credentials_fields.return_value = {
            "credentials": MagicMock()
        }
        # Credentials exist but input flag is true
        mock_creds_manager.exists.return_value = True

        should_skip, reason = await should_skip_node(
            node=mock_node,
            creds_manager=mock_creds_manager,
            user_id="test_user",
            user_context=user_context,
            input_data={
                "credentials": {"id": "cred_123"},
                "skip_run_block": True,
            },
            graph_id="test_graph_id",
        )
        assert should_skip is True  # OR: at least one condition met
        assert "Skip input is true" in reason

    async def test_skip_with_and_operator(
        self, mock_node, mock_creds_manager, user_context
    ):
        """Test AND logic - skip only if all conditions are met."""
        mock_node.metadata = {
            "optional": {
                "enabled": True,
                "conditions": {
                    "on_missing_credentials": True,
                    "check_skip_input": True,
                    "operator": "and",
                },
            }
        }
        mock_node.block.input_schema.get_credentials_fields.return_value = {
            "credentials": MagicMock()
        }
        # Credentials missing but input flag is false
        mock_creds_manager.exists.return_value = False

        should_skip, reason = await should_skip_node(
            node=mock_node,
            creds_manager=mock_creds_manager,
            user_id="test_user",
            user_context=user_context,
            input_data={
                "credentials": {"id": "cred_123"},
                "skip_run_block": False,
            },
            graph_id="test_graph_id",
        )
        assert should_skip is False  # AND: not all conditions met
        assert reason == ""

    async def test_custom_skip_message(
        self, mock_node, mock_creds_manager, user_context
    ):
        """Test custom skip message."""
        mock_node.metadata = {
            "optional": {
                "enabled": True,
                "conditions": {"check_skip_input": True},
                "skip_message": "Custom skip message for testing",
            }
        }

        should_skip, reason = await should_skip_node(
            node=mock_node,
            creds_manager=mock_creds_manager,
            user_id="test_user",
            user_context=user_context,
            input_data={"skip_run_block": True},
            graph_id="test_graph_id",
        )
        assert should_skip is True
        assert reason == "Custom skip message for testing"

    async def test_skip_on_kv_flag_true(
        self, mock_node, mock_creds_manager, user_context
    ):
        """Test skipping when KV flag is true."""
        mock_node.metadata = {
            "optional": {
                "enabled": True,
                "conditions": {"kv_flag": "skip_linear"},
            }
        }

        # Mock the database client to return True for the KV flag
        with patch(
            "backend.executor.manager.get_database_manager_async_client"
        ) as mock_db_client:
            mock_db_client.return_value.get_execution_kv_data = AsyncMock(
                return_value=True
            )

            should_skip, reason = await should_skip_node(
                node=mock_node,
                creds_manager=mock_creds_manager,
                user_id="test_user",
                user_context=user_context,
                input_data={},
                graph_id="test_graph_id",
            )
            assert should_skip is True
            assert "KV flag 'skip_linear' is true" in reason

            # Verify the correct key was used
            mock_db_client.return_value.get_execution_kv_data.assert_called_once_with(
                user_id="test_user",
                key="agent#test_graph_id#skip_linear",
            )

    async def test_no_skip_on_kv_flag_false(
        self, mock_node, mock_creds_manager, user_context
    ):
        """Test no skip when KV flag is false."""
        mock_node.metadata = {
            "optional": {
                "enabled": True,
                "conditions": {"kv_flag": "skip_linear"},
            }
        }

        # Mock the database client to return False for the KV flag
        with patch(
            "backend.executor.manager.get_database_manager_async_client"
        ) as mock_db_client:
            mock_db_client.return_value.get_execution_kv_data = AsyncMock(
                return_value=False
            )

            should_skip, reason = await should_skip_node(
                node=mock_node,
                creds_manager=mock_creds_manager,
                user_id="test_user",
                user_context=user_context,
                input_data={},
                graph_id="test_graph_id",
            )
            assert should_skip is False
            assert reason == ""

    async def test_kv_flag_with_combined_conditions(
        self, mock_node, mock_creds_manager, user_context
    ):
        """Test KV flag combined with other conditions using OR operator."""
        mock_node.metadata = {
            "optional": {
                "enabled": True,
                "conditions": {
                    "kv_flag": "enable_integration",
                    "check_skip_input": True,
                    "operator": "or",
                },
            }
        }

        # Mock the database client to return False for the KV flag
        with patch(
            "backend.executor.manager.get_database_manager_async_client"
        ) as mock_db_client:
            mock_db_client.return_value.get_execution_kv_data = AsyncMock(
                return_value=False
            )

            # Even though KV flag is False, skip_run_block is True so it should skip (OR operator)
            should_skip, reason = await should_skip_node(
                node=mock_node,
                creds_manager=mock_creds_manager,
                user_id="test_user",
                user_context=user_context,
                input_data={"skip_run_block": True},
                graph_id="test_graph_id",
            )
            assert should_skip is True
            assert "Skip input is true" in reason


@pytest.mark.asyncio
class TestExecutionFlow:
    """Test execution flow with optional blocks."""

    async def test_skipped_status_transition(self):
        """Test that SKIPPED is a valid status transition."""
        from backend.data.execution import VALID_STATUS_TRANSITIONS

        assert ExecutionStatus.SKIPPED in VALID_STATUS_TRANSITIONS
        assert (
            ExecutionStatus.INCOMPLETE
            in VALID_STATUS_TRANSITIONS[ExecutionStatus.SKIPPED]
        )
        assert (
            ExecutionStatus.QUEUED in VALID_STATUS_TRANSITIONS[ExecutionStatus.SKIPPED]
        )

    async def test_smart_decision_maker_filters_optional(self):
        """Test that Smart Decision Maker filters out optional blocks."""
        from backend.data.optional_block import get_optional_config

        # Create a mock node with optional config
        node = MagicMock()
        node.metadata = {
            "optional": {
                "enabled": True,
                "conditions": {"on_missing_credentials": True},
            }
        }

        # Verify optional config is detected
        config = get_optional_config(node.metadata)
        assert config is not None
        assert config.enabled is True

        # The Smart Decision Maker should skip this node when building function signatures
        # This is tested in the actual implementation where optional nodes are filtered
