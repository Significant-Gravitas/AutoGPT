"""
Tests for activity status generator functionality.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks.llm import LLMResponse
from backend.data.execution import ExecutionStatus, NodeExecutionResult
from backend.data.model import GraphExecutionStats
from backend.executor.activity_status_generator import (
    _build_execution_summary,
    _call_llm_direct,
    generate_activity_status_for_execution,
)


@pytest.fixture
def mock_node_executions():
    """Create mock node executions for testing."""
    return [
        NodeExecutionResult(
            user_id="test_user",
            graph_id="test_graph",
            graph_version=1,
            graph_exec_id="test_exec",
            node_exec_id="123e4567-e89b-12d3-a456-426614174001",
            node_id="456e7890-e89b-12d3-a456-426614174002",
            block_id="789e1234-e89b-12d3-a456-426614174003",
            status=ExecutionStatus.COMPLETED,
            input_data={"user_input": "Hello, world!"},
            output_data={"processed_input": ["Hello, world!"]},
            add_time=datetime.now(timezone.utc),
            queue_time=None,
            start_time=None,
            end_time=None,
        ),
        NodeExecutionResult(
            user_id="test_user",
            graph_id="test_graph",
            graph_version=1,
            graph_exec_id="test_exec",
            node_exec_id="234e5678-e89b-12d3-a456-426614174004",
            node_id="567e8901-e89b-12d3-a456-426614174005",
            block_id="890e2345-e89b-12d3-a456-426614174006",
            status=ExecutionStatus.COMPLETED,
            input_data={"data": "Hello, world!"},
            output_data={"result": ["Processed data"]},
            add_time=datetime.now(timezone.utc),
            queue_time=None,
            start_time=None,
            end_time=None,
        ),
        NodeExecutionResult(
            user_id="test_user",
            graph_id="test_graph",
            graph_version=1,
            graph_exec_id="test_exec",
            node_exec_id="345e6789-e89b-12d3-a456-426614174007",
            node_id="678e9012-e89b-12d3-a456-426614174008",
            block_id="901e3456-e89b-12d3-a456-426614174009",
            status=ExecutionStatus.FAILED,
            input_data={"final_data": "Processed data"},
            output_data={
                "error": ["Connection timeout: Unable to reach external service"]
            },
            add_time=datetime.now(timezone.utc),
            queue_time=None,
            start_time=None,
            end_time=None,
        ),
    ]


@pytest.fixture
def mock_execution_stats():
    """Create mock execution stats for testing."""
    return GraphExecutionStats(
        walltime=2.5,
        cputime=1.8,
        nodes_walltime=2.0,
        nodes_cputime=1.5,
        node_count=3,
        node_error_count=1,
        cost=10,
        error=None,
    )


@pytest.fixture
def mock_execution_stats_with_graph_error():
    """Create mock execution stats with graph-level error."""
    return GraphExecutionStats(
        walltime=2.5,
        cputime=1.8,
        nodes_walltime=2.0,
        nodes_cputime=1.5,
        node_count=3,
        node_error_count=1,
        cost=10,
        error="Graph execution failed: Invalid API credentials",
    )


@pytest.fixture
def mock_blocks():
    """Create mock blocks for testing."""
    input_block = MagicMock()
    input_block.name = "AgentInputBlock"
    input_block.description = "Handles user input"

    process_block = MagicMock()
    process_block.name = "ProcessingBlock"
    process_block.description = "Processes data"

    output_block = MagicMock()
    output_block.name = "AgentOutputBlock"
    output_block.description = "Provides output to user"

    return {
        "789e1234-e89b-12d3-a456-426614174003": input_block,
        "890e2345-e89b-12d3-a456-426614174006": process_block,
        "901e3456-e89b-12d3-a456-426614174009": output_block,
        "process_block_id": process_block,  # Keep old key for different error format test
    }


class TestBuildExecutionSummary:
    """Tests for _build_execution_summary function."""

    def test_build_summary_with_successful_execution(
        self, mock_node_executions, mock_execution_stats, mock_blocks
    ):
        """Test building summary for successful execution."""
        # Create mock links with realistic UUIDs
        mock_links = [
            MagicMock(
                source_id="456e7890-e89b-12d3-a456-426614174002",
                sink_id="567e8901-e89b-12d3-a456-426614174005",
                source_name="output",
                sink_name="input",
                is_static=False,
            )
        ]

        with patch(
            "backend.executor.activity_status_generator.get_block"
        ) as mock_get_block:
            mock_get_block.side_effect = lambda block_id: mock_blocks.get(block_id)

            summary = _build_execution_summary(
                mock_node_executions[:2],
                mock_execution_stats,
                "Test Graph",
                "A test graph for processing",
                mock_links,
                ExecutionStatus.COMPLETED,
            )

            # Check graph info
            assert summary["graph_info"]["name"] == "Test Graph"
            assert summary["graph_info"]["description"] == "A test graph for processing"

            # Check nodes with per-node counts
            assert len(summary["nodes"]) == 2
            assert summary["nodes"][0]["block_name"] == "AgentInputBlock"
            assert summary["nodes"][0]["execution_count"] == 1
            assert summary["nodes"][0]["error_count"] == 0
            assert summary["nodes"][1]["block_name"] == "ProcessingBlock"
            assert summary["nodes"][1]["execution_count"] == 1
            assert summary["nodes"][1]["error_count"] == 0

            # Check node relations (UUIDs are truncated to first segment)
            assert len(summary["node_relations"]) == 1
            assert (
                summary["node_relations"][0]["source_node_id"] == "456e7890"
            )  # Truncated
            assert (
                summary["node_relations"][0]["sink_node_id"] == "567e8901"
            )  # Truncated
            assert (
                summary["node_relations"][0]["source_block_name"] == "AgentInputBlock"
            )
            assert summary["node_relations"][0]["sink_block_name"] == "ProcessingBlock"

            # Check overall status
            assert summary["overall_status"]["total_nodes_in_graph"] == 2
            assert summary["overall_status"]["total_executions"] == 3
            assert summary["overall_status"]["total_errors"] == 1
            assert summary["overall_status"]["execution_time_seconds"] == 2.5
            assert summary["overall_status"]["graph_execution_status"] == "COMPLETED"

            # Check input/output data (using actual node UUIDs)
            assert (
                "456e7890-e89b-12d3-a456-426614174002_inputs"
                in summary["input_output_data"]
            )
            assert (
                "456e7890-e89b-12d3-a456-426614174002_outputs"
                in summary["input_output_data"]
            )

    def test_build_summary_with_failed_execution(
        self, mock_node_executions, mock_execution_stats, mock_blocks
    ):
        """Test building summary for execution with failures."""
        mock_links = []  # No links for this test

        with patch(
            "backend.executor.activity_status_generator.get_block"
        ) as mock_get_block:
            mock_get_block.side_effect = lambda block_id: mock_blocks.get(block_id)

            summary = _build_execution_summary(
                mock_node_executions,
                mock_execution_stats,
                "Failed Graph",
                "Test with failures",
                mock_links,
                ExecutionStatus.FAILED,
            )

            # Check that errors are now in node's recent_errors field
            # Find the output node (with truncated UUID)
            output_node = next(
                n for n in summary["nodes"] if n["node_id"] == "678e9012"  # Truncated
            )
            assert output_node["error_count"] == 1
            assert output_node["execution_count"] == 1

            # Check recent_errors field
            assert "recent_errors" in output_node
            assert len(output_node["recent_errors"]) == 1
            assert (
                output_node["recent_errors"][0]["error"]
                == "Connection timeout: Unable to reach external service"
            )
            assert (
                "execution_id" in output_node["recent_errors"][0]
            )  # Should include execution ID

    def test_build_summary_with_missing_blocks(
        self, mock_node_executions, mock_execution_stats
    ):
        """Test building summary when blocks are missing."""
        with patch(
            "backend.executor.activity_status_generator.get_block"
        ) as mock_get_block:
            mock_get_block.return_value = None

            summary = _build_execution_summary(
                mock_node_executions,
                mock_execution_stats,
                "Missing Blocks Graph",
                "Test with missing blocks",
                [],
                ExecutionStatus.COMPLETED,
            )

            # Should handle missing blocks gracefully
            assert len(summary["nodes"]) == 0
            # No top-level errors field anymore, errors are in nodes' recent_errors
            assert summary["graph_info"]["name"] == "Missing Blocks Graph"

    def test_build_summary_with_graph_error(
        self, mock_node_executions, mock_execution_stats_with_graph_error, mock_blocks
    ):
        """Test building summary with graph-level error."""
        mock_links = []

        with patch(
            "backend.executor.activity_status_generator.get_block"
        ) as mock_get_block:
            mock_get_block.side_effect = lambda block_id: mock_blocks.get(block_id)

            summary = _build_execution_summary(
                mock_node_executions,
                mock_execution_stats_with_graph_error,
                "Graph with Error",
                "Test with graph error",
                mock_links,
                ExecutionStatus.FAILED,
            )

            # Check that graph error is included in overall status
            assert summary["overall_status"]["has_errors"] is True
            assert (
                summary["overall_status"]["graph_error"]
                == "Graph execution failed: Invalid API credentials"
            )
            assert summary["overall_status"]["total_errors"] == 1
            assert summary["overall_status"]["graph_execution_status"] == "FAILED"

    def test_build_summary_with_different_error_formats(
        self, mock_execution_stats, mock_blocks
    ):
        """Test building summary with different error formats."""
        # Create node executions with different error formats and realistic UUIDs
        mock_executions = [
            NodeExecutionResult(
                user_id="test_user",
                graph_id="test_graph",
                graph_version=1,
                graph_exec_id="test_exec",
                node_exec_id="111e2222-e89b-12d3-a456-426614174010",
                node_id="333e4444-e89b-12d3-a456-426614174011",
                block_id="process_block_id",
                status=ExecutionStatus.FAILED,
                input_data={},
                output_data={"error": ["Simple string error message"]},
                add_time=datetime.now(timezone.utc),
                queue_time=None,
                start_time=None,
                end_time=None,
            ),
            NodeExecutionResult(
                user_id="test_user",
                graph_id="test_graph",
                graph_version=1,
                graph_exec_id="test_exec",
                node_exec_id="555e6666-e89b-12d3-a456-426614174012",
                node_id="777e8888-e89b-12d3-a456-426614174013",
                block_id="process_block_id",
                status=ExecutionStatus.FAILED,
                input_data={},
                output_data={},  # No error in output
                add_time=datetime.now(timezone.utc),
                queue_time=None,
                start_time=None,
                end_time=None,
            ),
        ]

        with patch(
            "backend.executor.activity_status_generator.get_block"
        ) as mock_get_block:
            mock_get_block.side_effect = lambda block_id: mock_blocks.get(block_id)

            summary = _build_execution_summary(
                mock_executions,
                mock_execution_stats,
                "Error Test Graph",
                "Testing error formats",
                [],
                ExecutionStatus.FAILED,
            )

            # Check different error formats - errors are now in nodes' recent_errors
            error_nodes = [n for n in summary["nodes"] if n.get("recent_errors")]
            assert len(error_nodes) == 2

            # String error format - find node with truncated ID
            string_error_node = next(
                n for n in summary["nodes"] if n["node_id"] == "333e4444"  # Truncated
            )
            assert len(string_error_node["recent_errors"]) == 1
            assert (
                string_error_node["recent_errors"][0]["error"]
                == "Simple string error message"
            )

            # No error output format - find node with truncated ID
            no_error_node = next(
                n for n in summary["nodes"] if n["node_id"] == "777e8888"  # Truncated
            )
            assert len(no_error_node["recent_errors"]) == 1
            assert no_error_node["recent_errors"][0]["error"] == "Unknown error"


class TestLLMCall:
    """Tests for LLM calling functionality."""

    @pytest.mark.asyncio
    async def test_call_llm_direct_success(self):
        """Test successful LLM call."""
        from pydantic import SecretStr

        from backend.data.model import APIKeyCredentials

        mock_response = LLMResponse(
            raw_response={},
            prompt=[],
            response="Agent successfully processed user input and generated response.",
            tool_calls=None,
            prompt_tokens=50,
            completion_tokens=20,
        )

        with patch(
            "backend.executor.activity_status_generator.llm_call"
        ) as mock_llm_call:
            mock_llm_call.return_value = mock_response

            credentials = APIKeyCredentials(
                id="test",
                provider="openai",
                api_key=SecretStr("test_key"),
                title="Test",
            )

            prompt = [{"role": "user", "content": "Test prompt"}]

            result = await _call_llm_direct(credentials, prompt)

            assert (
                result
                == "Agent successfully processed user input and generated response."
            )
            mock_llm_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_llm_direct_no_response(self):
        """Test LLM call with no response."""
        from pydantic import SecretStr

        from backend.data.model import APIKeyCredentials

        with patch(
            "backend.executor.activity_status_generator.llm_call"
        ) as mock_llm_call:
            mock_llm_call.return_value = None

            credentials = APIKeyCredentials(
                id="test",
                provider="openai",
                api_key=SecretStr("test_key"),
                title="Test",
            )

            prompt = [{"role": "user", "content": "Test prompt"}]

            result = await _call_llm_direct(credentials, prompt)

            assert result == "Unable to generate activity summary"


class TestGenerateActivityStatusForExecution:
    """Tests for the main generate_activity_status_for_execution function."""

    @pytest.mark.asyncio
    async def test_generate_status_success(
        self, mock_node_executions, mock_execution_stats, mock_blocks
    ):
        """Test successful activity status generation."""
        mock_db_client = AsyncMock()
        mock_db_client.get_node_executions.return_value = mock_node_executions

        mock_graph_metadata = MagicMock()
        mock_graph_metadata.name = "Test Agent"
        mock_graph_metadata.description = "A test agent"
        mock_db_client.get_graph_metadata.return_value = mock_graph_metadata

        mock_graph = MagicMock()
        mock_graph.links = []
        mock_db_client.get_graph.return_value = mock_graph

        with patch(
            "backend.executor.activity_status_generator.get_block"
        ) as mock_get_block, patch(
            "backend.executor.activity_status_generator.Settings"
        ) as mock_settings, patch(
            "backend.executor.activity_status_generator._call_llm_direct"
        ) as mock_llm, patch(
            "backend.executor.activity_status_generator.is_feature_enabled",
            return_value=True,
        ):

            mock_get_block.side_effect = lambda block_id: mock_blocks.get(block_id)
            mock_settings.return_value.secrets.openai_internal_api_key = "test_key"
            mock_llm.return_value = (
                "I analyzed your data and provided the requested insights."
            )

            result = await generate_activity_status_for_execution(
                graph_exec_id="test_exec",
                graph_id="test_graph",
                graph_version=1,
                execution_stats=mock_execution_stats,
                db_client=mock_db_client,
                user_id="test_user",
            )

            assert result == "I analyzed your data and provided the requested insights."
            mock_db_client.get_node_executions.assert_called_once()
            mock_db_client.get_graph_metadata.assert_called_once()
            mock_db_client.get_graph.assert_called_once()
            mock_llm.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_status_feature_disabled(self, mock_execution_stats):
        """Test activity status generation when feature is disabled."""
        mock_db_client = AsyncMock()

        with patch(
            "backend.executor.activity_status_generator.is_feature_enabled",
            return_value=False,
        ):
            result = await generate_activity_status_for_execution(
                graph_exec_id="test_exec",
                graph_id="test_graph",
                graph_version=1,
                execution_stats=mock_execution_stats,
                db_client=mock_db_client,
                user_id="test_user",
            )

            assert result is None
            mock_db_client.get_node_executions.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_status_no_api_key(self, mock_execution_stats):
        """Test activity status generation with no API key."""
        mock_db_client = AsyncMock()

        with patch(
            "backend.executor.activity_status_generator.Settings"
        ) as mock_settings, patch(
            "backend.executor.activity_status_generator.is_feature_enabled",
            return_value=True,
        ):
            mock_settings.return_value.secrets.openai_internal_api_key = ""

            result = await generate_activity_status_for_execution(
                graph_exec_id="test_exec",
                graph_id="test_graph",
                graph_version=1,
                execution_stats=mock_execution_stats,
                db_client=mock_db_client,
                user_id="test_user",
            )

            assert result is None
            mock_db_client.get_node_executions.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_status_exception_handling(self, mock_execution_stats):
        """Test activity status generation with exception."""
        mock_db_client = AsyncMock()
        mock_db_client.get_node_executions.side_effect = Exception("Database error")

        with patch(
            "backend.executor.activity_status_generator.Settings"
        ) as mock_settings, patch(
            "backend.executor.activity_status_generator.is_feature_enabled",
            return_value=True,
        ):
            mock_settings.return_value.secrets.openai_internal_api_key = "test_key"

            result = await generate_activity_status_for_execution(
                graph_exec_id="test_exec",
                graph_id="test_graph",
                graph_version=1,
                execution_stats=mock_execution_stats,
                db_client=mock_db_client,
                user_id="test_user",
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_generate_status_with_graph_name_fallback(
        self, mock_node_executions, mock_execution_stats, mock_blocks
    ):
        """Test activity status generation with graph name fallback."""
        mock_db_client = AsyncMock()
        mock_db_client.get_node_executions.return_value = mock_node_executions
        mock_db_client.get_graph_metadata.return_value = None  # No metadata
        mock_db_client.get_graph.return_value = None  # No graph

        with patch(
            "backend.executor.activity_status_generator.get_block"
        ) as mock_get_block, patch(
            "backend.executor.activity_status_generator.Settings"
        ) as mock_settings, patch(
            "backend.executor.activity_status_generator._call_llm_direct"
        ) as mock_llm, patch(
            "backend.executor.activity_status_generator.is_feature_enabled",
            return_value=True,
        ):

            mock_get_block.side_effect = lambda block_id: mock_blocks.get(block_id)
            mock_settings.return_value.secrets.openai_internal_api_key = "test_key"
            mock_llm.return_value = "Agent completed execution."

            result = await generate_activity_status_for_execution(
                graph_exec_id="test_exec",
                graph_id="test_graph",
                graph_version=1,
                execution_stats=mock_execution_stats,
                db_client=mock_db_client,
                user_id="test_user",
            )

            assert result == "Agent completed execution."
            # Should use fallback graph name in prompt
            call_args = mock_llm.call_args[0][1]  # prompt argument
            assert "Graph test_graph" in call_args[1]["content"]


class TestIntegration:
    """Integration tests to verify the complete flow."""

    @pytest.mark.asyncio
    async def test_full_integration_flow(
        self, mock_node_executions, mock_execution_stats, mock_blocks
    ):
        """Test the complete integration flow."""
        mock_db_client = AsyncMock()
        mock_db_client.get_node_executions.return_value = mock_node_executions

        mock_graph_metadata = MagicMock()
        mock_graph_metadata.name = "Test Integration Agent"
        mock_graph_metadata.description = "Integration test agent"
        mock_db_client.get_graph_metadata.return_value = mock_graph_metadata

        mock_graph = MagicMock()
        mock_graph.links = []
        mock_db_client.get_graph.return_value = mock_graph

        expected_activity = "I processed user input but failed during final output generation due to system error."

        with patch(
            "backend.executor.activity_status_generator.get_block"
        ) as mock_get_block, patch(
            "backend.executor.activity_status_generator.Settings"
        ) as mock_settings, patch(
            "backend.executor.activity_status_generator.llm_call"
        ) as mock_llm_call, patch(
            "backend.executor.activity_status_generator.is_feature_enabled",
            return_value=True,
        ):

            mock_get_block.side_effect = lambda block_id: mock_blocks.get(block_id)
            mock_settings.return_value.secrets.openai_internal_api_key = "test_key"

            mock_response = LLMResponse(
                raw_response={},
                prompt=[],
                response=expected_activity,
                tool_calls=None,
                prompt_tokens=100,
                completion_tokens=30,
            )
            mock_llm_call.return_value = mock_response

            result = await generate_activity_status_for_execution(
                graph_exec_id="test_exec",
                graph_id="test_graph",
                graph_version=1,
                execution_stats=mock_execution_stats,
                db_client=mock_db_client,
                user_id="test_user",
            )

            assert result == expected_activity

            # Verify the correct data was passed to LLM
            llm_call_args = mock_llm_call.call_args
            prompt = llm_call_args[1]["prompt"]

            # Check system prompt
            assert prompt[0]["role"] == "system"
            assert "user's perspective" in prompt[0]["content"]

            # Check user prompt contains expected data
            user_content = prompt[1]["content"]
            assert "Test Integration Agent" in user_content
            assert "user-friendly terms" in user_content.lower()

            # Verify that execution data is present in the prompt
            assert "{" in user_content  # Should contain JSON data
            assert "overall_status" in user_content

    @pytest.mark.asyncio
    async def test_manager_integration_with_disabled_feature(
        self, mock_execution_stats
    ):
        """Test that when feature returns None, manager doesn't set activity_status."""
        mock_db_client = AsyncMock()

        with patch(
            "backend.executor.activity_status_generator.is_feature_enabled",
            return_value=False,
        ):
            result = await generate_activity_status_for_execution(
                graph_exec_id="test_exec",
                graph_id="test_graph",
                graph_version=1,
                execution_stats=mock_execution_stats,
                db_client=mock_db_client,
                user_id="test_user",
            )

            # Should return None when disabled
            assert result is None

            # Verify no database calls were made
            mock_db_client.get_node_executions.assert_not_called()
            mock_db_client.get_graph_metadata.assert_not_called()
            mock_db_client.get_graph.assert_not_called()
