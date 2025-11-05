"""
Tests for activity status generator functionality.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks.llm import LlmModel, LLMResponse
from backend.data.execution import ExecutionStatus, NodeExecutionResult
from backend.data.model import GraphExecutionStats
from backend.executor.activity_status_generator import (
    _build_execution_summary,
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
    async def test_structured_llm_call_success(self):
        """Test successful structured LLM call."""
        from pydantic import SecretStr

        from backend.blocks.llm import AIStructuredResponseGeneratorBlock
        from backend.data.model import APIKeyCredentials

        with patch("backend.blocks.llm.llm_call") as mock_llm_call, patch(
            "backend.blocks.llm.secrets.token_hex", return_value="test123"
        ):
            mock_llm_call.return_value = LLMResponse(
                raw_response={},
                prompt=[],
                response='<json_output id="test123">{"activity_status": "Test completed successfully", "correctness_score": 0.9}</json_output>',
                tool_calls=None,
                prompt_tokens=50,
                completion_tokens=20,
            )

            credentials = APIKeyCredentials(
                id="test",
                provider="openai",
                api_key=SecretStr("test_key"),
                title="Test",
            )

            prompt = [{"role": "user", "content": "Test prompt"}]
            expected_format = {
                "activity_status": "User-friendly summary",
                "correctness_score": "Float score from 0.0 to 1.0",
            }

            # Create structured block and input
            structured_block = AIStructuredResponseGeneratorBlock()
            credentials_input = {
                "provider": credentials.provider,
                "id": credentials.id,
                "type": credentials.type,
                "title": credentials.title,
            }

            structured_input = AIStructuredResponseGeneratorBlock.Input(
                prompt=prompt[0]["content"],
                expected_format=expected_format,
                model=LlmModel.GPT4O_MINI,
                credentials=credentials_input,  # type: ignore
            )

            # Execute the structured LLM call
            result = None
            async for output_name, output_data in structured_block.run(
                structured_input, credentials=credentials
            ):
                if output_name == "response":
                    result = output_data
                    break

            assert result is not None
            assert result["activity_status"] == "Test completed successfully"
            assert result["correctness_score"] == 0.9
            mock_llm_call.assert_called()

    @pytest.mark.asyncio
    async def test_structured_llm_call_validation_error(self):
        """Test structured LLM call with validation error."""
        from pydantic import SecretStr

        from backend.blocks.llm import AIStructuredResponseGeneratorBlock
        from backend.data.model import APIKeyCredentials

        with patch("backend.blocks.llm.llm_call") as mock_llm_call, patch(
            "backend.blocks.llm.secrets.token_hex", return_value="test123"
        ):
            # Return invalid JSON that will fail validation (missing required field)
            mock_llm_call.return_value = LLMResponse(
                raw_response={},
                prompt=[],
                response='<json_output id="test123">{"activity_status": "Test completed successfully"}</json_output>',
                tool_calls=None,
                prompt_tokens=50,
                completion_tokens=20,
            )

            credentials = APIKeyCredentials(
                id="test",
                provider="openai",
                api_key=SecretStr("test_key"),
                title="Test",
            )

            prompt = [{"role": "user", "content": "Test prompt"}]
            expected_format = {
                "activity_status": "User-friendly summary",
                "correctness_score": "Float score from 0.0 to 1.0",
            }

            # Create structured block and input
            structured_block = AIStructuredResponseGeneratorBlock()
            credentials_input = {
                "provider": credentials.provider,
                "id": credentials.id,
                "type": credentials.type,
                "title": credentials.title,
            }

            structured_input = AIStructuredResponseGeneratorBlock.Input(
                prompt=prompt[0]["content"],
                expected_format=expected_format,
                model=LlmModel.GPT4O_MINI,
                credentials=credentials_input,  # type: ignore
                retry=1,  # Use fewer retries for faster test
            )

            with pytest.raises(
                Exception
            ):  # AIStructuredResponseGeneratorBlock may raise different exceptions
                async for output_name, output_data in structured_block.run(
                    structured_input, credentials=credentials
                ):
                    if output_name == "response":
                        break


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
            "backend.executor.activity_status_generator.AIStructuredResponseGeneratorBlock"
        ) as mock_structured_block, patch(
            "backend.executor.activity_status_generator.is_feature_enabled",
            return_value=True,
        ):

            mock_get_block.side_effect = lambda block_id: mock_blocks.get(block_id)
            mock_settings.return_value.secrets.openai_internal_api_key = "test_key"

            # Mock the structured block to return our expected response
            mock_instance = mock_structured_block.return_value

            async def mock_run(*args, **kwargs):
                yield "response", {
                    "activity_status": "I analyzed your data and provided the requested insights.",
                    "correctness_score": 0.85,
                }

            mock_instance.run = mock_run

            result = await generate_activity_status_for_execution(
                graph_exec_id="test_exec",
                graph_id="test_graph",
                graph_version=1,
                execution_stats=mock_execution_stats,
                db_client=mock_db_client,
                user_id="test_user",
            )

            assert result is not None
            assert (
                result["activity_status"]
                == "I analyzed your data and provided the requested insights."
            )
            assert result["correctness_score"] == 0.85
            mock_db_client.get_node_executions.assert_called_once()
            mock_db_client.get_graph_metadata.assert_called_once()
            mock_db_client.get_graph.assert_called_once()
            mock_structured_block.assert_called_once()

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
            "backend.executor.activity_status_generator.AIStructuredResponseGeneratorBlock"
        ) as mock_structured_block, patch(
            "backend.executor.activity_status_generator.is_feature_enabled",
            return_value=True,
        ):

            mock_get_block.side_effect = lambda block_id: mock_blocks.get(block_id)
            mock_settings.return_value.secrets.openai_internal_api_key = "test_key"

            # Mock the structured block to return our expected response
            mock_instance = mock_structured_block.return_value

            async def mock_run(*args, **kwargs):
                yield "response", {
                    "activity_status": "Agent completed execution.",
                    "correctness_score": 0.8,
                }

            mock_instance.run = mock_run

            result = await generate_activity_status_for_execution(
                graph_exec_id="test_exec",
                graph_id="test_graph",
                graph_version=1,
                execution_stats=mock_execution_stats,
                db_client=mock_db_client,
                user_id="test_user",
            )

            assert result is not None
            assert result["activity_status"] == "Agent completed execution."
            assert result["correctness_score"] == 0.8
            # The structured block should have been instantiated
            assert mock_structured_block.called


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
            "backend.executor.activity_status_generator.AIStructuredResponseGeneratorBlock"
        ) as mock_structured_block, patch(
            "backend.executor.activity_status_generator.is_feature_enabled",
            return_value=True,
        ):

            mock_get_block.side_effect = lambda block_id: mock_blocks.get(block_id)
            mock_settings.return_value.secrets.openai_internal_api_key = "test_key"

            # Mock the structured block to return our expected response
            mock_instance = mock_structured_block.return_value

            async def mock_run(*args, **kwargs):
                yield "response", {
                    "activity_status": expected_activity,
                    "correctness_score": 0.3,  # Low score since there was a failure
                }

            mock_instance.run = mock_run

            result = await generate_activity_status_for_execution(
                graph_exec_id="test_exec",
                graph_id="test_graph",
                graph_version=1,
                execution_stats=mock_execution_stats,
                db_client=mock_db_client,
                user_id="test_user",
            )

            assert result is not None
            assert result["activity_status"] == expected_activity
            assert result["correctness_score"] == 0.3

            # Verify the structured block was called
            assert mock_structured_block.called
            # The structured block should have been instantiated
            mock_structured_block.assert_called_once()

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
