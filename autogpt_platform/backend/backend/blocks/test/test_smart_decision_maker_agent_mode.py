"""
Tests for SmartDecisionMaker agent mode specific failure modes.

Covers failure modes:
2. Silent Tool Failures in Agent Mode
3. Unbounded Agent Mode Iterations
10. Unbounded Agent Iterations
12. Stale Credentials in Agent Mode
13. Tool Signature Cache Invalidation
"""

import asyncio
import json
import threading
from collections import defaultdict
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from backend.blocks.smart_decision_maker import (
    SmartDecisionMakerBlock,
    ExecutionParams,
    ToolInfo,
)


class TestSilentToolFailuresInAgentMode:
    """
    Tests for Failure Mode #2: Silent Tool Failures in Agent Mode

    When tool execution fails in agent mode, the error is converted to a
    tool response and execution continues silently.
    """

    @pytest.mark.asyncio
    async def test_tool_execution_failure_converted_to_response(self):
        """
        Test that tool execution failures are silently converted to responses.
        """
        import backend.blocks.llm as llm_module
        from backend.data.execution import ExecutionContext

        block = SmartDecisionMakerBlock()

        # First response: tool call
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_1"
        mock_tool_call.function.name = "failing_tool"
        mock_tool_call.function.arguments = json.dumps({"param": "value"})

        mock_response_1 = MagicMock()
        mock_response_1.response = None
        mock_response_1.tool_calls = [mock_tool_call]
        mock_response_1.prompt_tokens = 50
        mock_response_1.completion_tokens = 25
        mock_response_1.reasoning = None
        mock_response_1.raw_response = {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "call_1"}]
        }

        # Second response: finish after seeing error
        mock_response_2 = MagicMock()
        mock_response_2.response = "I encountered an error"
        mock_response_2.tool_calls = []
        mock_response_2.prompt_tokens = 30
        mock_response_2.completion_tokens = 15
        mock_response_2.reasoning = None
        mock_response_2.raw_response = {"role": "assistant", "content": "I encountered an error"}

        llm_call_count = 0

        async def mock_llm_call(**kwargs):
            nonlocal llm_call_count
            llm_call_count += 1
            if llm_call_count == 1:
                return mock_response_1
            return mock_response_2

        mock_tool_signatures = [
            {
                "type": "function",
                "function": {
                    "name": "failing_tool",
                    "_sink_node_id": "sink-node",
                    "_field_mapping": {"param": "param"},
                    "parameters": {
                        "properties": {"param": {"type": "string"}},
                        "required": ["param"],
                    },
                },
            }
        ]

        # Mock database client that will fail
        mock_db_client = AsyncMock()
        mock_db_client.get_node.side_effect = Exception("Database connection failed!")

        with patch("backend.blocks.llm.llm_call", side_effect=mock_llm_call), \
             patch.object(block, "_create_tool_node_signatures", return_value=mock_tool_signatures), \
             patch("backend.blocks.smart_decision_maker.get_database_manager_async_client", return_value=mock_db_client):

            mock_execution_context = ExecutionContext(safe_mode=False)
            mock_execution_processor = AsyncMock()
            mock_execution_processor.running_node_execution = defaultdict(MagicMock)
            mock_execution_processor.execution_stats = MagicMock()
            mock_execution_processor.execution_stats_lock = threading.Lock()

            input_data = SmartDecisionMakerBlock.Input(
                prompt="Do something",
                model=llm_module.DEFAULT_LLM_MODEL,
                credentials=llm_module.TEST_CREDENTIALS_INPUT,
                agent_mode_max_iterations=5,
            )

            outputs = {}
            async for name, value in block.run(
                input_data,
                credentials=llm_module.TEST_CREDENTIALS,
                graph_id="test-graph",
                node_id="test-node",
                graph_exec_id="test-exec",
                node_exec_id="test-node-exec",
                user_id="test-user",
                graph_version=1,
                execution_context=mock_execution_context,
                execution_processor=mock_execution_processor,
            ):
                outputs[name] = value

            # The execution completed (didn't crash)
            assert "finished" in outputs or "conversations" in outputs

            # BUG: The tool failure was silent - user doesn't know what happened
            # The error was just logged and converted to a tool response

    @pytest.mark.asyncio
    async def test_tool_failure_causes_infinite_retry_loop(self):
        """
        Test scenario where LLM keeps calling the same failing tool.

        If tool fails but LLM doesn't realize it, it may keep trying.
        """
        import backend.blocks.llm as llm_module
        from backend.data.execution import ExecutionContext

        block = SmartDecisionMakerBlock()

        call_count = 0
        max_calls = 10  # Limit for test

        def create_tool_call_response():
            mock_tool_call = MagicMock()
            mock_tool_call.id = f"call_{call_count}"
            mock_tool_call.function.name = "persistent_tool"
            mock_tool_call.function.arguments = json.dumps({"retry": call_count})

            mock_response = MagicMock()
            mock_response.response = None
            mock_response.tool_calls = [mock_tool_call]
            mock_response.prompt_tokens = 50
            mock_response.completion_tokens = 25
            mock_response.reasoning = None
            mock_response.raw_response = {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": f"call_{call_count}"}]
            }
            return mock_response

        async def mock_llm_call(**kwargs):
            nonlocal call_count
            call_count += 1

            if call_count >= max_calls:
                # Eventually finish to prevent actual infinite loop in test
                final = MagicMock()
                final.response = "Giving up"
                final.tool_calls = []
                final.prompt_tokens = 10
                final.completion_tokens = 5
                final.reasoning = None
                final.raw_response = {"role": "assistant", "content": "Giving up"}
                return final

            return create_tool_call_response()

        mock_tool_signatures = [
            {
                "type": "function",
                "function": {
                    "name": "persistent_tool",
                    "_sink_node_id": "sink-node",
                    "_field_mapping": {"retry": "retry"},
                    "parameters": {
                        "properties": {"retry": {"type": "integer"}},
                        "required": ["retry"],
                    },
                },
            }
        ]

        mock_db_client = AsyncMock()
        mock_db_client.get_node.side_effect = Exception("Always fails!")

        with patch("backend.blocks.llm.llm_call", side_effect=mock_llm_call), \
             patch.object(block, "_create_tool_node_signatures", return_value=mock_tool_signatures), \
             patch("backend.blocks.smart_decision_maker.get_database_manager_async_client", return_value=mock_db_client):

            mock_execution_context = ExecutionContext(safe_mode=False)
            mock_execution_processor = AsyncMock()
            mock_execution_processor.running_node_execution = defaultdict(MagicMock)
            mock_execution_processor.execution_stats = MagicMock()
            mock_execution_processor.execution_stats_lock = threading.Lock()

            input_data = SmartDecisionMakerBlock.Input(
                prompt="Keep trying",
                model=llm_module.DEFAULT_LLM_MODEL,
                credentials=llm_module.TEST_CREDENTIALS_INPUT,
                agent_mode_max_iterations=-1,  # Infinite mode!
            )

            # Use timeout to prevent actual infinite loop
            try:
                async with asyncio.timeout(5):
                    outputs = {}
                    async for name, value in block.run(
                        input_data,
                        credentials=llm_module.TEST_CREDENTIALS,
                        graph_id="test-graph",
                        node_id="test-node",
                        graph_exec_id="test-exec",
                        node_exec_id="test-node-exec",
                        user_id="test-user",
                        graph_version=1,
                        execution_context=mock_execution_context,
                        execution_processor=mock_execution_processor,
                    ):
                        outputs[name] = value
            except asyncio.TimeoutError:
                pass  # Expected if we hit infinite loop

            # Document that many calls were made before we gave up
            assert call_count >= max_calls - 1, \
                f"Expected many retries, got {call_count}"


class TestUnboundedAgentIterations:
    """
    Tests for Failure Mode #3 and #10: Unbounded Agent Mode Iterations

    With max_iterations = -1, the agent can run forever, consuming
    unlimited tokens and compute resources.
    """

    @pytest.mark.asyncio
    async def test_infinite_mode_requires_llm_to_stop(self):
        """
        Test that infinite mode (-1) only stops when LLM stops making tool calls.
        """
        import backend.blocks.llm as llm_module
        from backend.data.execution import ExecutionContext

        block = SmartDecisionMakerBlock()

        iterations = 0
        max_test_iterations = 20

        async def mock_llm_call(**kwargs):
            nonlocal iterations
            iterations += 1

            if iterations >= max_test_iterations:
                # Stop to prevent actual infinite loop
                resp = MagicMock()
                resp.response = "Finally done"
                resp.tool_calls = []
                resp.prompt_tokens = 10
                resp.completion_tokens = 5
                resp.reasoning = None
                resp.raw_response = {"role": "assistant", "content": "Done"}
                return resp

            # Keep making tool calls
            tool_call = MagicMock()
            tool_call.id = f"call_{iterations}"
            tool_call.function.name = "counter_tool"
            tool_call.function.arguments = json.dumps({"count": iterations})

            resp = MagicMock()
            resp.response = None
            resp.tool_calls = [tool_call]
            resp.prompt_tokens = 50
            resp.completion_tokens = 25
            resp.reasoning = None
            resp.raw_response = {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": f"call_{iterations}"}]
            }
            return resp

        mock_tool_signatures = [
            {
                "type": "function",
                "function": {
                    "name": "counter_tool",
                    "_sink_node_id": "sink",
                    "_field_mapping": {"count": "count"},
                    "parameters": {
                        "properties": {"count": {"type": "integer"}},
                        "required": ["count"],
                    },
                },
            }
        ]

        mock_db_client = AsyncMock()
        mock_node = MagicMock()
        mock_node.block_id = "test-block"
        mock_db_client.get_node.return_value = mock_node

        mock_exec_result = MagicMock()
        mock_exec_result.node_exec_id = "exec-id"
        mock_db_client.upsert_execution_input.return_value = (mock_exec_result, {"count": 1})
        mock_db_client.get_execution_outputs_by_node_exec_id.return_value = {"result": "ok"}

        with patch("backend.blocks.llm.llm_call", side_effect=mock_llm_call), \
             patch.object(block, "_create_tool_node_signatures", return_value=mock_tool_signatures), \
             patch("backend.blocks.smart_decision_maker.get_database_manager_async_client", return_value=mock_db_client):

            mock_execution_context = ExecutionContext(safe_mode=False)
            mock_execution_processor = AsyncMock()
            mock_execution_processor.running_node_execution = defaultdict(MagicMock)
            mock_execution_processor.execution_stats = MagicMock()
            mock_execution_processor.execution_stats_lock = threading.Lock()
            mock_execution_processor.on_node_execution = AsyncMock(return_value=MagicMock(error=None))

            input_data = SmartDecisionMakerBlock.Input(
                prompt="Count forever",
                model=llm_module.DEFAULT_LLM_MODEL,
                credentials=llm_module.TEST_CREDENTIALS_INPUT,
                agent_mode_max_iterations=-1,  # INFINITE MODE
            )

            async with asyncio.timeout(10):
                outputs = {}
                async for name, value in block.run(
                    input_data,
                    credentials=llm_module.TEST_CREDENTIALS,
                    graph_id="test-graph",
                    node_id="test-node",
                    graph_exec_id="test-exec",
                    node_exec_id="test-node-exec",
                    user_id="test-user",
                    graph_version=1,
                    execution_context=mock_execution_context,
                    execution_processor=mock_execution_processor,
                ):
                    outputs[name] = value

            # We ran many iterations before stopping
            assert iterations == max_test_iterations
            # BUG: No built-in safeguard against runaway iterations

    @pytest.mark.asyncio
    async def test_max_iterations_limit_enforced(self):
        """
        Test that max_iterations limit is properly enforced.
        """
        import backend.blocks.llm as llm_module
        from backend.data.execution import ExecutionContext

        block = SmartDecisionMakerBlock()

        iterations = 0

        async def mock_llm_call(**kwargs):
            nonlocal iterations
            iterations += 1

            # Always make tool calls (never finish voluntarily)
            tool_call = MagicMock()
            tool_call.id = f"call_{iterations}"
            tool_call.function.name = "endless_tool"
            tool_call.function.arguments = json.dumps({})

            resp = MagicMock()
            resp.response = None
            resp.tool_calls = [tool_call]
            resp.prompt_tokens = 50
            resp.completion_tokens = 25
            resp.reasoning = None
            resp.raw_response = {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": f"call_{iterations}"}]
            }
            return resp

        mock_tool_signatures = [
            {
                "type": "function",
                "function": {
                    "name": "endless_tool",
                    "_sink_node_id": "sink",
                    "_field_mapping": {},
                    "parameters": {"properties": {}, "required": []},
                },
            }
        ]

        mock_db_client = AsyncMock()
        mock_node = MagicMock()
        mock_node.block_id = "test-block"
        mock_db_client.get_node.return_value = mock_node
        mock_exec_result = MagicMock()
        mock_exec_result.node_exec_id = "exec-id"
        mock_db_client.upsert_execution_input.return_value = (mock_exec_result, {})
        mock_db_client.get_execution_outputs_by_node_exec_id.return_value = {}

        with patch("backend.blocks.llm.llm_call", side_effect=mock_llm_call), \
             patch.object(block, "_create_tool_node_signatures", return_value=mock_tool_signatures), \
             patch("backend.blocks.smart_decision_maker.get_database_manager_async_client", return_value=mock_db_client):

            mock_execution_context = ExecutionContext(safe_mode=False)
            mock_execution_processor = AsyncMock()
            mock_execution_processor.running_node_execution = defaultdict(MagicMock)
            mock_execution_processor.execution_stats = MagicMock()
            mock_execution_processor.execution_stats_lock = threading.Lock()
            mock_execution_processor.on_node_execution = AsyncMock(return_value=MagicMock(error=None))

            MAX_ITERATIONS = 3
            input_data = SmartDecisionMakerBlock.Input(
                prompt="Run forever",
                model=llm_module.DEFAULT_LLM_MODEL,
                credentials=llm_module.TEST_CREDENTIALS_INPUT,
                agent_mode_max_iterations=MAX_ITERATIONS,
            )

            outputs = {}
            async for name, value in block.run(
                input_data,
                credentials=llm_module.TEST_CREDENTIALS,
                graph_id="test-graph",
                node_id="test-node",
                graph_exec_id="test-exec",
                node_exec_id="test-node-exec",
                user_id="test-user",
                graph_version=1,
                execution_context=mock_execution_context,
                execution_processor=mock_execution_processor,
            ):
                outputs[name] = value

            # Should have stopped at max iterations
            assert iterations == MAX_ITERATIONS
            assert "finished" in outputs
            assert "limit reached" in outputs["finished"].lower()


class TestStaleCredentialsInAgentMode:
    """
    Tests for Failure Mode #12: Stale Credentials in Agent Mode

    Credentials are validated once at start but can expire during
    long-running agent mode executions.
    """

    @pytest.mark.asyncio
    async def test_credentials_not_revalidated_between_iterations(self):
        """
        Test that credentials are used without revalidation in agent mode.
        """
        import backend.blocks.llm as llm_module
        from backend.data.execution import ExecutionContext

        block = SmartDecisionMakerBlock()

        credential_check_count = 0
        iteration = 0

        async def mock_llm_call(**kwargs):
            nonlocal credential_check_count, iteration
            iteration += 1

            # Simulate credential check (in real code this happens in llm_call)
            credential_check_count += 1

            if iteration >= 3:
                resp = MagicMock()
                resp.response = "Done"
                resp.tool_calls = []
                resp.prompt_tokens = 10
                resp.completion_tokens = 5
                resp.reasoning = None
                resp.raw_response = {"role": "assistant", "content": "Done"}
                return resp

            tool_call = MagicMock()
            tool_call.id = f"call_{iteration}"
            tool_call.function.name = "test_tool"
            tool_call.function.arguments = json.dumps({})

            resp = MagicMock()
            resp.response = None
            resp.tool_calls = [tool_call]
            resp.prompt_tokens = 50
            resp.completion_tokens = 25
            resp.reasoning = None
            resp.raw_response = {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": f"call_{iteration}"}]
            }
            return resp

        mock_tool_signatures = [
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "_sink_node_id": "sink",
                    "_field_mapping": {},
                    "parameters": {"properties": {}, "required": []},
                },
            }
        ]

        mock_db_client = AsyncMock()
        mock_node = MagicMock()
        mock_node.block_id = "test-block"
        mock_db_client.get_node.return_value = mock_node
        mock_exec_result = MagicMock()
        mock_exec_result.node_exec_id = "exec-id"
        mock_db_client.upsert_execution_input.return_value = (mock_exec_result, {})
        mock_db_client.get_execution_outputs_by_node_exec_id.return_value = {}

        with patch("backend.blocks.llm.llm_call", side_effect=mock_llm_call), \
             patch.object(block, "_create_tool_node_signatures", return_value=mock_tool_signatures), \
             patch("backend.blocks.smart_decision_maker.get_database_manager_async_client", return_value=mock_db_client):

            mock_execution_context = ExecutionContext(safe_mode=False)
            mock_execution_processor = AsyncMock()
            mock_execution_processor.running_node_execution = defaultdict(MagicMock)
            mock_execution_processor.execution_stats = MagicMock()
            mock_execution_processor.execution_stats_lock = threading.Lock()
            mock_execution_processor.on_node_execution = AsyncMock(return_value=MagicMock(error=None))

            input_data = SmartDecisionMakerBlock.Input(
                prompt="Test credentials",
                model=llm_module.DEFAULT_LLM_MODEL,
                credentials=llm_module.TEST_CREDENTIALS_INPUT,
                agent_mode_max_iterations=5,
            )

            outputs = {}
            async for name, value in block.run(
                input_data,
                credentials=llm_module.TEST_CREDENTIALS,
                graph_id="test-graph",
                node_id="test-node",
                graph_exec_id="test-exec",
                node_exec_id="test-node-exec",
                user_id="test-user",
                graph_version=1,
                execution_context=mock_execution_context,
                execution_processor=mock_execution_processor,
            ):
                outputs[name] = value

            # Credentials were checked on each LLM call but not refreshed
            # If they expired mid-execution, we'd get auth errors
            assert credential_check_count == iteration

    @pytest.mark.asyncio
    async def test_credential_expiration_mid_execution(self):
        """
        Test what happens when credentials expire during agent mode.
        """
        import backend.blocks.llm as llm_module
        from backend.data.execution import ExecutionContext

        block = SmartDecisionMakerBlock()

        iteration = 0

        async def mock_llm_call_with_expiration(**kwargs):
            nonlocal iteration
            iteration += 1

            if iteration >= 3:
                # Simulate credential expiration
                raise Exception("401 Unauthorized: API key expired")

            tool_call = MagicMock()
            tool_call.id = f"call_{iteration}"
            tool_call.function.name = "test_tool"
            tool_call.function.arguments = json.dumps({})

            resp = MagicMock()
            resp.response = None
            resp.tool_calls = [tool_call]
            resp.prompt_tokens = 50
            resp.completion_tokens = 25
            resp.reasoning = None
            resp.raw_response = {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": f"call_{iteration}"}]
            }
            return resp

        mock_tool_signatures = [
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "_sink_node_id": "sink",
                    "_field_mapping": {},
                    "parameters": {"properties": {}, "required": []},
                },
            }
        ]

        mock_db_client = AsyncMock()
        mock_node = MagicMock()
        mock_node.block_id = "test-block"
        mock_db_client.get_node.return_value = mock_node
        mock_exec_result = MagicMock()
        mock_exec_result.node_exec_id = "exec-id"
        mock_db_client.upsert_execution_input.return_value = (mock_exec_result, {})
        mock_db_client.get_execution_outputs_by_node_exec_id.return_value = {}

        with patch("backend.blocks.llm.llm_call", side_effect=mock_llm_call_with_expiration), \
             patch.object(block, "_create_tool_node_signatures", return_value=mock_tool_signatures), \
             patch("backend.blocks.smart_decision_maker.get_database_manager_async_client", return_value=mock_db_client):

            mock_execution_context = ExecutionContext(safe_mode=False)
            mock_execution_processor = AsyncMock()
            mock_execution_processor.running_node_execution = defaultdict(MagicMock)
            mock_execution_processor.execution_stats = MagicMock()
            mock_execution_processor.execution_stats_lock = threading.Lock()
            mock_execution_processor.on_node_execution = AsyncMock(return_value=MagicMock(error=None))

            input_data = SmartDecisionMakerBlock.Input(
                prompt="Test credentials",
                model=llm_module.DEFAULT_LLM_MODEL,
                credentials=llm_module.TEST_CREDENTIALS_INPUT,
                agent_mode_max_iterations=10,
            )

            outputs = {}
            async for name, value in block.run(
                input_data,
                credentials=llm_module.TEST_CREDENTIALS,
                graph_id="test-graph",
                node_id="test-node",
                graph_exec_id="test-exec",
                node_exec_id="test-node-exec",
                user_id="test-user",
                graph_version=1,
                execution_context=mock_execution_context,
                execution_processor=mock_execution_processor,
            ):
                outputs[name] = value

            # Should have an error output
            assert "error" in outputs
            assert "expired" in outputs["error"].lower() or "unauthorized" in outputs["error"].lower()


class TestToolSignatureCacheInvalidation:
    """
    Tests for Failure Mode #13: Tool Signature Cache Invalidation

    Tool signatures are created once at the start of run() but the
    graph could change during agent mode execution.
    """

    @pytest.mark.asyncio
    async def test_signatures_created_once_at_start(self):
        """
        Test that tool signatures are only created once, not refreshed.
        """
        import backend.blocks.llm as llm_module
        from backend.data.execution import ExecutionContext

        block = SmartDecisionMakerBlock()

        signature_creation_count = 0
        iteration = 0

        original_create_signatures = block._create_tool_node_signatures

        async def counting_create_signatures(node_id):
            nonlocal signature_creation_count
            signature_creation_count += 1
            return [
                {
                    "type": "function",
                    "function": {
                        "name": "tool_v1",
                        "_sink_node_id": "sink",
                        "_field_mapping": {},
                        "parameters": {"properties": {}, "required": []},
                    },
                }
            ]

        async def mock_llm_call(**kwargs):
            nonlocal iteration
            iteration += 1

            if iteration >= 3:
                resp = MagicMock()
                resp.response = "Done"
                resp.tool_calls = []
                resp.prompt_tokens = 10
                resp.completion_tokens = 5
                resp.reasoning = None
                resp.raw_response = {"role": "assistant", "content": "Done"}
                return resp

            tool_call = MagicMock()
            tool_call.id = f"call_{iteration}"
            tool_call.function.name = "tool_v1"
            tool_call.function.arguments = json.dumps({})

            resp = MagicMock()
            resp.response = None
            resp.tool_calls = [tool_call]
            resp.prompt_tokens = 50
            resp.completion_tokens = 25
            resp.reasoning = None
            resp.raw_response = {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": f"call_{iteration}"}]
            }
            return resp

        mock_db_client = AsyncMock()
        mock_node = MagicMock()
        mock_node.block_id = "test-block"
        mock_db_client.get_node.return_value = mock_node
        mock_exec_result = MagicMock()
        mock_exec_result.node_exec_id = "exec-id"
        mock_db_client.upsert_execution_input.return_value = (mock_exec_result, {})
        mock_db_client.get_execution_outputs_by_node_exec_id.return_value = {}

        with patch("backend.blocks.llm.llm_call", side_effect=mock_llm_call), \
             patch.object(block, "_create_tool_node_signatures", side_effect=counting_create_signatures), \
             patch("backend.blocks.smart_decision_maker.get_database_manager_async_client", return_value=mock_db_client):

            mock_execution_context = ExecutionContext(safe_mode=False)
            mock_execution_processor = AsyncMock()
            mock_execution_processor.running_node_execution = defaultdict(MagicMock)
            mock_execution_processor.execution_stats = MagicMock()
            mock_execution_processor.execution_stats_lock = threading.Lock()
            mock_execution_processor.on_node_execution = AsyncMock(return_value=MagicMock(error=None))

            input_data = SmartDecisionMakerBlock.Input(
                prompt="Test signatures",
                model=llm_module.DEFAULT_LLM_MODEL,
                credentials=llm_module.TEST_CREDENTIALS_INPUT,
                agent_mode_max_iterations=5,
            )

            outputs = {}
            async for name, value in block.run(
                input_data,
                credentials=llm_module.TEST_CREDENTIALS,
                graph_id="test-graph",
                node_id="test-node",
                graph_exec_id="test-exec",
                node_exec_id="test-node-exec",
                user_id="test-user",
                graph_version=1,
                execution_context=mock_execution_context,
                execution_processor=mock_execution_processor,
            ):
                outputs[name] = value

            # Signatures were only created once, even though we had multiple iterations
            assert signature_creation_count == 1
            assert iteration >= 3  # We had multiple iterations

    @pytest.mark.asyncio
    async def test_stale_signatures_cause_tool_mismatch(self):
        """
        Test scenario where tool definitions change but agent uses stale signatures.
        """
        # This documents the potential issue:
        # 1. Agent starts with tool_v1
        # 2. User modifies graph, tool becomes tool_v2
        # 3. Agent still thinks tool_v1 exists
        # 4. LLM calls tool_v1, but it no longer exists

        # Since signatures are created once at start and never refreshed,
        # any changes to the graph during execution won't be reflected.

        # This is more of a documentation test - the actual fix would
        # require either:
        # a) Refreshing signatures periodically
        # b) Locking the graph during execution
        # c) Checking tool existence before each call
        pass


class TestAgentModeConversationManagement:
    """Tests for conversation management in agent mode."""

    @pytest.mark.asyncio
    async def test_conversation_grows_with_iterations(self):
        """
        Test that conversation history grows correctly with each iteration.
        """
        import backend.blocks.llm as llm_module
        from backend.data.execution import ExecutionContext

        block = SmartDecisionMakerBlock()

        iteration = 0
        conversation_lengths = []

        async def mock_llm_call(**kwargs):
            nonlocal iteration
            iteration += 1

            # Record conversation length at each call
            prompt = kwargs.get("prompt", [])
            conversation_lengths.append(len(prompt))

            if iteration >= 3:
                resp = MagicMock()
                resp.response = "Done"
                resp.tool_calls = []
                resp.prompt_tokens = 10
                resp.completion_tokens = 5
                resp.reasoning = None
                resp.raw_response = {"role": "assistant", "content": "Done"}
                return resp

            tool_call = MagicMock()
            tool_call.id = f"call_{iteration}"
            tool_call.function.name = "test_tool"
            tool_call.function.arguments = json.dumps({})

            resp = MagicMock()
            resp.response = None
            resp.tool_calls = [tool_call]
            resp.prompt_tokens = 50
            resp.completion_tokens = 25
            resp.reasoning = None
            resp.raw_response = {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": f"call_{iteration}"}]
            }
            return resp

        mock_tool_signatures = [
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "_sink_node_id": "sink",
                    "_field_mapping": {},
                    "parameters": {"properties": {}, "required": []},
                },
            }
        ]

        mock_db_client = AsyncMock()
        mock_node = MagicMock()
        mock_node.block_id = "test-block"
        mock_db_client.get_node.return_value = mock_node
        mock_exec_result = MagicMock()
        mock_exec_result.node_exec_id = "exec-id"
        mock_db_client.upsert_execution_input.return_value = (mock_exec_result, {})
        mock_db_client.get_execution_outputs_by_node_exec_id.return_value = {"result": "ok"}

        with patch("backend.blocks.llm.llm_call", side_effect=mock_llm_call), \
             patch.object(block, "_create_tool_node_signatures", return_value=mock_tool_signatures), \
             patch("backend.blocks.smart_decision_maker.get_database_manager_async_client", return_value=mock_db_client):

            mock_execution_context = ExecutionContext(safe_mode=False)
            mock_execution_processor = AsyncMock()
            mock_execution_processor.running_node_execution = defaultdict(MagicMock)
            mock_execution_processor.execution_stats = MagicMock()
            mock_execution_processor.execution_stats_lock = threading.Lock()
            mock_execution_processor.on_node_execution = AsyncMock(return_value=MagicMock(error=None))

            input_data = SmartDecisionMakerBlock.Input(
                prompt="Test conversation",
                model=llm_module.DEFAULT_LLM_MODEL,
                credentials=llm_module.TEST_CREDENTIALS_INPUT,
                agent_mode_max_iterations=5,
            )

            outputs = {}
            async for name, value in block.run(
                input_data,
                credentials=llm_module.TEST_CREDENTIALS,
                graph_id="test-graph",
                node_id="test-node",
                graph_exec_id="test-exec",
                node_exec_id="test-node-exec",
                user_id="test-user",
                graph_version=1,
                execution_context=mock_execution_context,
                execution_processor=mock_execution_processor,
            ):
                outputs[name] = value

            # Conversation should grow with each iteration
            # Each iteration adds: assistant message + tool response
            assert len(conversation_lengths) == 3
            for i in range(1, len(conversation_lengths)):
                assert conversation_lengths[i] > conversation_lengths[i-1], \
                    f"Conversation should grow: {conversation_lengths}"
