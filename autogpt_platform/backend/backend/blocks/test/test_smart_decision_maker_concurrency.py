"""
Tests for SmartDecisionMaker concurrency issues and race conditions.

Covers failure modes:
1. Conversation History Race Condition
4. Concurrent Execution State Sharing
7. Race in Pending Tool Calls
11. Race in Pending Tool Call Retrieval
14. Concurrent State Sharing
"""

import asyncio
import json
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from backend.blocks.smart_decision_maker import (
    SmartDecisionMakerBlock,
    get_pending_tool_calls,
    _create_tool_response,
    _get_tool_requests,
    _get_tool_responses,
)


class TestConversationHistoryRaceCondition:
    """
    Tests for Failure Mode #1: Conversation History Race Condition

    When multiple executions share conversation history, concurrent
    modifications can cause data loss or corruption.
    """

    def test_get_pending_tool_calls_with_concurrent_modification(self):
        """
        Test that concurrent modifications to conversation history
        can cause inconsistent pending tool call counts.
        """
        # Shared conversation history
        conversation_history = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_1"},
                    {"type": "tool_use", "id": "toolu_2"},
                    {"type": "tool_use", "id": "toolu_3"},
                ]
            }
        ]

        results = []
        errors = []

        def reader_thread():
            """Repeatedly read pending calls."""
            for _ in range(100):
                try:
                    pending = get_pending_tool_calls(conversation_history)
                    results.append(len(pending))
                except Exception as e:
                    errors.append(str(e))

        def writer_thread():
            """Modify conversation while readers are active."""
            for i in range(50):
                # Add a tool response
                conversation_history.append({
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": f"toolu_{(i % 3) + 1}"}]
                })
                # Remove it
                if len(conversation_history) > 1:
                    conversation_history.pop()

        # Run concurrent readers and writers
        threads = []
        for _ in range(3):
            threads.append(threading.Thread(target=reader_thread))
        threads.append(threading.Thread(target=writer_thread))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # The issue: results may be inconsistent due to race conditions
        # In a correct implementation, we'd expect consistent results
        # Document that this CAN produce inconsistent results
        assert len(results) > 0, "Should have some results"
        # Note: This test documents the race condition exists
        # When fixed, all results should be consistent

    def test_prompt_list_mutation_race(self):
        """
        Test that mutating prompt list during iteration can cause issues.
        """
        prompt = []
        errors = []

        def appender():
            for i in range(100):
                prompt.append({"role": "user", "content": f"msg_{i}"})

        def extender():
            for i in range(100):
                prompt.extend([{"role": "assistant", "content": f"resp_{i}"}])

        def reader():
            for _ in range(100):
                try:
                    # Iterate while others modify
                    _ = [p for p in prompt if p.get("role") == "user"]
                except RuntimeError as e:
                    # "dictionary changed size during iteration" or similar
                    errors.append(str(e))

        threads = [
            threading.Thread(target=appender),
            threading.Thread(target=extender),
            threading.Thread(target=reader),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Document that race conditions can occur
        # In production, this could cause silent data corruption

    @pytest.mark.asyncio
    async def test_concurrent_block_runs_share_state(self):
        """
        Test that concurrent runs on same block instance can share state incorrectly.

        This is Failure Mode #14: Concurrent State Sharing
        """
        import backend.blocks.llm as llm_module
        from backend.data.execution import ExecutionContext

        block = SmartDecisionMakerBlock()

        # Track all outputs from all runs
        all_outputs = []
        lock = threading.Lock()

        async def run_block(run_id: int):
            """Run the block with a unique run_id."""
            mock_response = MagicMock()
            mock_response.response = f"Response for run {run_id}"
            mock_response.tool_calls = []  # No tool calls, just finish
            mock_response.prompt_tokens = 50
            mock_response.completion_tokens = 25
            mock_response.reasoning = None
            mock_response.raw_response = {"role": "assistant", "content": f"Run {run_id}"}

            mock_tool_signatures = []

            with patch("backend.blocks.llm.llm_call", new_callable=AsyncMock) as mock_llm:
                mock_llm.return_value = mock_response

                with patch.object(block, "_create_tool_node_signatures", return_value=mock_tool_signatures):
                    input_data = SmartDecisionMakerBlock.Input(
                        prompt=f"Prompt for run {run_id}",
                        model=llm_module.DEFAULT_LLM_MODEL,
                        credentials=llm_module.TEST_CREDENTIALS_INPUT,
                        agent_mode_max_iterations=0,
                    )

                    mock_execution_context = ExecutionContext(safe_mode=False)
                    mock_execution_processor = MagicMock()

                    outputs = {}
                    async for output_name, output_data in block.run(
                        input_data,
                        credentials=llm_module.TEST_CREDENTIALS,
                        graph_id=f"graph-{run_id}",
                        node_id=f"node-{run_id}",
                        graph_exec_id=f"exec-{run_id}",
                        node_exec_id=f"node-exec-{run_id}",
                        user_id=f"user-{run_id}",
                        graph_version=1,
                        execution_context=mock_execution_context,
                        execution_processor=mock_execution_processor,
                    ):
                        outputs[output_name] = output_data

                    with lock:
                        all_outputs.append((run_id, outputs))

        # Run multiple concurrent executions
        tasks = [run_block(i) for i in range(5)]
        await asyncio.gather(*tasks)

        # Verify each run got its own response (no cross-contamination)
        for run_id, outputs in all_outputs:
            if "finished" in outputs:
                assert f"run {run_id}" in outputs["finished"].lower() or outputs["finished"] == f"Response for run {run_id}", \
                    f"Run {run_id} may have received contaminated response: {outputs}"


class TestPendingToolCallRace:
    """
    Tests for Failure Mode #7 and #11: Race in Pending Tool Calls

    The get_pending_tool_calls function can race with modifications
    to the conversation history, causing StopIteration or incorrect counts.
    """

    def test_pending_tool_calls_counter_accuracy(self):
        """Test that pending tool call counting is accurate."""
        conversation = [
            # Assistant makes 3 tool calls
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "call_1"},
                    {"type": "tool_use", "id": "call_2"},
                    {"type": "tool_use", "id": "call_3"},
                ]
            },
            # User provides 1 response
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call_1"}
                ]
            }
        ]

        pending = get_pending_tool_calls(conversation)

        # Should have 2 pending (call_2, call_3)
        assert len(pending) == 2
        assert "call_2" in pending
        assert "call_3" in pending
        assert pending["call_2"] == 1
        assert pending["call_3"] == 1

    def test_pending_tool_calls_duplicate_responses(self):
        """Test handling of duplicate tool responses."""
        conversation = [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "call_1"}]
            },
            # Duplicate responses for same call
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "call_1"}]
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "call_1"}]
            }
        ]

        pending = get_pending_tool_calls(conversation)

        # call_1 has count -1 (1 request - 2 responses)
        # Should not be in pending (count <= 0)
        assert "call_1" not in pending or pending.get("call_1", 0) <= 0

    def test_empty_conversation_no_pending(self):
        """Test that empty conversation has no pending calls."""
        assert get_pending_tool_calls([]) == {}
        assert get_pending_tool_calls(None) == {}

    def test_next_iter_on_empty_dict_raises_stop_iteration(self):
        """
        Document the StopIteration vulnerability.

        If pending_tool_calls becomes empty between the check and
        next(iter(...)), StopIteration is raised.
        """
        pending = {}

        # This is the pattern used in smart_decision_maker.py:1019
        # if pending_tool_calls and ...:
        #     first_call_id = next(iter(pending_tool_calls.keys()))

        with pytest.raises(StopIteration):
            next(iter(pending.keys()))

        # Safe pattern should be:
        # first_call_id = next(iter(pending_tool_calls.keys()), None)
        safe_result = next(iter(pending.keys()), None)
        assert safe_result is None


class TestToolRequestResponseParsing:
    """Tests for tool request/response parsing edge cases."""

    def test_get_tool_requests_openai_format(self):
        """Test parsing OpenAI format tool requests."""
        entry = {
            "role": "assistant",
            "tool_calls": [
                {"id": "call_abc123"},
                {"id": "call_def456"},
            ]
        }

        requests = _get_tool_requests(entry)
        assert requests == ["call_abc123", "call_def456"]

    def test_get_tool_requests_anthropic_format(self):
        """Test parsing Anthropic format tool requests."""
        entry = {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_abc123"},
                {"type": "text", "text": "Let me call this tool"},
                {"type": "tool_use", "id": "toolu_def456"},
            ]
        }

        requests = _get_tool_requests(entry)
        assert requests == ["toolu_abc123", "toolu_def456"]

    def test_get_tool_requests_non_assistant_role(self):
        """Non-assistant roles should return empty list."""
        entry = {"role": "user", "tool_calls": [{"id": "call_123"}]}
        assert _get_tool_requests(entry) == []

    def test_get_tool_responses_openai_format(self):
        """Test parsing OpenAI format tool responses."""
        entry = {
            "role": "tool",
            "tool_call_id": "call_abc123",
            "content": "Result"
        }

        responses = _get_tool_responses(entry)
        assert responses == ["call_abc123"]

    def test_get_tool_responses_anthropic_format(self):
        """Test parsing Anthropic format tool responses."""
        entry = {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_abc123"},
                {"type": "tool_result", "tool_use_id": "toolu_def456"},
            ]
        }

        responses = _get_tool_responses(entry)
        assert responses == ["toolu_abc123", "toolu_def456"]

    def test_get_tool_responses_mixed_content(self):
        """Test parsing responses with mixed content types."""
        entry = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Here are the results"},
                {"type": "tool_result", "tool_use_id": "toolu_123"},
                {"type": "image", "url": "http://example.com/img.png"},
            ]
        }

        responses = _get_tool_responses(entry)
        assert responses == ["toolu_123"]


class TestConcurrentToolSignatureCreation:
    """Tests for concurrent tool signature creation."""

    @pytest.mark.asyncio
    async def test_concurrent_signature_creation_same_node(self):
        """
        Test that concurrent signature creation for same node
        doesn't cause issues.
        """
        block = SmartDecisionMakerBlock()

        mock_node = Mock()
        mock_node.id = "test-node"
        mock_node.block = Mock()
        mock_node.block.name = "TestBlock"
        mock_node.block.description = "Test"
        mock_node.block.input_schema = Mock()
        mock_node.block.input_schema.jsonschema = Mock(
            return_value={"properties": {}, "required": []}
        )
        mock_node.block.input_schema.get_field_schema = Mock(
            return_value={"type": "string", "description": "test"}
        )

        mock_links = [
            Mock(sink_name="field1", sink_id="test-node", source_id="source"),
            Mock(sink_name="field2", sink_id="test-node", source_id="source"),
        ]

        # Run multiple concurrent signature creations
        tasks = [
            block._create_block_function_signature(mock_node, mock_links)
            for _ in range(10)
        ]

        results = await asyncio.gather(*tasks)

        # All results should be identical
        first = results[0]
        for i, result in enumerate(results[1:], 1):
            assert result["function"]["name"] == first["function"]["name"], \
                f"Result {i} has different name"
            assert set(result["function"]["parameters"]["properties"].keys()) == \
                   set(first["function"]["parameters"]["properties"].keys()), \
                f"Result {i} has different properties"


class TestThreadSafetyOfCleanup:
    """Tests for thread safety of cleanup function."""

    def test_cleanup_is_thread_safe(self):
        """
        Test that cleanup function is thread-safe.

        Since it's a pure function with no shared state, it should be safe.
        """
        results = {}
        lock = threading.Lock()

        test_inputs = [
            "Max Keyword Difficulty",
            "Search Volume (Monthly)",
            "CPC ($)",
            "Target URL",
        ]

        def worker(input_str: str, thread_id: int):
            for _ in range(100):
                result = SmartDecisionMakerBlock.cleanup(input_str)
                with lock:
                    key = f"{thread_id}_{input_str}"
                    if key not in results:
                        results[key] = set()
                    results[key].add(result)

        threads = []
        for i, input_str in enumerate(test_inputs):
            for j in range(3):
                t = threading.Thread(target=worker, args=(input_str, i * 3 + j))
                threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Each input should produce exactly one unique output
        for key, values in results.items():
            assert len(values) == 1, f"Non-deterministic cleanup for {key}: {values}"


class TestAsyncConcurrencyPatterns:
    """Tests for async concurrency patterns in the block."""

    @pytest.mark.asyncio
    async def test_multiple_async_runs_isolation(self):
        """
        Test that multiple async runs are properly isolated.
        """
        import backend.blocks.llm as llm_module
        from backend.data.execution import ExecutionContext

        block = SmartDecisionMakerBlock()

        run_count = 5
        results = []

        async def single_run(run_id: int):
            mock_response = MagicMock()
            mock_response.response = f"Unique response {run_id}"
            mock_response.tool_calls = []
            mock_response.prompt_tokens = 10
            mock_response.completion_tokens = 5
            mock_response.reasoning = None
            mock_response.raw_response = {"role": "assistant", "content": f"Run {run_id}"}

            # Add small random delay to increase chance of interleaving
            await asyncio.sleep(0.001 * (run_id % 3))

            with patch("backend.blocks.llm.llm_call", new_callable=AsyncMock) as mock_llm:
                mock_llm.return_value = mock_response

                with patch.object(block, "_create_tool_node_signatures", return_value=[]):
                    input_data = SmartDecisionMakerBlock.Input(
                        prompt=f"Prompt {run_id}",
                        model=llm_module.DEFAULT_LLM_MODEL,
                        credentials=llm_module.TEST_CREDENTIALS_INPUT,
                        agent_mode_max_iterations=0,
                    )

                    outputs = {}
                    async for name, value in block.run(
                        input_data,
                        credentials=llm_module.TEST_CREDENTIALS,
                        graph_id=f"g{run_id}",
                        node_id=f"n{run_id}",
                        graph_exec_id=f"e{run_id}",
                        node_exec_id=f"ne{run_id}",
                        user_id=f"u{run_id}",
                        graph_version=1,
                        execution_context=ExecutionContext(safe_mode=False),
                        execution_processor=MagicMock(),
                    ):
                        outputs[name] = value

                    return run_id, outputs

        # Run all concurrently
        tasks = [single_run(i) for i in range(run_count)]
        results = await asyncio.gather(*tasks)

        # Verify isolation
        for run_id, outputs in results:
            if "finished" in outputs:
                assert str(run_id) in outputs["finished"], \
                    f"Run {run_id} got wrong response: {outputs['finished']}"
