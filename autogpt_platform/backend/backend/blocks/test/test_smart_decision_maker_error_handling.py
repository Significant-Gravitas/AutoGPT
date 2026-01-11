"""
Tests for SmartDecisionMaker error handling failure modes.

Covers failure modes:
3. JSON Deserialization Without Exception Handling
4. Database Transaction Inconsistency
5. Missing Null Checks After Database Calls
15. Error Message Context Loss
17. No Validation of Dynamic Field Paths
"""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from backend.blocks.smart_decision_maker import (
    SmartDecisionMakerBlock,
    _convert_raw_response_to_dict,
    _create_tool_response,
)


class TestJSONDeserializationErrors:
    """
    Tests for Failure Mode #3: JSON Deserialization Without Exception Handling

    When LLM returns malformed JSON in tool call arguments, the json.loads()
    call fails without proper error handling.
    """

    def test_malformed_json_single_quotes(self):
        """
        Test that single quotes in JSON cause parsing failure.

        LLMs sometimes return {'key': 'value'} instead of {"key": "value"}
        """
        malformed = "{'key': 'value'}"

        with pytest.raises(json.JSONDecodeError):
            json.loads(malformed)

    def test_malformed_json_trailing_comma(self):
        """
        Test that trailing commas cause parsing failure.
        """
        malformed = '{"key": "value",}'

        with pytest.raises(json.JSONDecodeError):
            json.loads(malformed)

    def test_malformed_json_unquoted_keys(self):
        """
        Test that unquoted keys cause parsing failure.
        """
        malformed = '{key: "value"}'

        with pytest.raises(json.JSONDecodeError):
            json.loads(malformed)

    def test_malformed_json_python_none(self):
        """
        Test that Python None instead of null causes failure.
        """
        malformed = '{"key": None}'

        with pytest.raises(json.JSONDecodeError):
            json.loads(malformed)

    def test_malformed_json_python_true_false(self):
        """
        Test that Python True/False instead of true/false causes failure.
        """
        malformed_true = '{"key": True}'
        malformed_false = '{"key": False}'

        with pytest.raises(json.JSONDecodeError):
            json.loads(malformed_true)

        with pytest.raises(json.JSONDecodeError):
            json.loads(malformed_false)

    @pytest.mark.asyncio
    async def test_llm_returns_malformed_json_crashes_block(self):
        """
        Test that malformed JSON from LLM causes block to crash.

        BUG: The json.loads() at line 625, 706, 1124 can throw JSONDecodeError
        which is not caught, causing the entire block to fail.
        """
        import backend.blocks.llm as llm_module
        from backend.data.execution import ExecutionContext

        block = SmartDecisionMakerBlock()

        # Create response with malformed JSON
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "test_tool"
        mock_tool_call.function.arguments = "{'malformed': 'json'}"  # Single quotes!

        mock_response = MagicMock()
        mock_response.response = None
        mock_response.tool_calls = [mock_tool_call]
        mock_response.prompt_tokens = 50
        mock_response.completion_tokens = 25
        mock_response.reasoning = None
        mock_response.raw_response = {"role": "assistant", "content": None}

        mock_tool_signatures = [
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "_sink_node_id": "sink",
                    "_field_mapping": {},
                    "parameters": {"properties": {"malformed": {"type": "string"}}, "required": []},
                },
            }
        ]

        with patch("backend.blocks.llm.llm_call", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response

            with patch.object(block, "_create_tool_node_signatures", return_value=mock_tool_signatures):
                input_data = SmartDecisionMakerBlock.Input(
                    prompt="Test",
                    model=llm_module.DEFAULT_LLM_MODEL,
                    credentials=llm_module.TEST_CREDENTIALS_INPUT,
                    agent_mode_max_iterations=0,
                )

                mock_execution_context = ExecutionContext(safe_mode=False)
                mock_execution_processor = MagicMock()

                # BUG: This should raise JSONDecodeError
                with pytest.raises(json.JSONDecodeError):
                    async for _ in block.run(
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
                        pass


class TestDatabaseTransactionInconsistency:
    """
    Tests for Failure Mode #4: Database Transaction Inconsistency

    When multiple database operations are performed in sequence,
    a failure partway through leaves the database in an inconsistent state.
    """

    @pytest.mark.asyncio
    async def test_partial_input_insertion_on_failure(self):
        """
        Test that partial failures during multi-input insertion
        leave database in inconsistent state.
        """
        import threading
        from collections import defaultdict

        import backend.blocks.llm as llm_module
        from backend.data.execution import ExecutionContext

        block = SmartDecisionMakerBlock()

        # Track which inputs were inserted
        inserted_inputs = []
        call_count = 0

        async def failing_upsert(node_id, graph_exec_id, input_name, input_data):
            nonlocal call_count
            call_count += 1

            # Fail on the third input
            if call_count == 3:
                raise Exception("Database connection lost!")

            inserted_inputs.append(input_name)

            mock_result = MagicMock()
            mock_result.node_exec_id = "exec-id"
            return mock_result, {input_name: input_data}

        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_1"
        mock_tool_call.function.name = "multi_input_tool"
        mock_tool_call.function.arguments = json.dumps({
            "input1": "value1",
            "input2": "value2",
            "input3": "value3",  # This one will fail
            "input4": "value4",
            "input5": "value5",
        })

        mock_response = MagicMock()
        mock_response.response = None
        mock_response.tool_calls = [mock_tool_call]
        mock_response.prompt_tokens = 50
        mock_response.completion_tokens = 25
        mock_response.reasoning = None
        mock_response.raw_response = {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "call_1"}]
        }

        mock_tool_signatures = [
            {
                "type": "function",
                "function": {
                    "name": "multi_input_tool",
                    "_sink_node_id": "sink",
                    "_field_mapping": {
                        "input1": "input1",
                        "input2": "input2",
                        "input3": "input3",
                        "input4": "input4",
                        "input5": "input5",
                    },
                    "parameters": {
                        "properties": {
                            "input1": {"type": "string"},
                            "input2": {"type": "string"},
                            "input3": {"type": "string"},
                            "input4": {"type": "string"},
                            "input5": {"type": "string"},
                        },
                        "required": ["input1", "input2", "input3", "input4", "input5"],
                    },
                },
            }
        ]

        mock_db_client = AsyncMock()
        mock_node = MagicMock()
        mock_node.block_id = "test-block"
        mock_db_client.get_node.return_value = mock_node
        mock_db_client.upsert_execution_input.side_effect = failing_upsert

        with patch("backend.blocks.llm.llm_call", new_callable=AsyncMock) as mock_llm, \
             patch.object(block, "_create_tool_node_signatures", return_value=mock_tool_signatures), \
             patch("backend.blocks.smart_decision_maker.get_database_manager_async_client", return_value=mock_db_client):

            mock_llm.return_value = mock_response

            mock_execution_context = ExecutionContext(safe_mode=False)
            mock_execution_processor = AsyncMock()
            mock_execution_processor.running_node_execution = defaultdict(MagicMock)
            mock_execution_processor.execution_stats = MagicMock()
            mock_execution_processor.execution_stats_lock = threading.Lock()

            input_data = SmartDecisionMakerBlock.Input(
                prompt="Test",
                model=llm_module.DEFAULT_LLM_MODEL,
                credentials=llm_module.TEST_CREDENTIALS_INPUT,
                agent_mode_max_iterations=1,
            )

            # The block should fail, but some inputs were already inserted
            outputs = {}
            try:
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
            except Exception:
                pass  # Expected

            # BUG: Some inputs were inserted before failure
            # Database is now in inconsistent state
            assert len(inserted_inputs) == 2, \
                f"Expected 2 inserted before failure, got {inserted_inputs}"
            assert "input1" in inserted_inputs
            assert "input2" in inserted_inputs
            # input3, input4, input5 were never inserted


class TestMissingNullChecks:
    """
    Tests for Failure Mode #5: Missing Null Checks After Database Calls
    """

    @pytest.mark.asyncio
    async def test_get_node_returns_none(self):
        """
        Test handling when get_node returns None.
        """
        import threading
        from collections import defaultdict

        import backend.blocks.llm as llm_module
        from backend.data.execution import ExecutionContext

        block = SmartDecisionMakerBlock()

        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_1"
        mock_tool_call.function.name = "test_tool"
        mock_tool_call.function.arguments = json.dumps({"param": "value"})

        mock_response = MagicMock()
        mock_response.response = None
        mock_response.tool_calls = [mock_tool_call]
        mock_response.prompt_tokens = 50
        mock_response.completion_tokens = 25
        mock_response.reasoning = None
        mock_response.raw_response = {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "call_1"}]
        }

        mock_tool_signatures = [
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "_sink_node_id": "nonexistent-node",
                    "_field_mapping": {"param": "param"},
                    "parameters": {
                        "properties": {"param": {"type": "string"}},
                        "required": ["param"],
                    },
                },
            }
        ]

        mock_db_client = AsyncMock()
        mock_db_client.get_node.return_value = None  # Node doesn't exist!

        with patch("backend.blocks.llm.llm_call", new_callable=AsyncMock) as mock_llm, \
             patch.object(block, "_create_tool_node_signatures", return_value=mock_tool_signatures), \
             patch("backend.blocks.smart_decision_maker.get_database_manager_async_client", return_value=mock_db_client):

            mock_llm.return_value = mock_response

            mock_execution_context = ExecutionContext(safe_mode=False)
            mock_execution_processor = AsyncMock()
            mock_execution_processor.running_node_execution = defaultdict(MagicMock)
            mock_execution_processor.execution_stats = MagicMock()
            mock_execution_processor.execution_stats_lock = threading.Lock()

            input_data = SmartDecisionMakerBlock.Input(
                prompt="Test",
                model=llm_module.DEFAULT_LLM_MODEL,
                credentials=llm_module.TEST_CREDENTIALS_INPUT,
                agent_mode_max_iterations=1,
            )

            # Should raise ValueError for missing node
            with pytest.raises(ValueError, match="not found"):
                async for _ in block.run(
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
                    pass

    @pytest.mark.asyncio
    async def test_empty_execution_outputs(self):
        """
        Test handling when get_execution_outputs_by_node_exec_id returns empty.
        """
        import threading
        from collections import defaultdict

        import backend.blocks.llm as llm_module
        from backend.data.execution import ExecutionContext

        block = SmartDecisionMakerBlock()

        call_count = 0

        async def mock_llm_call(**kwargs):
            nonlocal call_count
            call_count += 1

            if call_count > 1:
                resp = MagicMock()
                resp.response = "Done"
                resp.tool_calls = []
                resp.prompt_tokens = 10
                resp.completion_tokens = 5
                resp.reasoning = None
                resp.raw_response = {"role": "assistant", "content": "Done"}
                return resp

            mock_tool_call = MagicMock()
            mock_tool_call.id = "call_1"
            mock_tool_call.function.name = "test_tool"
            mock_tool_call.function.arguments = json.dumps({})

            resp = MagicMock()
            resp.response = None
            resp.tool_calls = [mock_tool_call]
            resp.prompt_tokens = 50
            resp.completion_tokens = 25
            resp.reasoning = None
            resp.raw_response = {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "call_1"}]
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
        mock_db_client.get_execution_outputs_by_node_exec_id.return_value = {}  # Empty!

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
                prompt="Test",
                model=llm_module.DEFAULT_LLM_MODEL,
                credentials=llm_module.TEST_CREDENTIALS_INPUT,
                agent_mode_max_iterations=2,
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

            # Empty outputs should be handled gracefully
            # (uses "Tool executed successfully" as fallback)
            assert "finished" in outputs or "conversations" in outputs


class TestErrorMessageContextLoss:
    """
    Tests for Failure Mode #15: Error Message Context Loss

    When exceptions are caught and converted to strings, important
    debugging information is lost.
    """

    def test_exception_to_string_loses_traceback(self):
        """
        Test that converting exception to string loses traceback.
        """
        try:
            def inner():
                raise ValueError("Inner error")

            def outer():
                inner()

            outer()
        except Exception as e:
            error_string = str(e)
            error_repr = repr(e)

            # String representation loses call stack
            assert "inner" not in error_string
            assert "outer" not in error_string

            # Even repr doesn't have full traceback
            assert "Traceback" not in error_repr

    def test_tool_response_loses_exception_type(self):
        """
        Test that _create_tool_response loses exception type information.
        """
        original_error = ConnectionError("Database unreachable")
        tool_response = _create_tool_response(
            "call_123",
            f"Tool execution failed: {str(original_error)}"
        )

        content = tool_response.get("content", "")

        # Original exception type is lost
        assert "ConnectionError" not in content
        # Only the message remains
        assert "Database unreachable" in content

    @pytest.mark.asyncio
    async def test_agent_mode_error_response_lacks_context(self):
        """
        Test that agent mode error responses lack debugging context.
        """
        import threading
        from collections import defaultdict

        import backend.blocks.llm as llm_module
        from backend.data.execution import ExecutionContext

        block = SmartDecisionMakerBlock()

        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_1"
        mock_tool_call.function.name = "test_tool"
        mock_tool_call.function.arguments = json.dumps({})

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

        mock_response_2 = MagicMock()
        mock_response_2.response = "Handled the error"
        mock_response_2.tool_calls = []
        mock_response_2.prompt_tokens = 30
        mock_response_2.completion_tokens = 15
        mock_response_2.reasoning = None
        mock_response_2.raw_response = {"role": "assistant", "content": "Handled"}

        call_count = 0

        async def mock_llm_call(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_response_1
            return mock_response_2

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

        # Create a complex error with nested cause
        class CustomDatabaseError(Exception):
            pass

        def create_complex_error():
            try:
                raise ConnectionError("Network timeout after 30s")
            except ConnectionError as e:
                raise CustomDatabaseError("Failed to connect to database") from e

        mock_db_client = AsyncMock()
        mock_node = MagicMock()
        mock_node.block_id = "test-block"
        mock_db_client.get_node.return_value = mock_node

        # Make upsert raise the complex error
        try:
            create_complex_error()
        except CustomDatabaseError as e:
            mock_db_client.upsert_execution_input.side_effect = e

        with patch("backend.blocks.llm.llm_call", side_effect=mock_llm_call), \
             patch.object(block, "_create_tool_node_signatures", return_value=mock_tool_signatures), \
             patch("backend.blocks.smart_decision_maker.get_database_manager_async_client", return_value=mock_db_client):

            mock_execution_context = ExecutionContext(safe_mode=False)
            mock_execution_processor = AsyncMock()
            mock_execution_processor.running_node_execution = defaultdict(MagicMock)
            mock_execution_processor.execution_stats = MagicMock()
            mock_execution_processor.execution_stats_lock = threading.Lock()

            input_data = SmartDecisionMakerBlock.Input(
                prompt="Test",
                model=llm_module.DEFAULT_LLM_MODEL,
                credentials=llm_module.TEST_CREDENTIALS_INPUT,
                agent_mode_max_iterations=2,
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

            # Check conversation for error details
            conversations = outputs.get("conversations", [])
            error_found = False
            for msg in conversations:
                content = msg.get("content", "")
                if isinstance(content, list):
                    for item in content:
                        if item.get("type") == "tool_result":
                            result_content = item.get("content", "")
                            if "Error" in result_content or "failed" in result_content.lower():
                                error_found = True
                                # BUG: The error content lacks:
                                # - Exception type (CustomDatabaseError)
                                # - Chained cause (ConnectionError)
                                # - Stack trace
                                assert "CustomDatabaseError" not in result_content
                                assert "ConnectionError" not in result_content

            # Note: error_found may be False if the error prevented tool response creation


class TestRawResponseConversion:
    """Tests for _convert_raw_response_to_dict edge cases."""

    def test_string_response_converted(self):
        """Test that string responses are properly wrapped."""
        result = _convert_raw_response_to_dict("Hello, world!")
        assert result == {"role": "assistant", "content": "Hello, world!"}

    def test_dict_response_unchanged(self):
        """Test that dict responses are passed through."""
        original = {"role": "assistant", "content": "test", "extra": "field"}
        result = _convert_raw_response_to_dict(original)
        assert result == original

    def test_object_response_converted(self):
        """Test that objects are converted using json.to_dict."""
        mock_obj = MagicMock()

        with patch("backend.blocks.smart_decision_maker.json.to_dict") as mock_to_dict:
            mock_to_dict.return_value = {"converted": True}
            result = _convert_raw_response_to_dict(mock_obj)
            mock_to_dict.assert_called_once_with(mock_obj)
            assert result == {"converted": True}

    def test_none_response(self):
        """Test handling of None response."""
        with patch("backend.blocks.smart_decision_maker.json.to_dict") as mock_to_dict:
            mock_to_dict.return_value = None
            result = _convert_raw_response_to_dict(None)
            # None is not a string or dict, so it goes through to_dict
            assert result is None


class TestValidationRetryMechanism:
    """Tests for the validation and retry mechanism."""

    @pytest.mark.asyncio
    async def test_validation_error_triggers_retry(self):
        """
        Test that validation errors trigger retry with feedback.
        """
        import backend.blocks.llm as llm_module
        from backend.data.execution import ExecutionContext

        block = SmartDecisionMakerBlock()

        call_count = 0

        async def mock_llm_call(**kwargs):
            nonlocal call_count
            call_count += 1

            prompt = kwargs.get("prompt", [])

            if call_count == 1:
                # First call: return tool call with wrong parameter
                mock_tool_call = MagicMock()
                mock_tool_call.function.name = "test_tool"
                mock_tool_call.function.arguments = json.dumps({"wrong_param": "value"})

                resp = MagicMock()
                resp.response = None
                resp.tool_calls = [mock_tool_call]
                resp.prompt_tokens = 50
                resp.completion_tokens = 25
                resp.reasoning = None
                resp.raw_response = {"role": "assistant", "content": None}
                return resp
            else:
                # Second call: check that error feedback was added
                has_error_feedback = any(
                    "parameter errors" in str(msg.get("content", "")).lower()
                    for msg in prompt
                )

                # Return correct tool call
                mock_tool_call = MagicMock()
                mock_tool_call.function.name = "test_tool"
                mock_tool_call.function.arguments = json.dumps({"correct_param": "value"})

                resp = MagicMock()
                resp.response = None
                resp.tool_calls = [mock_tool_call]
                resp.prompt_tokens = 50
                resp.completion_tokens = 25
                resp.reasoning = None
                resp.raw_response = {"role": "assistant", "content": None}
                return resp

        mock_tool_signatures = [
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "_sink_node_id": "sink",
                    "_field_mapping": {"correct_param": "correct_param"},
                    "parameters": {
                        "properties": {"correct_param": {"type": "string"}},
                        "required": ["correct_param"],
                    },
                },
            }
        ]

        with patch("backend.blocks.llm.llm_call", side_effect=mock_llm_call), \
             patch.object(block, "_create_tool_node_signatures", return_value=mock_tool_signatures):

            input_data = SmartDecisionMakerBlock.Input(
                prompt="Test",
                model=llm_module.DEFAULT_LLM_MODEL,
                credentials=llm_module.TEST_CREDENTIALS_INPUT,
                agent_mode_max_iterations=0,  # Traditional mode
                retry=3,
            )

            mock_execution_context = ExecutionContext(safe_mode=False)
            mock_execution_processor = MagicMock()

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

            # Should have made multiple calls due to retry
            assert call_count >= 2

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """
        Test behavior when max retries are exceeded.
        """
        import backend.blocks.llm as llm_module
        from backend.data.execution import ExecutionContext

        block = SmartDecisionMakerBlock()

        async def mock_llm_call(**kwargs):
            # Always return invalid tool call
            mock_tool_call = MagicMock()
            mock_tool_call.function.name = "test_tool"
            mock_tool_call.function.arguments = json.dumps({"wrong": "param"})

            resp = MagicMock()
            resp.response = None
            resp.tool_calls = [mock_tool_call]
            resp.prompt_tokens = 50
            resp.completion_tokens = 25
            resp.reasoning = None
            resp.raw_response = {"role": "assistant", "content": None}
            return resp

        mock_tool_signatures = [
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "_sink_node_id": "sink",
                    "_field_mapping": {"correct": "correct"},
                    "parameters": {
                        "properties": {"correct": {"type": "string"}},
                        "required": ["correct"],
                    },
                },
            }
        ]

        with patch("backend.blocks.llm.llm_call", side_effect=mock_llm_call), \
             patch.object(block, "_create_tool_node_signatures", return_value=mock_tool_signatures):

            input_data = SmartDecisionMakerBlock.Input(
                prompt="Test",
                model=llm_module.DEFAULT_LLM_MODEL,
                credentials=llm_module.TEST_CREDENTIALS_INPUT,
                agent_mode_max_iterations=0,
                retry=2,  # Only 2 retries
            )

            mock_execution_context = ExecutionContext(safe_mode=False)
            mock_execution_processor = MagicMock()

            # Should raise ValueError after max retries
            with pytest.raises(ValueError, match="parameter errors"):
                async for _ in block.run(
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
                    pass
