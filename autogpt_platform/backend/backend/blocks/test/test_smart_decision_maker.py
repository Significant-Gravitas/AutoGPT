import logging
import threading
from collections import defaultdict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.api.model import CreateGraph
from backend.api.rest_api import AgentServer
from backend.data.execution import ExecutionContext
from backend.data.model import ProviderName, User
from backend.usecases.sample import create_test_graph, create_test_user
from backend.util.test import SpinTestServer, wait_execution

logger = logging.getLogger(__name__)


async def create_graph(s: SpinTestServer, g, u: User):
    logger.info("Creating graph for user %s", u.id)
    return await s.agent_server.test_create_graph(CreateGraph(graph=g), u.id)


async def create_credentials(s: SpinTestServer, u: User):
    import backend.blocks.llm as llm_module

    provider = ProviderName.OPENAI
    credentials = llm_module.TEST_CREDENTIALS
    return await s.agent_server.test_create_credentials(u.id, provider, credentials)


async def execute_graph(
    agent_server: AgentServer,
    test_graph,
    test_user: User,
    input_data: dict,
    num_execs: int = 4,
) -> str:
    logger.info("Executing graph %s for user %s", test_graph.id, test_user.id)
    logger.info("Input data: %s", input_data)

    # --- Test adding new executions --- #
    graph_exec = await agent_server.test_execute_graph(
        user_id=test_user.id,
        graph_id=test_graph.id,
        graph_version=test_graph.version,
        node_input=input_data,
    )
    logger.info("Created execution with ID: %s", graph_exec.id)

    # Execution queue should be empty
    logger.info("Waiting for execution to complete...")
    result = await wait_execution(test_user.id, graph_exec.id, 30)
    logger.info("Execution completed with %d results", len(result))
    return graph_exec.id


@pytest.mark.asyncio(loop_scope="session")
async def test_graph_validation_with_tool_nodes_correct(server: SpinTestServer):
    from backend.blocks.agent import AgentExecutorBlock
    from backend.blocks.smart_decision_maker import SmartDecisionMakerBlock
    from backend.data import graph

    test_user = await create_test_user()
    test_tool_graph = await create_graph(server, create_test_graph(), test_user)
    creds = await create_credentials(server, test_user)

    nodes = [
        graph.Node(
            block_id=SmartDecisionMakerBlock().id,
            input_default={
                "prompt": "Hello, World!",
                "credentials": creds,
            },
        ),
        graph.Node(
            block_id=AgentExecutorBlock().id,
            input_default={
                "graph_id": test_tool_graph.id,
                "graph_version": test_tool_graph.version,
                "input_schema": test_tool_graph.input_schema,
                "output_schema": test_tool_graph.output_schema,
            },
        ),
    ]

    links = [
        graph.Link(
            source_id=nodes[0].id,
            sink_id=nodes[1].id,
            source_name="tools_^_sample_tool_input_1",
            sink_name="input_1",
        ),
        graph.Link(
            source_id=nodes[0].id,
            sink_id=nodes[1].id,
            source_name="tools_^_sample_tool_input_2",
            sink_name="input_2",
        ),
    ]

    test_graph = graph.Graph(
        name="TestGraph",
        description="Test graph",
        nodes=nodes,
        links=links,
    )
    test_graph = await create_graph(server, test_graph, test_user)


@pytest.mark.asyncio(loop_scope="session")
async def test_smart_decision_maker_function_signature(server: SpinTestServer):
    from backend.blocks.agent import AgentExecutorBlock
    from backend.blocks.basic import StoreValueBlock
    from backend.blocks.smart_decision_maker import SmartDecisionMakerBlock
    from backend.data import graph

    test_user = await create_test_user()
    test_tool_graph = await create_graph(server, create_test_graph(), test_user)
    creds = await create_credentials(server, test_user)

    nodes = [
        graph.Node(
            block_id=SmartDecisionMakerBlock().id,
            input_default={
                "prompt": "Hello, World!",
                "credentials": creds,
            },
        ),
        graph.Node(
            block_id=AgentExecutorBlock().id,
            input_default={
                "graph_id": test_tool_graph.id,
                "graph_version": test_tool_graph.version,
                "input_schema": test_tool_graph.input_schema,
                "output_schema": test_tool_graph.output_schema,
            },
        ),
        graph.Node(
            block_id=StoreValueBlock().id,
        ),
    ]

    links = [
        graph.Link(
            source_id=nodes[0].id,
            sink_id=nodes[1].id,
            source_name="tools_^_sample_tool_input_1",
            sink_name="input_1",
        ),
        graph.Link(
            source_id=nodes[0].id,
            sink_id=nodes[1].id,
            source_name="tools_^_sample_tool_input_2",
            sink_name="input_2",
        ),
        graph.Link(
            source_id=nodes[0].id,
            sink_id=nodes[2].id,
            source_name="tools_^_store_value_input",
            sink_name="input",
        ),
    ]

    test_graph = graph.Graph(
        name="TestGraph",
        description="Test graph",
        nodes=nodes,
        links=links,
    )
    test_graph = await create_graph(server, test_graph, test_user)

    tool_functions = await SmartDecisionMakerBlock._create_tool_node_signatures(
        test_graph.nodes[0].id
    )
    assert tool_functions is not None, "Tool functions should not be None"

    assert (
        len(tool_functions) == 2
    ), f"Expected 2 tool functions, got {len(tool_functions)}"

    # Check the first tool function (testgraph)
    assert tool_functions[0]["type"] == "function"
    assert tool_functions[0]["function"]["name"] == "testgraph"
    assert tool_functions[0]["function"]["description"] == "Test graph description"
    assert "input_1" in tool_functions[0]["function"]["parameters"]["properties"]
    assert "input_2" in tool_functions[0]["function"]["parameters"]["properties"]

    # Check the second tool function (storevalueblock)
    assert tool_functions[1]["type"] == "function"
    assert tool_functions[1]["function"]["name"] == "storevalueblock"
    assert "input" in tool_functions[1]["function"]["parameters"]["properties"]
    assert (
        tool_functions[1]["function"]["parameters"]["properties"]["input"][
            "description"
        ]
        == "Trigger the block to produce the output. The value is only used when `data` is None."
    )


@pytest.mark.asyncio
async def test_smart_decision_maker_tracks_llm_stats():
    """Test that SmartDecisionMakerBlock correctly tracks LLM usage stats."""
    import backend.blocks.llm as llm_module
    from backend.blocks.smart_decision_maker import SmartDecisionMakerBlock

    block = SmartDecisionMakerBlock()

    # Mock the llm.llm_call function to return controlled data
    mock_response = MagicMock()
    mock_response.response = "I need to think about this."
    mock_response.tool_calls = None  # No tool calls for simplicity
    mock_response.prompt_tokens = 50
    mock_response.completion_tokens = 25
    mock_response.reasoning = None
    mock_response.raw_response = {
        "role": "assistant",
        "content": "I need to think about this.",
    }

    # Mock the _create_tool_node_signatures method to avoid database calls

    with patch(
        "backend.blocks.llm.llm_call",
        new_callable=AsyncMock,
        return_value=mock_response,
    ), patch.object(
        SmartDecisionMakerBlock,
        "_create_tool_node_signatures",
        new_callable=AsyncMock,
        return_value=[],
    ):

        # Create test input
        input_data = SmartDecisionMakerBlock.Input(
            prompt="Should I continue with this task?",
            model=llm_module.DEFAULT_LLM_MODEL,
            credentials=llm_module.TEST_CREDENTIALS_INPUT,  # type: ignore
            agent_mode_max_iterations=0,
        )

        # Execute the block
        outputs = {}
        # Create execution context

        mock_execution_context = ExecutionContext(safe_mode=False)

        # Create a mock execution processor for tests

        mock_execution_processor = MagicMock()

        async for output_name, output_data in block.run(
            input_data,
            credentials=llm_module.TEST_CREDENTIALS,
            graph_id="test-graph-id",
            node_id="test-node-id",
            graph_exec_id="test-exec-id",
            node_exec_id="test-node-exec-id",
            user_id="test-user-id",
            graph_version=1,
            execution_context=mock_execution_context,
            execution_processor=mock_execution_processor,
        ):
            outputs[output_name] = output_data

        # Verify stats tracking
        assert block.execution_stats is not None
        assert block.execution_stats.input_token_count == 50
        assert block.execution_stats.output_token_count == 25
        assert block.execution_stats.llm_call_count == 1

        # Verify outputs
        assert "finished" in outputs  # Should have finished since no tool calls
        assert outputs["finished"] == "I need to think about this."


@pytest.mark.asyncio
async def test_smart_decision_maker_parameter_validation():
    """Test that SmartDecisionMakerBlock correctly validates tool call parameters."""
    import backend.blocks.llm as llm_module
    from backend.blocks.smart_decision_maker import SmartDecisionMakerBlock

    block = SmartDecisionMakerBlock()

    # Mock tool functions with specific parameter schema
    mock_tool_functions = [
        {
            "type": "function",
            "function": {
                "name": "search_keywords",
                "description": "Search for keywords with difficulty filtering",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "max_keyword_difficulty": {
                            "type": "integer",
                            "description": "Maximum keyword difficulty (required)",
                        },
                        "optional_param": {
                            "type": "string",
                            "description": "Optional parameter with default",
                            "default": "default_value",
                        },
                    },
                    "required": ["query", "max_keyword_difficulty"],
                },
                "_sink_node_id": "test-sink-node-id",
            },
        }
    ]

    # Test case 1: Tool call with TYPO in parameter name (should retry and eventually fail)
    mock_tool_call_with_typo = MagicMock()
    mock_tool_call_with_typo.function.name = "search_keywords"
    mock_tool_call_with_typo.function.arguments = '{"query": "test", "maximum_keyword_difficulty": 50}'  # TYPO: maximum instead of max

    mock_response_with_typo = MagicMock()
    mock_response_with_typo.response = None
    mock_response_with_typo.tool_calls = [mock_tool_call_with_typo]
    mock_response_with_typo.prompt_tokens = 50
    mock_response_with_typo.completion_tokens = 25
    mock_response_with_typo.reasoning = None
    mock_response_with_typo.raw_response = {"role": "assistant", "content": None}

    with patch(
        "backend.blocks.llm.llm_call",
        new_callable=AsyncMock,
        return_value=mock_response_with_typo,
    ) as mock_llm_call, patch.object(
        SmartDecisionMakerBlock,
        "_create_tool_node_signatures",
        new_callable=AsyncMock,
        return_value=mock_tool_functions,
    ):

        input_data = SmartDecisionMakerBlock.Input(
            prompt="Search for keywords",
            model=llm_module.DEFAULT_LLM_MODEL,
            credentials=llm_module.TEST_CREDENTIALS_INPUT,  # type: ignore
            retry=2,  # Set retry to 2 for testing
            agent_mode_max_iterations=0,
        )

        # Create execution context

        mock_execution_context = ExecutionContext(safe_mode=False)

        # Create a mock execution processor for tests

        mock_execution_processor = MagicMock()

        # Should raise ValueError after retries due to typo'd parameter name
        with pytest.raises(ValueError) as exc_info:
            outputs = {}
            async for output_name, output_data in block.run(
                input_data,
                credentials=llm_module.TEST_CREDENTIALS,
                graph_id="test-graph-id",
                node_id="test-node-id",
                graph_exec_id="test-exec-id",
                node_exec_id="test-node-exec-id",
                user_id="test-user-id",
                graph_version=1,
                execution_context=mock_execution_context,
                execution_processor=mock_execution_processor,
            ):
                outputs[output_name] = output_data

        # Verify error message contains details about the typo
        error_msg = str(exc_info.value)
        assert "Tool call 'search_keywords' has parameter errors" in error_msg
        assert "Unknown parameters: ['maximum_keyword_difficulty']" in error_msg

        # Verify that LLM was called the expected number of times (retries)
        assert mock_llm_call.call_count == 2  # Should retry based on input_data.retry

    # Test case 2: Tool call missing REQUIRED parameter (should raise ValueError)
    mock_tool_call_missing_required = MagicMock()
    mock_tool_call_missing_required.function.name = "search_keywords"
    mock_tool_call_missing_required.function.arguments = (
        '{"query": "test"}'  # Missing required max_keyword_difficulty
    )

    mock_response_missing_required = MagicMock()
    mock_response_missing_required.response = None
    mock_response_missing_required.tool_calls = [mock_tool_call_missing_required]
    mock_response_missing_required.prompt_tokens = 50
    mock_response_missing_required.completion_tokens = 25
    mock_response_missing_required.reasoning = None
    mock_response_missing_required.raw_response = {"role": "assistant", "content": None}

    with patch(
        "backend.blocks.llm.llm_call",
        new_callable=AsyncMock,
        return_value=mock_response_missing_required,
    ), patch.object(
        SmartDecisionMakerBlock,
        "_create_tool_node_signatures",
        new_callable=AsyncMock,
        return_value=mock_tool_functions,
    ):

        input_data = SmartDecisionMakerBlock.Input(
            prompt="Search for keywords",
            model=llm_module.DEFAULT_LLM_MODEL,
            credentials=llm_module.TEST_CREDENTIALS_INPUT,  # type: ignore
            agent_mode_max_iterations=0,
        )

        # Create execution context

        mock_execution_context = ExecutionContext(safe_mode=False)

        # Create a mock execution processor for tests

        mock_execution_processor = MagicMock()

        # Should raise ValueError due to missing required parameter
        with pytest.raises(ValueError) as exc_info:
            outputs = {}
            async for output_name, output_data in block.run(
                input_data,
                credentials=llm_module.TEST_CREDENTIALS,
                graph_id="test-graph-id",
                node_id="test-node-id",
                graph_exec_id="test-exec-id",
                node_exec_id="test-node-exec-id",
                user_id="test-user-id",
                graph_version=1,
                execution_context=mock_execution_context,
                execution_processor=mock_execution_processor,
            ):
                outputs[output_name] = output_data

        error_msg = str(exc_info.value)
        assert "Tool call 'search_keywords' has parameter errors" in error_msg
        assert "Missing required parameters: ['max_keyword_difficulty']" in error_msg

    # Test case 3: Valid tool call with OPTIONAL parameter missing (should succeed)
    mock_tool_call_valid = MagicMock()
    mock_tool_call_valid.function.name = "search_keywords"
    mock_tool_call_valid.function.arguments = '{"query": "test", "max_keyword_difficulty": 50}'  # optional_param missing, but that's OK

    mock_response_valid = MagicMock()
    mock_response_valid.response = None
    mock_response_valid.tool_calls = [mock_tool_call_valid]
    mock_response_valid.prompt_tokens = 50
    mock_response_valid.completion_tokens = 25
    mock_response_valid.reasoning = None
    mock_response_valid.raw_response = {"role": "assistant", "content": None}

    with patch(
        "backend.blocks.llm.llm_call",
        new_callable=AsyncMock,
        return_value=mock_response_valid,
    ), patch.object(
        SmartDecisionMakerBlock,
        "_create_tool_node_signatures",
        new_callable=AsyncMock,
        return_value=mock_tool_functions,
    ):

        input_data = SmartDecisionMakerBlock.Input(
            prompt="Search for keywords",
            model=llm_module.DEFAULT_LLM_MODEL,
            credentials=llm_module.TEST_CREDENTIALS_INPUT,  # type: ignore
            agent_mode_max_iterations=0,
        )

        # Should succeed - optional parameter missing is OK
        outputs = {}
        # Create execution context

        mock_execution_context = ExecutionContext(safe_mode=False)

        # Create a mock execution processor for tests

        mock_execution_processor = MagicMock()

        async for output_name, output_data in block.run(
            input_data,
            credentials=llm_module.TEST_CREDENTIALS,
            graph_id="test-graph-id",
            node_id="test-node-id",
            graph_exec_id="test-exec-id",
            node_exec_id="test-node-exec-id",
            user_id="test-user-id",
            graph_version=1,
            execution_context=mock_execution_context,
            execution_processor=mock_execution_processor,
        ):
            outputs[output_name] = output_data

        # Verify tool outputs were generated correctly
        assert "tools_^_test-sink-node-id_~_query" in outputs
        assert outputs["tools_^_test-sink-node-id_~_query"] == "test"
        assert "tools_^_test-sink-node-id_~_max_keyword_difficulty" in outputs
        assert outputs["tools_^_test-sink-node-id_~_max_keyword_difficulty"] == 50
        # Optional parameter should be None when not provided
        assert "tools_^_test-sink-node-id_~_optional_param" in outputs
        assert outputs["tools_^_test-sink-node-id_~_optional_param"] is None

    # Test case 4: Valid tool call with ALL parameters (should succeed)
    mock_tool_call_all_params = MagicMock()
    mock_tool_call_all_params.function.name = "search_keywords"
    mock_tool_call_all_params.function.arguments = '{"query": "test", "max_keyword_difficulty": 50, "optional_param": "custom_value"}'

    mock_response_all_params = MagicMock()
    mock_response_all_params.response = None
    mock_response_all_params.tool_calls = [mock_tool_call_all_params]
    mock_response_all_params.prompt_tokens = 50
    mock_response_all_params.completion_tokens = 25
    mock_response_all_params.reasoning = None
    mock_response_all_params.raw_response = {"role": "assistant", "content": None}

    with patch(
        "backend.blocks.llm.llm_call",
        new_callable=AsyncMock,
        return_value=mock_response_all_params,
    ), patch.object(
        SmartDecisionMakerBlock,
        "_create_tool_node_signatures",
        new_callable=AsyncMock,
        return_value=mock_tool_functions,
    ):

        input_data = SmartDecisionMakerBlock.Input(
            prompt="Search for keywords",
            model=llm_module.DEFAULT_LLM_MODEL,
            credentials=llm_module.TEST_CREDENTIALS_INPUT,  # type: ignore
            agent_mode_max_iterations=0,
        )

        # Should succeed with all parameters
        outputs = {}
        # Create execution context

        mock_execution_context = ExecutionContext(safe_mode=False)

        # Create a mock execution processor for tests

        mock_execution_processor = MagicMock()

        async for output_name, output_data in block.run(
            input_data,
            credentials=llm_module.TEST_CREDENTIALS,
            graph_id="test-graph-id",
            node_id="test-node-id",
            graph_exec_id="test-exec-id",
            node_exec_id="test-node-exec-id",
            user_id="test-user-id",
            graph_version=1,
            execution_context=mock_execution_context,
            execution_processor=mock_execution_processor,
        ):
            outputs[output_name] = output_data

        # Verify all tool outputs were generated correctly
        assert outputs["tools_^_test-sink-node-id_~_query"] == "test"
        assert outputs["tools_^_test-sink-node-id_~_max_keyword_difficulty"] == 50
        assert outputs["tools_^_test-sink-node-id_~_optional_param"] == "custom_value"


@pytest.mark.asyncio
async def test_smart_decision_maker_raw_response_conversion():
    """Test that SmartDecisionMaker correctly handles different raw_response types with retry mechanism."""
    import backend.blocks.llm as llm_module
    from backend.blocks.smart_decision_maker import SmartDecisionMakerBlock

    block = SmartDecisionMakerBlock()

    # Mock tool functions
    mock_tool_functions = [
        {
            "type": "function",
            "function": {
                "name": "test_tool",
                "parameters": {
                    "type": "object",
                    "properties": {"param": {"type": "string"}},
                    "required": ["param"],
                },
                "_sink_node_id": "test-sink-node-id",
            },
        }
    ]

    # Test case 1: Simulate ChatCompletionMessage raw_response that caused the original error
    class MockChatCompletionMessage:
        """Simulate OpenAI's ChatCompletionMessage object that lacks .get() method"""

        def __init__(self, role, content, tool_calls=None):
            self.role = role
            self.content = content
            self.tool_calls = tool_calls or []

        # This is what caused the error - no .get() method
        # def get(self, key, default=None):  # Intentionally missing

    # First response: has invalid parameter name (triggers retry)
    mock_tool_call_invalid = MagicMock()
    mock_tool_call_invalid.function.name = "test_tool"
    mock_tool_call_invalid.function.arguments = (
        '{"wrong_param": "test_value"}'  # Invalid parameter name
    )

    mock_response_retry = MagicMock()
    mock_response_retry.response = None
    mock_response_retry.tool_calls = [mock_tool_call_invalid]
    mock_response_retry.prompt_tokens = 50
    mock_response_retry.completion_tokens = 25
    mock_response_retry.reasoning = None
    # This would cause the original error without our fix
    mock_response_retry.raw_response = MockChatCompletionMessage(
        role="assistant", content=None, tool_calls=[mock_tool_call_invalid]
    )

    # Second response: successful (correct parameter name)
    mock_tool_call_valid = MagicMock()
    mock_tool_call_valid.function.name = "test_tool"
    mock_tool_call_valid.function.arguments = (
        '{"param": "test_value"}'  # Correct parameter name
    )

    mock_response_success = MagicMock()
    mock_response_success.response = None
    mock_response_success.tool_calls = [mock_tool_call_valid]
    mock_response_success.prompt_tokens = 50
    mock_response_success.completion_tokens = 25
    mock_response_success.reasoning = None
    mock_response_success.raw_response = MockChatCompletionMessage(
        role="assistant", content=None, tool_calls=[mock_tool_call_valid]
    )

    # Mock llm_call to return different responses on different calls

    with patch(
        "backend.blocks.llm.llm_call", new_callable=AsyncMock
    ) as mock_llm_call, patch.object(
        SmartDecisionMakerBlock,
        "_create_tool_node_signatures",
        new_callable=AsyncMock,
        return_value=mock_tool_functions,
    ):
        # First call returns response that will trigger retry due to validation error
        # Second call returns successful response
        mock_llm_call.side_effect = [mock_response_retry, mock_response_success]

        input_data = SmartDecisionMakerBlock.Input(
            prompt="Test prompt",
            model=llm_module.DEFAULT_LLM_MODEL,
            credentials=llm_module.TEST_CREDENTIALS_INPUT,  # type: ignore
            retry=2,
            agent_mode_max_iterations=0,
        )

        # Should succeed after retry, demonstrating our helper function works
        outputs = {}
        # Create execution context

        mock_execution_context = ExecutionContext(safe_mode=False)

        # Create a mock execution processor for tests

        mock_execution_processor = MagicMock()

        async for output_name, output_data in block.run(
            input_data,
            credentials=llm_module.TEST_CREDENTIALS,
            graph_id="test-graph-id",
            node_id="test-node-id",
            graph_exec_id="test-exec-id",
            node_exec_id="test-node-exec-id",
            user_id="test-user-id",
            graph_version=1,
            execution_context=mock_execution_context,
            execution_processor=mock_execution_processor,
        ):
            outputs[output_name] = output_data

        # Verify the tool output was generated successfully
        assert "tools_^_test-sink-node-id_~_param" in outputs
        assert outputs["tools_^_test-sink-node-id_~_param"] == "test_value"

        # Verify conversation history was properly maintained
        assert "conversations" in outputs
        conversations = outputs["conversations"]
        assert len(conversations) > 0

        # The conversations should contain properly converted raw_response objects as dicts
        # This would have failed with the original bug due to ChatCompletionMessage.get() error
        for msg in conversations:
            assert isinstance(msg, dict), f"Expected dict, got {type(msg)}"
            if msg.get("role") == "assistant":
                # Should have been converted from ChatCompletionMessage to dict
                assert "role" in msg

        # Verify LLM was called twice (initial + 1 retry)
        assert mock_llm_call.call_count == 2

    # Test case 2: Test with different raw_response types (Ollama string, dict)
    # Test Ollama string response
    mock_response_ollama = MagicMock()
    mock_response_ollama.response = "I'll help you with that."
    mock_response_ollama.tool_calls = None
    mock_response_ollama.prompt_tokens = 30
    mock_response_ollama.completion_tokens = 15
    mock_response_ollama.reasoning = None
    mock_response_ollama.raw_response = (
        "I'll help you with that."  # Ollama returns string
    )

    with patch(
        "backend.blocks.llm.llm_call",
        new_callable=AsyncMock,
        return_value=mock_response_ollama,
    ), patch.object(
        SmartDecisionMakerBlock,
        "_create_tool_node_signatures",
        new_callable=AsyncMock,
        return_value=[],  # No tools for this test
    ):
        input_data = SmartDecisionMakerBlock.Input(
            prompt="Simple prompt",
            model=llm_module.DEFAULT_LLM_MODEL,
            credentials=llm_module.TEST_CREDENTIALS_INPUT,  # type: ignore
            agent_mode_max_iterations=0,
        )

        outputs = {}
        # Create execution context

        mock_execution_context = ExecutionContext(safe_mode=False)

        # Create a mock execution processor for tests

        mock_execution_processor = MagicMock()

        async for output_name, output_data in block.run(
            input_data,
            credentials=llm_module.TEST_CREDENTIALS,
            graph_id="test-graph-id",
            node_id="test-node-id",
            graph_exec_id="test-exec-id",
            node_exec_id="test-node-exec-id",
            user_id="test-user-id",
            graph_version=1,
            execution_context=mock_execution_context,
            execution_processor=mock_execution_processor,
        ):
            outputs[output_name] = output_data

        # Should finish since no tool calls
        assert "finished" in outputs
        assert outputs["finished"] == "I'll help you with that."

    # Test case 3: Test with dict raw_response (some providers/tests)
    mock_response_dict = MagicMock()
    mock_response_dict.response = "Test response"
    mock_response_dict.tool_calls = None
    mock_response_dict.prompt_tokens = 25
    mock_response_dict.completion_tokens = 10
    mock_response_dict.reasoning = None
    mock_response_dict.raw_response = {
        "role": "assistant",
        "content": "Test response",
    }  # Dict format

    with patch(
        "backend.blocks.llm.llm_call",
        new_callable=AsyncMock,
        return_value=mock_response_dict,
    ), patch.object(
        SmartDecisionMakerBlock,
        "_create_tool_node_signatures",
        new_callable=AsyncMock,
        return_value=[],
    ):
        input_data = SmartDecisionMakerBlock.Input(
            prompt="Another test",
            model=llm_module.DEFAULT_LLM_MODEL,
            credentials=llm_module.TEST_CREDENTIALS_INPUT,  # type: ignore
            agent_mode_max_iterations=0,
        )

        outputs = {}
        # Create execution context

        mock_execution_context = ExecutionContext(safe_mode=False)

        # Create a mock execution processor for tests

        mock_execution_processor = MagicMock()

        async for output_name, output_data in block.run(
            input_data,
            credentials=llm_module.TEST_CREDENTIALS,
            graph_id="test-graph-id",
            node_id="test-node-id",
            graph_exec_id="test-exec-id",
            node_exec_id="test-node-exec-id",
            user_id="test-user-id",
            graph_version=1,
            execution_context=mock_execution_context,
            execution_processor=mock_execution_processor,
        ):
            outputs[output_name] = output_data

        assert "finished" in outputs
        assert outputs["finished"] == "Test response"


@pytest.mark.asyncio
async def test_smart_decision_maker_agent_mode():
    """Test that agent mode executes tools directly and loops until finished."""
    import backend.blocks.llm as llm_module
    from backend.blocks.smart_decision_maker import SmartDecisionMakerBlock

    block = SmartDecisionMakerBlock()

    # Mock tool call that requires multiple iterations
    mock_tool_call_1 = MagicMock()
    mock_tool_call_1.id = "call_1"
    mock_tool_call_1.function.name = "search_keywords"
    mock_tool_call_1.function.arguments = (
        '{"query": "test", "max_keyword_difficulty": 50}'
    )

    mock_response_1 = MagicMock()
    mock_response_1.response = None
    mock_response_1.tool_calls = [mock_tool_call_1]
    mock_response_1.prompt_tokens = 50
    mock_response_1.completion_tokens = 25
    mock_response_1.reasoning = "Using search tool"
    mock_response_1.raw_response = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "call_1", "type": "function"}],
    }

    # Final response with no tool calls (finished)
    mock_response_2 = MagicMock()
    mock_response_2.response = "Task completed successfully"
    mock_response_2.tool_calls = []
    mock_response_2.prompt_tokens = 30
    mock_response_2.completion_tokens = 15
    mock_response_2.reasoning = None
    mock_response_2.raw_response = {
        "role": "assistant",
        "content": "Task completed successfully",
    }

    # Mock the LLM call to return different responses on each iteration
    llm_call_mock = AsyncMock()
    llm_call_mock.side_effect = [mock_response_1, mock_response_2]

    # Mock tool node signatures
    mock_tool_signatures = [
        {
            "type": "function",
            "function": {
                "name": "search_keywords",
                "_sink_node_id": "test-sink-node-id",
                "_field_mapping": {},
                "parameters": {
                    "properties": {
                        "query": {"type": "string"},
                        "max_keyword_difficulty": {"type": "integer"},
                    },
                    "required": ["query", "max_keyword_difficulty"],
                },
            },
        }
    ]

    # Mock database and execution components
    mock_db_client = AsyncMock()
    mock_node = MagicMock()
    mock_node.block_id = "test-block-id"
    mock_db_client.get_node.return_value = mock_node

    # Mock upsert_execution_input to return proper NodeExecutionResult and input data
    mock_node_exec_result = MagicMock()
    mock_node_exec_result.node_exec_id = "test-tool-exec-id"
    mock_input_data = {"query": "test", "max_keyword_difficulty": 50}
    mock_db_client.upsert_execution_input.return_value = (
        mock_node_exec_result,
        mock_input_data,
    )

    # No longer need mock_execute_node since we use execution_processor.on_node_execution

    with patch("backend.blocks.llm.llm_call", llm_call_mock), patch.object(
        block, "_create_tool_node_signatures", return_value=mock_tool_signatures
    ), patch(
        "backend.blocks.smart_decision_maker.get_database_manager_async_client",
        return_value=mock_db_client,
    ), patch(
        "backend.executor.manager.async_update_node_execution_status",
        new_callable=AsyncMock,
    ), patch(
        "backend.integrations.creds_manager.IntegrationCredentialsManager"
    ):

        # Create a mock execution context

        mock_execution_context = ExecutionContext(
            safe_mode=False,
        )

        # Create a mock execution processor for agent mode tests

        mock_execution_processor = AsyncMock()
        # Configure the execution processor mock with required attributes
        mock_execution_processor.running_node_execution = defaultdict(MagicMock)
        mock_execution_processor.execution_stats = MagicMock()
        mock_execution_processor.execution_stats_lock = threading.Lock()

        # Mock the on_node_execution method to return successful stats
        mock_node_stats = MagicMock()
        mock_node_stats.error = None  # No error
        mock_execution_processor.on_node_execution = AsyncMock(
            return_value=mock_node_stats
        )

        # Mock the get_execution_outputs_by_node_exec_id method
        mock_db_client.get_execution_outputs_by_node_exec_id.return_value = {
            "result": {"status": "success", "data": "search completed"}
        }

        # Test agent mode with max_iterations = 3
        input_data = SmartDecisionMakerBlock.Input(
            prompt="Complete this task using tools",
            model=llm_module.DEFAULT_LLM_MODEL,
            credentials=llm_module.TEST_CREDENTIALS_INPUT,  # type: ignore
            agent_mode_max_iterations=3,  # Enable agent mode with 3 max iterations
        )

        outputs = {}
        async for output_name, output_data in block.run(
            input_data,
            credentials=llm_module.TEST_CREDENTIALS,
            graph_id="test-graph-id",
            node_id="test-node-id",
            graph_exec_id="test-exec-id",
            node_exec_id="test-node-exec-id",
            user_id="test-user-id",
            graph_version=1,
            execution_context=mock_execution_context,
            execution_processor=mock_execution_processor,
        ):
            outputs[output_name] = output_data

        # Verify agent mode behavior
        assert "tool_functions" in outputs  # tool_functions is yielded in both modes
        assert "finished" in outputs
        assert outputs["finished"] == "Task completed successfully"
        assert "conversations" in outputs

        # Verify the conversation includes tool responses
        conversations = outputs["conversations"]
        assert len(conversations) > 2  # Should have multiple conversation entries

        # Verify LLM was called twice (once for tool call, once for finish)
        assert llm_call_mock.call_count == 2

        # Verify tool was executed via execution processor
        assert mock_execution_processor.on_node_execution.call_count == 1


@pytest.mark.asyncio
async def test_smart_decision_maker_traditional_mode_default():
    """Test that default behavior (agent_mode_max_iterations=0) works as traditional mode."""
    import backend.blocks.llm as llm_module
    from backend.blocks.smart_decision_maker import SmartDecisionMakerBlock

    block = SmartDecisionMakerBlock()

    # Mock tool call
    mock_tool_call = MagicMock()
    mock_tool_call.function.name = "search_keywords"
    mock_tool_call.function.arguments = (
        '{"query": "test", "max_keyword_difficulty": 50}'
    )

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
                "name": "search_keywords",
                "_sink_node_id": "test-sink-node-id",
                "_field_mapping": {},
                "parameters": {
                    "properties": {
                        "query": {"type": "string"},
                        "max_keyword_difficulty": {"type": "integer"},
                    },
                    "required": ["query", "max_keyword_difficulty"],
                },
            },
        }
    ]

    with patch(
        "backend.blocks.llm.llm_call",
        new_callable=AsyncMock,
        return_value=mock_response,
    ), patch.object(
        block, "_create_tool_node_signatures", return_value=mock_tool_signatures
    ):

        # Test default behavior (traditional mode)
        input_data = SmartDecisionMakerBlock.Input(
            prompt="Test prompt",
            model=llm_module.DEFAULT_LLM_MODEL,
            credentials=llm_module.TEST_CREDENTIALS_INPUT,  # type: ignore
            agent_mode_max_iterations=0,  # Traditional mode
        )

        # Create execution context

        mock_execution_context = ExecutionContext(safe_mode=False)

        # Create a mock execution processor for tests

        mock_execution_processor = MagicMock()

        outputs = {}
        async for output_name, output_data in block.run(
            input_data,
            credentials=llm_module.TEST_CREDENTIALS,
            graph_id="test-graph-id",
            node_id="test-node-id",
            graph_exec_id="test-exec-id",
            node_exec_id="test-node-exec-id",
            user_id="test-user-id",
            graph_version=1,
            execution_context=mock_execution_context,
            execution_processor=mock_execution_processor,
        ):
            outputs[output_name] = output_data

        # Verify traditional mode behavior
        assert (
            "tool_functions" in outputs
        )  # Should yield tool_functions in traditional mode
        assert (
            "tools_^_test-sink-node-id_~_query" in outputs
        )  # Should yield individual tool parameters
        assert "tools_^_test-sink-node-id_~_max_keyword_difficulty" in outputs
        assert "conversations" in outputs


@pytest.mark.asyncio
async def test_smart_decision_maker_uses_customized_name_for_blocks():
    """Test that SmartDecisionMakerBlock uses customized_name from node metadata for tool names."""
    from unittest.mock import MagicMock

    from backend.blocks.basic import StoreValueBlock
    from backend.blocks.smart_decision_maker import SmartDecisionMakerBlock
    from backend.data.graph import Link, Node

    # Create a mock node with customized_name in metadata
    mock_node = MagicMock(spec=Node)
    mock_node.id = "test-node-id"
    mock_node.block_id = StoreValueBlock().id
    mock_node.metadata = {"customized_name": "My Custom Tool Name"}
    mock_node.block = StoreValueBlock()

    # Create a mock link
    mock_link = MagicMock(spec=Link)
    mock_link.sink_name = "input"

    # Call the function directly
    result = await SmartDecisionMakerBlock._create_block_function_signature(
        mock_node, [mock_link]
    )

    # Verify the tool name uses the customized name (cleaned up)
    assert result["type"] == "function"
    assert result["function"]["name"] == "my_custom_tool_name"  # Cleaned version
    assert result["function"]["_sink_node_id"] == "test-node-id"


@pytest.mark.asyncio
async def test_smart_decision_maker_falls_back_to_block_name():
    """Test that SmartDecisionMakerBlock falls back to block.name when no customized_name."""
    from unittest.mock import MagicMock

    from backend.blocks.basic import StoreValueBlock
    from backend.blocks.smart_decision_maker import SmartDecisionMakerBlock
    from backend.data.graph import Link, Node

    # Create a mock node without customized_name
    mock_node = MagicMock(spec=Node)
    mock_node.id = "test-node-id"
    mock_node.block_id = StoreValueBlock().id
    mock_node.metadata = {}  # No customized_name
    mock_node.block = StoreValueBlock()

    # Create a mock link
    mock_link = MagicMock(spec=Link)
    mock_link.sink_name = "input"

    # Call the function directly
    result = await SmartDecisionMakerBlock._create_block_function_signature(
        mock_node, [mock_link]
    )

    # Verify the tool name uses the block's default name
    assert result["type"] == "function"
    assert result["function"]["name"] == "storevalueblock"  # Default block name cleaned
    assert result["function"]["_sink_node_id"] == "test-node-id"


@pytest.mark.asyncio
async def test_smart_decision_maker_uses_customized_name_for_agents():
    """Test that SmartDecisionMakerBlock uses customized_name from metadata for agent nodes."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from backend.blocks.smart_decision_maker import SmartDecisionMakerBlock
    from backend.data.graph import Link, Node

    # Create a mock node with customized_name in metadata
    mock_node = MagicMock(spec=Node)
    mock_node.id = "test-agent-node-id"
    mock_node.metadata = {"customized_name": "My Custom Agent"}
    mock_node.input_default = {
        "graph_id": "test-graph-id",
        "graph_version": 1,
        "input_schema": {"properties": {"test_input": {"description": "Test input"}}},
    }

    # Create a mock link
    mock_link = MagicMock(spec=Link)
    mock_link.sink_name = "test_input"

    # Mock the database client
    mock_graph_meta = MagicMock()
    mock_graph_meta.name = "Original Agent Name"
    mock_graph_meta.description = "Agent description"

    mock_db_client = AsyncMock()
    mock_db_client.get_graph_metadata.return_value = mock_graph_meta

    with patch(
        "backend.blocks.smart_decision_maker.get_database_manager_async_client",
        return_value=mock_db_client,
    ):
        result = await SmartDecisionMakerBlock._create_agent_function_signature(
            mock_node, [mock_link]
        )

    # Verify the tool name uses the customized name (cleaned up)
    assert result["type"] == "function"
    assert result["function"]["name"] == "my_custom_agent"  # Cleaned version
    assert result["function"]["_sink_node_id"] == "test-agent-node-id"


@pytest.mark.asyncio
async def test_smart_decision_maker_agent_falls_back_to_graph_name():
    """Test that agent node falls back to graph name when no customized_name."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from backend.blocks.smart_decision_maker import SmartDecisionMakerBlock
    from backend.data.graph import Link, Node

    # Create a mock node without customized_name
    mock_node = MagicMock(spec=Node)
    mock_node.id = "test-agent-node-id"
    mock_node.metadata = {}  # No customized_name
    mock_node.input_default = {
        "graph_id": "test-graph-id",
        "graph_version": 1,
        "input_schema": {"properties": {"test_input": {"description": "Test input"}}},
    }

    # Create a mock link
    mock_link = MagicMock(spec=Link)
    mock_link.sink_name = "test_input"

    # Mock the database client
    mock_graph_meta = MagicMock()
    mock_graph_meta.name = "Original Agent Name"
    mock_graph_meta.description = "Agent description"

    mock_db_client = AsyncMock()
    mock_db_client.get_graph_metadata.return_value = mock_graph_meta

    with patch(
        "backend.blocks.smart_decision_maker.get_database_manager_async_client",
        return_value=mock_db_client,
    ):
        result = await SmartDecisionMakerBlock._create_agent_function_signature(
            mock_node, [mock_link]
        )

    # Verify the tool name uses the graph's default name
    assert result["type"] == "function"
    assert result["function"]["name"] == "original_agent_name"  # Graph name cleaned
    assert result["function"]["_sink_node_id"] == "test-agent-node-id"
