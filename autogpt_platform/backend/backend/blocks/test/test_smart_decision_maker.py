import logging

import pytest

from backend.data.model import ProviderName, User
from backend.server.model import CreateGraph
from backend.server.rest_api import AgentServer
from backend.usecases.sample import create_test_graph, create_test_user
from backend.util.test import SpinTestServer, wait_execution

logger = logging.getLogger(__name__)


async def create_graph(s: SpinTestServer, g, u: User):
    logger.info("Creating graph for user %s", u.id)
    return await s.agent_server.test_create_graph(CreateGraph(graph=g), u.id)


async def create_credentials(s: SpinTestServer, u: User):
    import backend.blocks.llm as llm

    provider = ProviderName.OPENAI
    credentials = llm.TEST_CREDENTIALS
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

    tool_functions = await SmartDecisionMakerBlock._create_function_signature(
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
    from unittest.mock import MagicMock, patch

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

    # Mock the _create_function_signature method to avoid database calls
    with patch("backend.blocks.llm.llm_call", return_value=mock_response), patch.object(
        SmartDecisionMakerBlock, "_create_function_signature", return_value=[]
    ):

        # Create test input
        input_data = SmartDecisionMakerBlock.Input(
            prompt="Should I continue with this task?",
            model=llm_module.LlmModel.GPT4O,
            credentials=llm_module.TEST_CREDENTIALS_INPUT,  # type: ignore
        )

        # Execute the block
        outputs = {}
        async for output_name, output_data in block.run(
            input_data,
            credentials=llm_module.TEST_CREDENTIALS,
            graph_id="test-graph-id",
            node_id="test-node-id",
            graph_exec_id="test-exec-id",
            node_exec_id="test-node-exec-id",
            user_id="test-user-id",
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
    from unittest.mock import MagicMock, patch

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
        "backend.blocks.llm.llm_call", return_value=mock_response_with_typo
    ) as mock_llm_call, patch.object(
        SmartDecisionMakerBlock,
        "_create_function_signature",
        return_value=mock_tool_functions,
    ):

        input_data = SmartDecisionMakerBlock.Input(
            prompt="Search for keywords",
            model=llm_module.LlmModel.GPT4O,
            credentials=llm_module.TEST_CREDENTIALS_INPUT,  # type: ignore
            retry=2,  # Set retry to 2 for testing
        )

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
            ):
                outputs[output_name] = output_data

        # Verify error message contains details about the typo
        error_msg = str(exc_info.value)
        assert "Tool call validation failed" in error_msg
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
        "backend.blocks.llm.llm_call", return_value=mock_response_missing_required
    ), patch.object(
        SmartDecisionMakerBlock,
        "_create_function_signature",
        return_value=mock_tool_functions,
    ):

        input_data = SmartDecisionMakerBlock.Input(
            prompt="Search for keywords",
            model=llm_module.LlmModel.GPT4O,
            credentials=llm_module.TEST_CREDENTIALS_INPUT,  # type: ignore
        )

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
        "backend.blocks.llm.llm_call", return_value=mock_response_valid
    ), patch.object(
        SmartDecisionMakerBlock,
        "_create_function_signature",
        return_value=mock_tool_functions,
    ):

        input_data = SmartDecisionMakerBlock.Input(
            prompt="Search for keywords",
            model=llm_module.LlmModel.GPT4O,
            credentials=llm_module.TEST_CREDENTIALS_INPUT,  # type: ignore
        )

        # Should succeed - optional parameter missing is OK
        outputs = {}
        async for output_name, output_data in block.run(
            input_data,
            credentials=llm_module.TEST_CREDENTIALS,
            graph_id="test-graph-id",
            node_id="test-node-id",
            graph_exec_id="test-exec-id",
            node_exec_id="test-node-exec-id",
            user_id="test-user-id",
        ):
            outputs[output_name] = output_data

        # Verify tool outputs were generated correctly
        assert "tools_^_search_keywords_~_query" in outputs
        assert outputs["tools_^_search_keywords_~_query"] == "test"
        assert "tools_^_search_keywords_~_max_keyword_difficulty" in outputs
        assert outputs["tools_^_search_keywords_~_max_keyword_difficulty"] == 50
        # Optional parameter should be None when not provided
        assert "tools_^_search_keywords_~_optional_param" in outputs
        assert outputs["tools_^_search_keywords_~_optional_param"] is None

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
        "backend.blocks.llm.llm_call", return_value=mock_response_all_params
    ), patch.object(
        SmartDecisionMakerBlock,
        "_create_function_signature",
        return_value=mock_tool_functions,
    ):

        input_data = SmartDecisionMakerBlock.Input(
            prompt="Search for keywords",
            model=llm_module.LlmModel.GPT4O,
            credentials=llm_module.TEST_CREDENTIALS_INPUT,  # type: ignore
        )

        # Should succeed with all parameters
        outputs = {}
        async for output_name, output_data in block.run(
            input_data,
            credentials=llm_module.TEST_CREDENTIALS,
            graph_id="test-graph-id",
            node_id="test-node-id",
            graph_exec_id="test-exec-id",
            node_exec_id="test-node-exec-id",
            user_id="test-user-id",
        ):
            outputs[output_name] = output_data

        # Verify all tool outputs were generated correctly
        assert outputs["tools_^_search_keywords_~_query"] == "test"
        assert outputs["tools_^_search_keywords_~_max_keyword_difficulty"] == 50
        assert outputs["tools_^_search_keywords_~_optional_param"] == "custom_value"
