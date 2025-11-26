import uuid

import orjson
import pytest

from backend.server.v2.chat.tools._test_data import (
    make_session,
    setup_llm_test_data,
    setup_test_data,
)
from backend.server.v2.chat.tools.run_agent import RunAgentTool

# This is so the formatter doesn't remove the fixture imports
setup_llm_test_data = setup_llm_test_data
setup_test_data = setup_test_data


@pytest.mark.asyncio(scope="session")
async def test_run_agent(setup_test_data):
    """Test that the run_agent tool successfully executes an approved agent"""
    # Use test data from fixture
    user = setup_test_data["user"]
    graph = setup_test_data["graph"]
    store_submission = setup_test_data["store_submission"]

    # Create the tool instance
    tool = RunAgentTool()

    # Build the proper marketplace agent_id format: username/slug
    agent_marketplace_id = f"{user.email.split('@')[0]}/{store_submission.slug}"

    # Build the session
    session = make_session(user_id=user.id)

    # Execute the tool
    response = await tool.execute(
        user_id=user.id,
        session_id=str(uuid.uuid4()),
        tool_call_id=str(uuid.uuid4()),
        username_agent_slug=agent_marketplace_id,
        inputs={"test_input": "Hello World"},
        session=session,
    )

    # Verify the response
    assert response is not None
    assert hasattr(response, "result")
    # Parse the result JSON to verify the execution started

    assert isinstance(response.result, str)
    result_data = orjson.loads(response.result)
    assert "execution_id" in result_data
    assert "graph_id" in result_data
    assert result_data["graph_id"] == graph.id
    assert "graph_name" in result_data
    assert result_data["graph_name"] == "Test Agent"


@pytest.mark.asyncio(scope="session")
async def test_run_agent_missing_inputs(setup_test_data):
    """Test that the run_agent tool returns error when inputs are missing"""
    # Use test data from fixture
    user = setup_test_data["user"]
    store_submission = setup_test_data["store_submission"]

    # Create the tool instance
    tool = RunAgentTool()

    # Build the proper marketplace agent_id format
    agent_marketplace_id = f"{user.email.split('@')[0]}/{store_submission.slug}"

    # Build the session
    session = make_session(user_id=user.id)

    # Execute the tool without required inputs
    response = await tool.execute(
        user_id=user.id,
        session_id=str(uuid.uuid4()),
        tool_call_id=str(uuid.uuid4()),
        username_agent_slug=agent_marketplace_id,
        inputs={},  # Missing required input
        session=session,
    )

    # Verify that we get an error response
    assert response is not None
    assert hasattr(response, "result")
    # The tool should return an ErrorResponse when setup info indicates not ready

    assert isinstance(response.result, str)
    result_data = orjson.loads(response.result)
    assert "message" in result_data


@pytest.mark.asyncio(scope="session")
async def test_run_agent_invalid_agent_id(setup_test_data):
    """Test that the run_agent tool returns error for invalid agent ID"""
    # Use test data from fixture
    user = setup_test_data["user"]

    # Create the tool instance
    tool = RunAgentTool()

    # Build the session
    session = make_session(user_id=user.id)

    # Execute the tool with invalid agent ID
    response = await tool.execute(
        user_id=user.id,
        session_id=str(uuid.uuid4()),
        tool_call_id=str(uuid.uuid4()),
        username_agent_slug="invalid/agent-id",
        inputs={"test_input": "Hello World"},
        session=session,
    )

    # Verify that we get an error response
    assert response is not None
    assert hasattr(response, "result")

    assert isinstance(response.result, str)
    result_data = orjson.loads(response.result)
    assert "message" in result_data
    # Should get an error about failed setup or not found
    assert any(
        phrase in result_data["message"].lower() for phrase in ["not found", "failed"]
    )


@pytest.mark.asyncio(scope="session")
async def test_run_agent_with_llm_credentials(setup_llm_test_data):
    """Test that run_agent works with an agent requiring LLM credentials"""
    # Use test data from fixture
    user = setup_llm_test_data["user"]
    graph = setup_llm_test_data["graph"]
    store_submission = setup_llm_test_data["store_submission"]

    # Create the tool instance
    tool = RunAgentTool()

    # Build the proper marketplace agent_id format
    agent_marketplace_id = f"{user.email.split('@')[0]}/{store_submission.slug}"

    # Build the session
    session = make_session(user_id=user.id)

    # Execute the tool with a prompt for the LLM
    response = await tool.execute(
        user_id=user.id,
        session_id=str(uuid.uuid4()),
        tool_call_id=str(uuid.uuid4()),
        username_agent_slug=agent_marketplace_id,
        inputs={"user_prompt": "What is 2+2?"},
        session=session,
    )

    # Verify the response
    assert response is not None
    assert hasattr(response, "result")

    # Parse the result JSON to verify the execution started

    assert isinstance(response.result, str)
    result_data = orjson.loads(response.result)

    # Should successfully start execution since credentials are available
    assert "execution_id" in result_data
    assert "graph_id" in result_data
    assert result_data["graph_id"] == graph.id
    assert "graph_name" in result_data
    assert result_data["graph_name"] == "LLM Test Agent"
