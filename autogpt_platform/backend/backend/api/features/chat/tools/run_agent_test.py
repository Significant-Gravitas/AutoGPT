import uuid

import orjson
import pytest

from ._test_data import (
    make_session,
    setup_firecrawl_test_data,
    setup_llm_test_data,
    setup_test_data,
)
from .run_agent import RunAgentTool

# This is so the formatter doesn't remove the fixture imports
setup_llm_test_data = setup_llm_test_data
setup_test_data = setup_test_data
setup_firecrawl_test_data = setup_firecrawl_test_data


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


@pytest.mark.asyncio(scope="session")
async def test_run_agent_shows_available_inputs_when_none_provided(setup_test_data):
    """Test that run_agent returns available inputs when called without inputs or use_defaults."""
    user = setup_test_data["user"]
    store_submission = setup_test_data["store_submission"]

    tool = RunAgentTool()
    agent_marketplace_id = f"{user.email.split('@')[0]}/{store_submission.slug}"
    session = make_session(user_id=user.id)

    # Execute without inputs and without use_defaults
    response = await tool.execute(
        user_id=user.id,
        session_id=str(uuid.uuid4()),
        tool_call_id=str(uuid.uuid4()),
        username_agent_slug=agent_marketplace_id,
        inputs={},
        use_defaults=False,
        session=session,
    )

    assert response is not None
    assert hasattr(response, "result")
    assert isinstance(response.result, str)
    result_data = orjson.loads(response.result)

    # Should return agent_details type showing available inputs
    assert result_data.get("type") == "agent_details"
    assert "agent" in result_data
    assert "message" in result_data
    # Message should mention inputs
    assert "inputs" in result_data["message"].lower()


@pytest.mark.asyncio(scope="session")
async def test_run_agent_with_use_defaults(setup_test_data):
    """Test that run_agent executes successfully with use_defaults=True."""
    user = setup_test_data["user"]
    graph = setup_test_data["graph"]
    store_submission = setup_test_data["store_submission"]

    tool = RunAgentTool()
    agent_marketplace_id = f"{user.email.split('@')[0]}/{store_submission.slug}"
    session = make_session(user_id=user.id)

    # Execute with use_defaults=True (no explicit inputs)
    response = await tool.execute(
        user_id=user.id,
        session_id=str(uuid.uuid4()),
        tool_call_id=str(uuid.uuid4()),
        username_agent_slug=agent_marketplace_id,
        inputs={},
        use_defaults=True,
        session=session,
    )

    assert response is not None
    assert hasattr(response, "result")
    assert isinstance(response.result, str)
    result_data = orjson.loads(response.result)

    # Should execute successfully
    assert "execution_id" in result_data
    assert result_data["graph_id"] == graph.id


@pytest.mark.asyncio(scope="session")
async def test_run_agent_missing_credentials(setup_firecrawl_test_data):
    """Test that run_agent returns setup_requirements when credentials are missing."""
    user = setup_firecrawl_test_data["user"]
    store_submission = setup_firecrawl_test_data["store_submission"]

    tool = RunAgentTool()
    agent_marketplace_id = f"{user.email.split('@')[0]}/{store_submission.slug}"
    session = make_session(user_id=user.id)

    # Execute - user doesn't have firecrawl credentials
    response = await tool.execute(
        user_id=user.id,
        session_id=str(uuid.uuid4()),
        tool_call_id=str(uuid.uuid4()),
        username_agent_slug=agent_marketplace_id,
        inputs={"url": "https://example.com"},
        session=session,
    )

    assert response is not None
    assert hasattr(response, "result")
    assert isinstance(response.result, str)
    result_data = orjson.loads(response.result)

    # Should return setup_requirements type with missing credentials
    assert result_data.get("type") == "setup_requirements"
    assert "setup_info" in result_data
    setup_info = result_data["setup_info"]
    assert "user_readiness" in setup_info
    assert setup_info["user_readiness"]["has_all_credentials"] is False
    assert len(setup_info["user_readiness"]["missing_credentials"]) > 0


@pytest.mark.asyncio(scope="session")
async def test_run_agent_invalid_slug_format(setup_test_data):
    """Test that run_agent returns error for invalid slug format (no slash)."""
    user = setup_test_data["user"]

    tool = RunAgentTool()
    session = make_session(user_id=user.id)

    # Execute with invalid slug format
    response = await tool.execute(
        user_id=user.id,
        session_id=str(uuid.uuid4()),
        tool_call_id=str(uuid.uuid4()),
        username_agent_slug="no-slash-here",
        inputs={},
        session=session,
    )

    assert response is not None
    assert hasattr(response, "result")
    assert isinstance(response.result, str)
    result_data = orjson.loads(response.result)

    # Should return error
    assert result_data.get("type") == "error"
    assert "username/agent-name" in result_data["message"]


@pytest.mark.asyncio(scope="session")
async def test_run_agent_unauthenticated():
    """Test that run_agent returns need_login for unauthenticated users."""
    tool = RunAgentTool()
    session = make_session(user_id=None)

    # Execute without user_id
    response = await tool.execute(
        user_id=None,
        session_id=str(uuid.uuid4()),
        tool_call_id=str(uuid.uuid4()),
        username_agent_slug="test/test-agent",
        inputs={},
        session=session,
    )

    assert response is not None
    assert hasattr(response, "result")
    assert isinstance(response.result, str)
    result_data = orjson.loads(response.result)

    # Base tool returns need_login type for unauthenticated users
    assert result_data.get("type") == "need_login"
    assert "sign in" in result_data["message"].lower()


@pytest.mark.asyncio(scope="session")
async def test_run_agent_schedule_without_cron(setup_test_data):
    """Test that run_agent returns error when scheduling without cron expression."""
    user = setup_test_data["user"]
    store_submission = setup_test_data["store_submission"]

    tool = RunAgentTool()
    agent_marketplace_id = f"{user.email.split('@')[0]}/{store_submission.slug}"
    session = make_session(user_id=user.id)

    # Try to schedule without cron
    response = await tool.execute(
        user_id=user.id,
        session_id=str(uuid.uuid4()),
        tool_call_id=str(uuid.uuid4()),
        username_agent_slug=agent_marketplace_id,
        inputs={"test_input": "test"},
        schedule_name="My Schedule",
        cron="",  # Empty cron
        session=session,
    )

    assert response is not None
    assert hasattr(response, "result")
    assert isinstance(response.result, str)
    result_data = orjson.loads(response.result)

    # Should return error about missing cron
    assert result_data.get("type") == "error"
    assert "cron" in result_data["message"].lower()


@pytest.mark.asyncio(scope="session")
async def test_run_agent_schedule_without_name(setup_test_data):
    """Test that run_agent returns error when scheduling without schedule_name."""
    user = setup_test_data["user"]
    store_submission = setup_test_data["store_submission"]

    tool = RunAgentTool()
    agent_marketplace_id = f"{user.email.split('@')[0]}/{store_submission.slug}"
    session = make_session(user_id=user.id)

    # Try to schedule without schedule_name
    response = await tool.execute(
        user_id=user.id,
        session_id=str(uuid.uuid4()),
        tool_call_id=str(uuid.uuid4()),
        username_agent_slug=agent_marketplace_id,
        inputs={"test_input": "test"},
        schedule_name="",  # Empty name
        cron="0 9 * * *",
        session=session,
    )

    assert response is not None
    assert hasattr(response, "result")
    assert isinstance(response.result, str)
    result_data = orjson.loads(response.result)

    # Should return error about missing schedule_name
    assert result_data.get("type") == "error"
    assert "schedule_name" in result_data["message"].lower()
