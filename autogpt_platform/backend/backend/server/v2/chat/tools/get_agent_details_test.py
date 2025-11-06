import uuid

import orjson
import pytest

from backend.server.v2.chat.tools._test_data import (
    make_session,
    setup_llm_test_data,
    setup_test_data,
)
from backend.server.v2.chat.tools.get_agent_details import GetAgentDetailsTool

# This is so the formatter doesn't remove the fixture imports
setup_llm_test_data = setup_llm_test_data
setup_test_data = setup_test_data


@pytest.mark.asyncio(scope="session")
async def test_get_agent_details_success(setup_test_data):
    """Test successfully getting agent details from marketplace"""
    # Use test data from fixture
    user = setup_test_data["user"]
    graph = setup_test_data["graph"]
    store_submission = setup_test_data["store_submission"]

    # Create the tool instance
    tool = GetAgentDetailsTool()

    # Build the proper marketplace agent_id format: username/slug
    agent_marketplace_id = f"{user.email.split('@')[0]}/{store_submission.slug}"

    # Build session
    session = make_session()

    # Execute the tool
    response = await tool.execute(
        user_id=user.id,
        session=session,
        tool_call_id=str(uuid.uuid4()),
        username_agent_slug=agent_marketplace_id,
    )

    # Verify the response
    assert response is not None
    assert hasattr(response, "result")

    # Parse the result JSON
    assert isinstance(response.result, str)
    result_data = orjson.loads(response.result)

    # Check the basic structure
    assert "agent" in result_data
    assert "message" in result_data
    assert "graph_id" in result_data
    assert "graph_version" in result_data
    assert "user_authenticated" in result_data

    # Check agent details
    agent = result_data["agent"]
    assert agent["id"] == graph.id
    assert agent["name"] == "Test Agent"
    assert (
        agent["description"] == "A simple test agent"
    )  # Description from store submission
    assert "inputs" in agent
    assert "credentials" in agent
    assert "execution_options" in agent

    # Check execution options
    exec_options = agent["execution_options"]
    assert "manual" in exec_options
    assert "scheduled" in exec_options
    assert "webhook" in exec_options

    # Check inputs schema
    assert isinstance(agent["inputs"], dict)
    # Should have properties for the input fields
    if "properties" in agent["inputs"]:
        assert "test_input" in agent["inputs"]["properties"]


@pytest.mark.asyncio(scope="session")
async def test_get_agent_details_with_llm_credentials(setup_llm_test_data):
    """Test getting agent details for an agent that requires LLM credentials"""
    # Use test data from fixture
    user = setup_llm_test_data["user"]
    store_submission = setup_llm_test_data["store_submission"]

    # Create the tool instance
    tool = GetAgentDetailsTool()

    # Build the proper marketplace agent_id format
    agent_marketplace_id = f"{user.email.split('@')[0]}/{store_submission.slug}"

    session = make_session(user_id=user.id)

    # Execute the tool
    response = await tool.execute(
        user_id=user.id,
        session=session,
        tool_call_id=str(uuid.uuid4()),
        username_agent_slug=agent_marketplace_id,
    )

    # Verify the response
    assert response is not None
    assert hasattr(response, "result")

    # Parse the result JSON
    assert isinstance(response.result, str)
    result_data = orjson.loads(response.result)

    # Check that agent details are returned
    assert "agent" in result_data
    agent = result_data["agent"]

    # Check that credentials are listed
    assert "credentials" in agent
    credentials = agent["credentials"]

    # The LLM agent should have OpenAI credentials listed
    assert isinstance(credentials, list)

    # Check that inputs include the user_prompt
    assert "inputs" in agent
    if "properties" in agent["inputs"]:
        assert "user_prompt" in agent["inputs"]["properties"]


@pytest.mark.asyncio(scope="session")
async def test_get_agent_details_invalid_format():
    """Test error handling when agent_id is not in correct format"""
    tool = GetAgentDetailsTool()

    session = make_session()
    session.user_id = str(uuid.uuid4())

    # Execute with invalid format (no slash)
    response = await tool.execute(
        user_id=session.user_id,
        session=session,
        tool_call_id=str(uuid.uuid4()),
        username_agent_slug="invalid-format",
    )

    # Verify error response
    assert response is not None
    assert hasattr(response, "result")

    assert isinstance(response.result, str)
    result_data = orjson.loads(response.result)
    assert "message" in result_data
    assert "creator/agent-name" in result_data["message"]


@pytest.mark.asyncio(scope="session")
async def test_get_agent_details_empty_slug():
    """Test error handling when agent_id is empty"""
    tool = GetAgentDetailsTool()

    session = make_session()
    session.user_id = str(uuid.uuid4())

    # Execute with empty slug
    response = await tool.execute(
        user_id=session.user_id,
        session=session,
        tool_call_id=str(uuid.uuid4()),
        username_agent_slug="",
    )

    # Verify error response
    assert response is not None
    assert hasattr(response, "result")

    assert isinstance(response.result, str)
    result_data = orjson.loads(response.result)
    assert "message" in result_data
    assert "creator/agent-name" in result_data["message"]


@pytest.mark.asyncio(scope="session")
async def test_get_agent_details_not_found():
    """Test error handling when agent is not found in marketplace"""
    tool = GetAgentDetailsTool()

    session = make_session()
    session.user_id = str(uuid.uuid4())

    # Execute with non-existent agent
    response = await tool.execute(
        user_id=session.user_id,
        session=session,
        tool_call_id=str(uuid.uuid4()),
        username_agent_slug="nonexistent/agent",
    )

    # Verify error response
    assert response is not None
    assert hasattr(response, "result")

    assert isinstance(response.result, str)
    result_data = orjson.loads(response.result)
    assert "message" in result_data
    assert "not found" in result_data["message"].lower()


@pytest.mark.asyncio(scope="session")
async def test_get_agent_details_anonymous_user(setup_test_data):
    """Test getting agent details as an anonymous user (no user_id)"""
    # Use test data from fixture
    user = setup_test_data["user"]
    store_submission = setup_test_data["store_submission"]

    # Create the tool instance
    tool = GetAgentDetailsTool()

    # Build the proper marketplace agent_id format
    agent_marketplace_id = f"{user.email.split('@')[0]}/{store_submission.slug}"

    session = make_session()
    # session.user_id stays as None

    # Execute the tool without a user_id (anonymous)
    response = await tool.execute(
        user_id=None,
        session=session,
        tool_call_id=str(uuid.uuid4()),
        username_agent_slug=agent_marketplace_id,
    )

    # Verify the response
    assert response is not None
    assert hasattr(response, "result")

    # Parse the result JSON
    assert isinstance(response.result, str)
    result_data = orjson.loads(response.result)

    # Should still get agent details
    assert "agent" in result_data
    assert "user_authenticated" in result_data

    # User should be marked as not authenticated
    assert result_data["user_authenticated"] is False


@pytest.mark.asyncio(scope="session")
async def test_get_agent_details_authenticated_user(setup_test_data):
    """Test getting agent details as an authenticated user"""
    # Use test data from fixture
    user = setup_test_data["user"]
    store_submission = setup_test_data["store_submission"]

    # Create the tool instance
    tool = GetAgentDetailsTool()

    # Build the proper marketplace agent_id format
    agent_marketplace_id = f"{user.email.split('@')[0]}/{store_submission.slug}"

    session = make_session()
    session.user_id = user.id

    # Execute the tool with a user_id (authenticated)
    response = await tool.execute(
        user_id=user.id,
        session=session,
        tool_call_id=str(uuid.uuid4()),
        username_agent_slug=agent_marketplace_id,
    )

    # Verify the response
    assert response is not None
    assert hasattr(response, "result")

    # Parse the result JSON
    assert isinstance(response.result, str)
    result_data = orjson.loads(response.result)

    # Should get agent details
    assert "agent" in result_data
    assert "user_authenticated" in result_data

    # User should be marked as authenticated
    assert result_data["user_authenticated"] is True


@pytest.mark.asyncio(scope="session")
async def test_get_agent_details_includes_execution_options(setup_test_data):
    """Test that agent details include execution options"""
    # Use test data from fixture
    user = setup_test_data["user"]
    store_submission = setup_test_data["store_submission"]

    # Create the tool instance
    tool = GetAgentDetailsTool()

    # Build the proper marketplace agent_id format
    agent_marketplace_id = f"{user.email.split('@')[0]}/{store_submission.slug}"

    session = make_session()
    session.user_id = user.id

    # Execute the tool
    response = await tool.execute(
        user_id=user.id,
        session=session,
        tool_call_id=str(uuid.uuid4()),
        username_agent_slug=agent_marketplace_id,
    )

    # Verify the response
    assert response is not None
    assert hasattr(response, "result")

    # Parse the result JSON
    assert isinstance(response.result, str)
    result_data = orjson.loads(response.result)

    # Check execution options
    assert "agent" in result_data
    agent = result_data["agent"]
    assert "execution_options" in agent

    exec_options = agent["execution_options"]

    # These should all be boolean values
    assert isinstance(exec_options["manual"], bool)
    assert isinstance(exec_options["scheduled"], bool)
    assert isinstance(exec_options["webhook"], bool)

    # For a regular agent (no webhook), manual and scheduled should be True
    assert exec_options["manual"] is True
    assert exec_options["scheduled"] is True
    assert exec_options["webhook"] is False
