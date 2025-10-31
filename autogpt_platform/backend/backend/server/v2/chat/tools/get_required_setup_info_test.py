import uuid

import orjson
import pytest

from backend.server.v2.chat.tools._test_data import (
    setup_firecrawl_test_data,
    setup_llm_test_data,
    setup_test_data,
)
from backend.server.v2.chat.tools.get_required_setup_info import (
    GetRequiredSetupInfoTool,
)

# This is so the formatter doesn't remove the fixture imports
setup_llm_test_data = setup_llm_test_data
setup_test_data = setup_test_data
setup_firecrawl_test_data = setup_firecrawl_test_data


@pytest.mark.asyncio(scope="session")
async def test_get_required_setup_info_success(setup_test_data):
    """Test successfully getting setup info for a simple agent"""
    # Use test data from fixture
    user = setup_test_data["user"]
    graph = setup_test_data["graph"]
    store_submission = setup_test_data["store_submission"]

    # Create the tool instance
    tool = GetRequiredSetupInfoTool()

    # Build the proper marketplace agent_id format: username/slug
    agent_marketplace_id = f"{user.email.split('@')[0]}/{store_submission.slug}"

    # Execute the tool
    response = await tool.execute(
        user_id=user.id,
        session_id=str(uuid.uuid4()),
        tool_call_id=str(uuid.uuid4()),
        username_agent_slug=agent_marketplace_id,
        inputs={"test_input": "Hello World"},
    )

    # Verify the response
    assert response is not None
    assert hasattr(response, "result")

    # Parse the result JSON
    assert isinstance(response.result, str)
    result_data = orjson.loads(response.result)

    # Check the basic structure
    assert "setup_info" in result_data
    setup_info = result_data["setup_info"]

    # Check agent info
    assert "agent_id" in setup_info
    assert setup_info["agent_id"] == graph.id
    assert "agent_name" in setup_info
    assert setup_info["agent_name"] == "Test Agent"

    # Check requirements
    assert "requirements" in setup_info
    requirements = setup_info["requirements"]
    assert "credentials" in requirements
    assert "inputs" in requirements
    assert "execution_modes" in requirements

    # Simple agent should have no credentials required
    assert isinstance(requirements["credentials"], list)
    assert len(requirements["credentials"]) == 0

    # Check inputs format
    assert isinstance(requirements["inputs"], list)
    if len(requirements["inputs"]) > 0:
        first_input = requirements["inputs"][0]
        assert "name" in first_input
        assert "title" in first_input
        assert "type" in first_input

    # Check execution modes
    assert isinstance(requirements["execution_modes"], list)
    assert "manual" in requirements["execution_modes"]
    assert "scheduled" in requirements["execution_modes"]

    # Check user readiness
    assert "user_readiness" in setup_info
    user_readiness = setup_info["user_readiness"]
    assert "has_all_credentials" in user_readiness
    assert "ready_to_run" in user_readiness
    # Simple agent with inputs provided should be ready
    assert user_readiness["ready_to_run"] is True


@pytest.mark.asyncio(scope="session")
async def test_get_required_setup_info_missing_credentials(setup_firecrawl_test_data):
    """Test getting setup info for an agent requiring missing credentials"""
    # Use test data from fixture
    user = setup_firecrawl_test_data["user"]
    store_submission = setup_firecrawl_test_data["store_submission"]

    # Create the tool instance
    tool = GetRequiredSetupInfoTool()

    # Build the proper marketplace agent_id format
    agent_marketplace_id = f"{user.email.split('@')[0]}/{store_submission.slug}"

    # Execute the tool
    response = await tool.execute(
        user_id=user.id,
        session_id=str(uuid.uuid4()),
        tool_call_id=str(uuid.uuid4()),
        username_agent_slug=agent_marketplace_id,
        inputs={"url": "https://example.com"},
    )

    # Verify the response
    assert response is not None
    assert hasattr(response, "result")

    # Parse the result JSON
    assert isinstance(response.result, str)
    result_data = orjson.loads(response.result)

    # Check setup info
    assert "setup_info" in result_data
    setup_info = result_data["setup_info"]

    # Check requirements
    requirements = setup_info["requirements"]

    # Should have Firecrawl credentials required
    assert "credentials" in requirements
    assert isinstance(requirements["credentials"], list)
    assert len(requirements["credentials"]) > 0

    # Check the credential requirement
    firecrawl_cred = requirements["credentials"][0]
    assert "provider" in firecrawl_cred
    assert firecrawl_cred["provider"] == "firecrawl"
    assert "type" in firecrawl_cred
    assert firecrawl_cred["type"] == "api_key"

    # Check user readiness - should NOT be ready since credentials are missing
    user_readiness = setup_info["user_readiness"]
    assert user_readiness["has_all_credentials"] is False
    assert user_readiness["ready_to_run"] is False

    # Check missing credentials
    assert "missing_credentials" in user_readiness
    assert isinstance(user_readiness["missing_credentials"], dict)
    assert len(user_readiness["missing_credentials"]) > 0


@pytest.mark.asyncio(scope="session")
async def test_get_required_setup_info_with_available_credentials(setup_llm_test_data):
    """Test getting setup info when user has required credentials"""
    # Use test data from fixture (includes OpenAI credentials)
    user = setup_llm_test_data["user"]
    store_submission = setup_llm_test_data["store_submission"]

    # Create the tool instance
    tool = GetRequiredSetupInfoTool()

    # Build the proper marketplace agent_id format
    agent_marketplace_id = f"{user.email.split('@')[0]}/{store_submission.slug}"

    # Execute the tool
    response = await tool.execute(
        user_id=user.id,
        session_id=str(uuid.uuid4()),
        tool_call_id=str(uuid.uuid4()),
        username_agent_slug=agent_marketplace_id,
        inputs={"user_prompt": "What is 2+2?"},
    )

    # Verify the response
    assert response is not None
    assert hasattr(response, "result")

    # Parse the result JSON
    assert isinstance(response.result, str)
    result_data = orjson.loads(response.result)

    # Check setup info
    setup_info = result_data["setup_info"]

    # Check user readiness - should be ready since credentials are available
    user_readiness = setup_info["user_readiness"]
    assert user_readiness["has_all_credentials"] is True
    assert user_readiness["ready_to_run"] is True

    # Missing credentials should be empty
    assert "missing_credentials" in user_readiness
    assert len(user_readiness["missing_credentials"]) == 0


@pytest.mark.asyncio(scope="session")
async def test_get_required_setup_info_missing_inputs(setup_test_data):
    """Test getting setup info when required inputs are not provided"""
    # Use test data from fixture
    user = setup_test_data["user"]
    store_submission = setup_test_data["store_submission"]

    # Create the tool instance
    tool = GetRequiredSetupInfoTool()

    # Build the proper marketplace agent_id format
    agent_marketplace_id = f"{user.email.split('@')[0]}/{store_submission.slug}"

    # Execute the tool WITHOUT providing inputs
    response = await tool.execute(
        user_id=user.id,
        session_id=str(uuid.uuid4()),
        tool_call_id=str(uuid.uuid4()),
        username_agent_slug=agent_marketplace_id,
        inputs={},  # Empty inputs
    )

    # Verify the response
    assert response is not None
    assert hasattr(response, "result")

    # Parse the result JSON
    assert isinstance(response.result, str)
    result_data = orjson.loads(response.result)

    # Check setup info
    setup_info = result_data["setup_info"]

    # Check requirements
    requirements = setup_info["requirements"]
    assert "inputs" in requirements
    assert isinstance(requirements["inputs"], list)

    # User readiness depends on whether inputs are required or optional
    user_readiness = setup_info["user_readiness"]
    assert "ready_to_run" in user_readiness


@pytest.mark.asyncio(scope="session")
async def test_get_required_setup_info_invalid_agent():
    """Test getting setup info for a non-existent agent"""
    # Create the tool instance
    tool = GetRequiredSetupInfoTool()

    # Execute with invalid agent ID
    response = await tool.execute(
        user_id=str(uuid.uuid4()),
        session_id=str(uuid.uuid4()),
        tool_call_id=str(uuid.uuid4()),
        username_agent_slug="invalid/agent",
        inputs={},
    )

    # Verify error response
    assert response is not None
    assert hasattr(response, "result")

    assert isinstance(response.result, str)
    result_data = orjson.loads(response.result)
    assert "message" in result_data
    # Should indicate failure or not found
    assert any(
        phrase in result_data["message"].lower()
        for phrase in ["not found", "failed", "error"]
    )


@pytest.mark.asyncio(scope="session")
async def test_get_required_setup_info_graph_metadata(setup_test_data):
    """Test that setup info includes graph metadata"""
    # Use test data from fixture
    user = setup_test_data["user"]
    graph = setup_test_data["graph"]
    store_submission = setup_test_data["store_submission"]

    # Create the tool instance
    tool = GetRequiredSetupInfoTool()

    # Build the proper marketplace agent_id format
    agent_marketplace_id = f"{user.email.split('@')[0]}/{store_submission.slug}"

    # Execute the tool
    response = await tool.execute(
        user_id=user.id,
        session_id=str(uuid.uuid4()),
        tool_call_id=str(uuid.uuid4()),
        username_agent_slug=agent_marketplace_id,
        inputs={"test_input": "test"},
    )

    # Verify the response
    assert response is not None
    assert hasattr(response, "result")

    # Parse the result JSON
    assert isinstance(response.result, str)
    result_data = orjson.loads(response.result)

    # Check that graph_id and graph_version are included
    assert "graph_id" in result_data
    assert result_data["graph_id"] == graph.id
    assert "graph_version" in result_data
    assert result_data["graph_version"] == graph.version


@pytest.mark.asyncio(scope="session")
async def test_get_required_setup_info_inputs_structure(setup_test_data):
    """Test that inputs are properly structured as a list"""
    # Use test data from fixture
    user = setup_test_data["user"]
    store_submission = setup_test_data["store_submission"]

    # Create the tool instance
    tool = GetRequiredSetupInfoTool()

    # Build the proper marketplace agent_id format
    agent_marketplace_id = f"{user.email.split('@')[0]}/{store_submission.slug}"

    # Execute the tool
    response = await tool.execute(
        user_id=user.id,
        session_id=str(uuid.uuid4()),
        tool_call_id=str(uuid.uuid4()),
        username_agent_slug=agent_marketplace_id,
        inputs={},
    )

    # Verify the response
    assert response is not None
    assert hasattr(response, "result")

    # Parse the result JSON
    assert isinstance(response.result, str)
    result_data = orjson.loads(response.result)

    # Check inputs structure
    setup_info = result_data["setup_info"]
    requirements = setup_info["requirements"]

    # Inputs should be a list
    assert isinstance(requirements["inputs"], list)

    # Each input should have proper structure
    for input_field in requirements["inputs"]:
        assert isinstance(input_field, dict)
        assert "name" in input_field
        assert "title" in input_field
        assert "type" in input_field
        assert "description" in input_field
        assert "required" in input_field
        assert isinstance(input_field["required"], bool)


@pytest.mark.asyncio(scope="session")
async def test_get_required_setup_info_execution_modes_structure(setup_test_data):
    """Test that execution_modes are properly structured as a list"""
    # Use test data from fixture
    user = setup_test_data["user"]
    store_submission = setup_test_data["store_submission"]

    # Create the tool instance
    tool = GetRequiredSetupInfoTool()

    # Build the proper marketplace agent_id format
    agent_marketplace_id = f"{user.email.split('@')[0]}/{store_submission.slug}"

    # Execute the tool
    response = await tool.execute(
        user_id=user.id,
        session_id=str(uuid.uuid4()),
        tool_call_id=str(uuid.uuid4()),
        username_agent_slug=agent_marketplace_id,
        inputs={},
    )

    # Verify the response
    assert response is not None
    assert hasattr(response, "result")

    # Parse the result JSON
    assert isinstance(response.result, str)
    result_data = orjson.loads(response.result)

    # Check execution modes structure
    setup_info = result_data["setup_info"]
    requirements = setup_info["requirements"]

    # execution_modes should be a list of strings
    assert isinstance(requirements["execution_modes"], list)
    for mode in requirements["execution_modes"]:
        assert isinstance(mode, str)
        assert mode in ["manual", "scheduled", "webhook"]
