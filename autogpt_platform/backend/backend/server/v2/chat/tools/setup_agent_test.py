import uuid

import orjson
import pytest

from backend.server.v2.chat.tools._test_data import (
    make_session,
    setup_llm_test_data,
    setup_test_data,
)
from backend.server.v2.chat.tools.setup_agent import SetupAgentTool
from backend.util.clients import get_scheduler_client

# This is so the formatter doesn't remove the fixture imports
setup_llm_test_data = setup_llm_test_data
setup_test_data = setup_test_data


@pytest.mark.asyncio(scope="session")
async def test_setup_agent_missing_cron(setup_test_data):
    """Test error when cron is missing for schedule setup"""
    # Use test data from fixture
    user = setup_test_data["user"]
    store_submission = setup_test_data["store_submission"]

    # Create the tool instance
    tool = SetupAgentTool()

    # Build the session
    session = make_session(user_id=user.id)

    # Build the proper marketplace agent_id format
    agent_marketplace_id = f"{user.email.split('@')[0]}/{store_submission.slug}"

    # Execute without cron
    response = await tool.execute(
        user_id=user.id,
        session=session,
        tool_call_id=str(uuid.uuid4()),
        username_agent_slug=agent_marketplace_id,
        setup_type="schedule",
        inputs={"test_input": "Hello World"},
        # Missing: cron and name
    )

    # Verify error response
    assert response is not None
    assert hasattr(response, "result")

    assert isinstance(response.result, str)
    result_data = orjson.loads(response.result)
    assert "message" in result_data
    assert (
        "cron" in result_data["message"].lower()
        or "name" in result_data["message"].lower()
    )


@pytest.mark.asyncio(scope="session")
async def test_setup_agent_webhook_not_supported(setup_test_data):
    """Test error when webhook setup is attempted"""
    # Use test data from fixture
    user = setup_test_data["user"]
    store_submission = setup_test_data["store_submission"]

    # Create the tool instance
    tool = SetupAgentTool()

    # Build the session
    session = make_session(user_id=user.id)

    # Build the proper marketplace agent_id format
    agent_marketplace_id = f"{user.email.split('@')[0]}/{store_submission.slug}"

    # Execute with webhook setup_type
    response = await tool.execute(
        user_id=user.id,
        session=session,
        tool_call_id=str(uuid.uuid4()),
        username_agent_slug=agent_marketplace_id,
        setup_type="webhook",
        inputs={"test_input": "Hello World"},
    )

    # Verify error response
    assert response is not None
    assert hasattr(response, "result")

    assert isinstance(response.result, str)
    result_data = orjson.loads(response.result)
    assert "message" in result_data
    message_lower = result_data["message"].lower()
    assert "schedule" in message_lower and "supported" in message_lower


@pytest.mark.asyncio(scope="session")
@pytest.mark.skip(reason="Requires scheduler service to be running")
async def test_setup_agent_schedule_success(setup_test_data):
    """Test successfully setting up an agent with a schedule"""
    # Use test data from fixture
    user = setup_test_data["user"]
    store_submission = setup_test_data["store_submission"]

    # Create the tool instance
    tool = SetupAgentTool()

    # Build the session
    session = make_session(user_id=user.id)

    # Build the proper marketplace agent_id format
    agent_marketplace_id = f"{user.email.split('@')[0]}/{store_submission.slug}"

    # Execute with schedule setup
    response = await tool.execute(
        user_id=user.id,
        session=session,
        tool_call_id=str(uuid.uuid4()),
        username_agent_slug=agent_marketplace_id,
        setup_type="schedule",
        name="Test Schedule",
        description="Test schedule description",
        cron="0 9 * * *",  # Daily at 9am
        timezone="UTC",
        inputs={"test_input": "Hello World"},
    )

    # Verify the response
    assert response is not None
    assert hasattr(response, "result")

    # Parse the result JSON
    assert isinstance(response.result, str)
    result_data = orjson.loads(response.result)

    # Check for execution started
    assert "message" in result_data
    assert "execution_id" in result_data
    assert "graph_id" in result_data
    assert "graph_name" in result_data


@pytest.mark.asyncio(scope="session")
@pytest.mark.skip(reason="Requires scheduler service to be running")
async def test_setup_agent_with_credentials(setup_llm_test_data):
    """Test setting up an agent that requires credentials"""
    # Use test data from fixture (includes OpenAI credentials)
    user = setup_llm_test_data["user"]
    store_submission = setup_llm_test_data["store_submission"]

    # Create the tool instance
    tool = SetupAgentTool()

    # Build the session
    session = make_session(user_id=user.id)

    # Build the proper marketplace agent_id format
    agent_marketplace_id = f"{user.email.split('@')[0]}/{store_submission.slug}"

    # Execute with schedule setup
    response = await tool.execute(
        user_id=user.id,
        session=session,
        tool_call_id=str(uuid.uuid4()),
        username_agent_slug=agent_marketplace_id,
        setup_type="schedule",
        name="LLM Schedule",
        description="LLM schedule with credentials",
        cron="*/30 * * * *",  # Every 30 minutes
        timezone="America/New_York",
        inputs={"user_prompt": "What is 2+2?"},
    )

    # Verify the response
    assert response is not None
    assert hasattr(response, "result")

    # Parse the result JSON
    assert isinstance(response.result, str)
    result_data = orjson.loads(response.result)

    # Should succeed since user has OpenAI credentials
    assert "execution_id" in result_data
    assert "graph_id" in result_data


@pytest.mark.asyncio(scope="session")
async def test_setup_agent_invalid_agent(setup_test_data):
    """Test error when agent doesn't exist"""
    # Use test data from fixture
    user = setup_test_data["user"]

    # Create the tool instance
    tool = SetupAgentTool()

    # Build the session
    session = make_session(user_id=user.id)

    # Execute with non-existent agent
    response = await tool.execute(
        user_id=user.id,
        session=session,
        tool_call_id=str(uuid.uuid4()),
        username_agent_slug="nonexistent/agent",
        setup_type="schedule",
        name="Test Schedule",
        cron="0 9 * * *",
        inputs={},
    )

    # Verify error response
    assert response is not None
    assert hasattr(response, "result")

    assert isinstance(response.result, str)
    result_data = orjson.loads(response.result)
    assert "message" in result_data
    # Should fail to find the agent
    assert any(
        phrase in result_data["message"].lower()
        for phrase in ["not found", "failed", "error"]
    )


@pytest.mark.asyncio(scope="session")
@pytest.mark.skip(reason="Requires scheduler service to be running")
async def test_setup_agent_schedule_created_in_scheduler(setup_test_data):
    """Test that the schedule is actually created in the scheduler service"""
    # Use test data from fixture
    user = setup_test_data["user"]
    graph = setup_test_data["graph"]
    store_submission = setup_test_data["store_submission"]

    # Create the tool instance
    tool = SetupAgentTool()

    # Build the session
    session = make_session(user_id=user.id)

    # Build the proper marketplace agent_id format
    agent_marketplace_id = f"{user.email.split('@')[0]}/{store_submission.slug}"

    # Create a unique schedule name to identify this test
    schedule_name = f"Test Schedule {uuid.uuid4()}"

    # Execute with schedule setup
    response = await tool.execute(
        user_id=user.id,
        session=session,
        tool_call_id=str(uuid.uuid4()),
        username_agent_slug=agent_marketplace_id,
        setup_type="schedule",
        name=schedule_name,
        description="Test schedule to verify credentials",
        cron="0 0 * * *",  # Daily at midnight
        timezone="UTC",
        inputs={"test_input": "Scheduled execution"},
    )

    # Verify the response
    assert response is not None
    assert isinstance(response.result, str)
    result_data = orjson.loads(response.result)
    assert "execution_id" in result_data

    # Now verify the schedule was created in the scheduler service
    scheduler = get_scheduler_client()
    schedules = await scheduler.get_execution_schedules(graph.id, user.id)

    # Find our schedule
    our_schedule = None
    for schedule in schedules:
        if schedule.name == schedule_name:
            our_schedule = schedule
            break

    assert (
        our_schedule is not None
    ), f"Schedule '{schedule_name}' not found in scheduler"
    assert our_schedule.cron == "0 0 * * *"
    assert our_schedule.graph_id == graph.id

    # Clean up: delete the schedule
    await scheduler.delete_schedule(our_schedule.id, user_id=user.id)


@pytest.mark.asyncio(scope="session")
@pytest.mark.skip(reason="Requires scheduler service to be running")
async def test_setup_agent_schedule_with_credentials_triggered(setup_llm_test_data):
    """Test that credentials are properly passed when a schedule is triggered"""
    # Use test data from fixture (includes OpenAI credentials)
    user = setup_llm_test_data["user"]
    graph = setup_llm_test_data["graph"]
    store_submission = setup_llm_test_data["store_submission"]

    # Create the tool instance
    tool = SetupAgentTool()

    # Build the session
    session = make_session(user_id=user.id)

    # Build the proper marketplace agent_id format
    agent_marketplace_id = f"{user.email.split('@')[0]}/{store_submission.slug}"

    # Create a unique schedule name
    schedule_name = f"LLM Test Schedule {uuid.uuid4()}"

    # Execute with schedule setup
    response = await tool.execute(
        user_id=user.id,
        session=session,
        tool_call_id=str(uuid.uuid4()),
        username_agent_slug=agent_marketplace_id,
        setup_type="schedule",
        name=schedule_name,
        description="Test LLM schedule with credentials",
        cron="* * * * *",  # Every minute (for testing)
        timezone="UTC",
        inputs={"user_prompt": "Test prompt for credentials"},
    )

    # Verify the response
    assert response is not None
    assert isinstance(response.result, str)
    result_data = orjson.loads(response.result)
    assert "execution_id" in result_data

    # Get the schedule from the scheduler
    scheduler = get_scheduler_client()
    schedules = await scheduler.get_execution_schedules(graph.id, user.id)

    # Find our schedule
    our_schedule = None
    for schedule in schedules:
        if schedule.name == schedule_name:
            our_schedule = schedule
            break

    assert our_schedule is not None, f"Schedule '{schedule_name}' not found"

    # Verify the schedule has the correct input data
    assert our_schedule.input_data is not None
    assert "user_prompt" in our_schedule.input_data
    assert our_schedule.input_data["user_prompt"] == "Test prompt for credentials"

    # Verify credentials are stored in the schedule
    # The credentials should be stored as input_credentials
    assert our_schedule.input_credentials is not None

    # The credentials should contain the OpenAI provider credential
    # Note: The exact structure depends on how credentials are serialized
    # We're checking that credentials data exists and has the right provider
    if our_schedule.input_credentials:
        # Convert to dict if needed
        creds_dict = (
            our_schedule.input_credentials
            if isinstance(our_schedule.input_credentials, dict)
            else {}
        )

        # Check if any credential has openai provider
        has_openai_cred = False
        for cred_key, cred_value in creds_dict.items():
            if isinstance(cred_value, dict):
                if cred_value.get("provider") == "openai":
                    has_openai_cred = True
                    # Verify the credential has the expected structure
                    assert "id" in cred_value or "api_key" in cred_value
                    break

        # If we have LLM block, we should have stored credentials
        assert has_openai_cred, "OpenAI credentials not found in schedule"

    # Clean up: delete the schedule
    await scheduler.delete_schedule(our_schedule.id, user_id=user.id)


@pytest.mark.asyncio(scope="session")
@pytest.mark.skip(reason="Requires scheduler service to be running")
async def test_setup_agent_creates_library_agent(setup_test_data):
    """Test that setup creates a library agent for the user"""
    # Use test data from fixture
    user = setup_test_data["user"]
    graph = setup_test_data["graph"]
    store_submission = setup_test_data["store_submission"]

    # Create the tool instance
    tool = SetupAgentTool()

    # Build the session
    session = make_session(user_id=user.id)

    # Build the proper marketplace agent_id format
    agent_marketplace_id = f"{user.email.split('@')[0]}/{store_submission.slug}"

    # Execute with schedule setup
    response = await tool.execute(
        user_id=user.id,
        session=session,
        tool_call_id=str(uuid.uuid4()),
        username_agent_slug=agent_marketplace_id,
        setup_type="schedule",
        name="Library Test Schedule",
        cron="0 12 * * *",  # Daily at noon
        inputs={"test_input": "Library test"},
    )

    # Verify the response
    assert response is not None
    assert isinstance(response.result, str)
    result_data = orjson.loads(response.result)
    assert "graph_id" in result_data
    assert result_data["graph_id"] == graph.id

    # Verify library agent was created
    from backend.server.v2.library import db as library_db

    library_agent = await library_db.get_library_agent_by_graph_id(
        graph_id=graph.id, user_id=user.id
    )
    assert library_agent is not None
    assert library_agent.graph_id == graph.id
    assert library_agent.name == "Test Agent"
