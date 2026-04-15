import uuid
from unittest.mock import AsyncMock, patch

import orjson
import pytest

from backend.executor.utils import is_credential_validation_error_message
from backend.util.exceptions import GraphValidationError

from ._test_data import (
    make_session,
    setup_firecrawl_test_data,
    setup_llm_test_data,
    setup_test_data,
)
from .models import SetupRequirementsResponse
from .run_agent import RunAgentTool

# This is so the formatter doesn't remove the fixture imports
setup_llm_test_data = setup_llm_test_data
setup_test_data = setup_test_data
setup_firecrawl_test_data = setup_firecrawl_test_data


@pytest.fixture(scope="session", autouse=True)
def mock_embedding_functions():
    """Mock embedding functions for all tests to avoid database/API dependencies."""
    with patch(
        "backend.api.features.store.db.ensure_embedding",
        new_callable=AsyncMock,
        return_value=True,
    ):
        yield


@pytest.mark.asyncio(loop_scope="session")
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
        dry_run=False,
        session=session,
    )

    # Verify the response
    assert response is not None
    assert hasattr(response, "output")
    # Parse the result JSON to verify the execution started

    assert isinstance(response.output, str)
    result_data = orjson.loads(response.output)
    assert "execution_id" in result_data
    assert "graph_id" in result_data
    assert result_data["graph_id"] == graph.id
    assert "graph_name" in result_data
    assert result_data["graph_name"] == "Test Agent"


@pytest.mark.asyncio(loop_scope="session")
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
        dry_run=False,
        session=session,
    )

    # Verify that we get an error response
    assert response is not None
    assert hasattr(response, "output")
    # The tool should return an ErrorResponse when setup info indicates not ready

    assert isinstance(response.output, str)
    result_data = orjson.loads(response.output)
    assert "message" in result_data


@pytest.mark.asyncio(loop_scope="session")
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
        dry_run=False,
        session=session,
    )

    # Verify that we get an error response
    assert response is not None
    assert hasattr(response, "output")

    assert isinstance(response.output, str)
    result_data = orjson.loads(response.output)
    assert "message" in result_data
    # Should get an error about failed setup or not found
    assert any(
        phrase in result_data["message"].lower() for phrase in ["not found", "failed"]
    )


@pytest.mark.asyncio(loop_scope="session")
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
        dry_run=False,
        session=session,
    )

    # Verify the response
    assert response is not None
    assert hasattr(response, "output")

    # Parse the result JSON to verify the execution started

    assert isinstance(response.output, str)
    result_data = orjson.loads(response.output)

    # Should successfully start execution since credentials are available
    assert "execution_id" in result_data
    assert "graph_id" in result_data
    assert result_data["graph_id"] == graph.id
    assert "graph_name" in result_data
    assert result_data["graph_name"] == "LLM Test Agent"


@pytest.mark.asyncio(loop_scope="session")
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
        dry_run=False,
        session=session,
    )

    assert response is not None
    assert hasattr(response, "output")
    assert isinstance(response.output, str)
    result_data = orjson.loads(response.output)

    # Should return agent_details type showing available inputs
    assert result_data.get("type") == "agent_details"
    assert "agent" in result_data
    assert "message" in result_data
    # Message should mention inputs
    assert "inputs" in result_data["message"].lower()


@pytest.mark.asyncio(loop_scope="session")
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
        dry_run=False,
        session=session,
    )

    assert response is not None
    assert hasattr(response, "output")
    assert isinstance(response.output, str)
    result_data = orjson.loads(response.output)

    # Should execute successfully
    assert "execution_id" in result_data
    assert result_data["graph_id"] == graph.id


@pytest.mark.asyncio(loop_scope="session")
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
        dry_run=False,
        session=session,
    )

    assert response is not None
    assert hasattr(response, "output")
    assert isinstance(response.output, str)
    result_data = orjson.loads(response.output)

    # Should return setup_requirements type with missing credentials
    assert result_data.get("type") == "setup_requirements"
    assert "setup_info" in result_data
    setup_info = result_data["setup_info"]
    assert "user_readiness" in setup_info
    assert setup_info["user_readiness"]["has_all_credentials"] is False
    assert len(setup_info["user_readiness"]["missing_credentials"]) > 0


@pytest.mark.asyncio(loop_scope="session")
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
        dry_run=False,
        session=session,
    )

    assert response is not None
    assert hasattr(response, "output")
    assert isinstance(response.output, str)
    result_data = orjson.loads(response.output)

    # Should return error
    assert result_data.get("type") == "error"
    assert "username/agent-name" in result_data["message"]


@pytest.mark.asyncio(loop_scope="session")
async def test_run_agent_unauthenticated():
    """Test that run_agent returns need_login for unauthenticated users."""
    tool = RunAgentTool()
    # Session has a user_id (session owner), but we test tool execution without user_id
    session = make_session(user_id="test-session-owner")

    # Execute without user_id to test unauthenticated behavior
    response = await tool.execute(
        user_id=None,
        session_id=str(uuid.uuid4()),
        tool_call_id=str(uuid.uuid4()),
        username_agent_slug="test/test-agent",
        inputs={},
        dry_run=False,
        session=session,
    )

    assert response is not None
    assert hasattr(response, "output")
    assert isinstance(response.output, str)
    result_data = orjson.loads(response.output)

    # Base tool returns need_login type for unauthenticated users
    assert result_data.get("type") == "need_login"
    assert "sign in" in result_data["message"].lower()


@pytest.mark.asyncio(loop_scope="session")
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
        dry_run=False,
        session=session,
    )

    assert response is not None
    assert hasattr(response, "output")
    assert isinstance(response.output, str)
    result_data = orjson.loads(response.output)

    # Should return error about missing cron
    assert result_data.get("type") == "error"
    assert "cron" in result_data["message"].lower()


@pytest.mark.asyncio(loop_scope="session")
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
        dry_run=False,
        session=session,
    )

    assert response is not None
    assert hasattr(response, "output")
    assert isinstance(response.output, str)
    result_data = orjson.loads(response.output)

    # Should return error about missing schedule_name
    assert result_data.get("type") == "error"
    assert "schedule_name" in result_data["message"].lower()


@pytest.mark.asyncio(loop_scope="session")
async def test_run_agent_rejects_unknown_input_fields(setup_test_data):
    """Test that run_agent returns input_validation_error for unknown input fields."""
    user = setup_test_data["user"]
    store_submission = setup_test_data["store_submission"]

    tool = RunAgentTool()
    agent_marketplace_id = f"{user.email.split('@')[0]}/{store_submission.slug}"
    session = make_session(user_id=user.id)

    # Execute with unknown input field names
    response = await tool.execute(
        user_id=user.id,
        session_id=str(uuid.uuid4()),
        tool_call_id=str(uuid.uuid4()),
        username_agent_slug=agent_marketplace_id,
        inputs={
            "unknown_field": "some value",
            "another_unknown": "another value",
        },
        dry_run=False,
        session=session,
    )

    assert response is not None
    assert hasattr(response, "output")
    assert isinstance(response.output, str)
    result_data = orjson.loads(response.output)

    # Should return input_validation_error type with unrecognized fields
    assert result_data.get("type") == "input_validation_error"
    assert "unrecognized_fields" in result_data
    assert set(result_data["unrecognized_fields"]) == {
        "another_unknown",
        "unknown_field",
    }
    assert "inputs" in result_data  # Contains the valid schema
    assert "Agent was not executed" in result_data["message"]


# ---------------------------------------------------------------------------
# Credential-race-condition handling
#
# ``_check_prerequisites`` already catches the common "missing creds" case
# at the top of ``_execute``, but the scheduler / executor re-validates and
# can raise ``GraphValidationError`` if creds were deleted between the
# prereq check and the actual call.  The tool turns these credential
# errors back into the inline ``SetupRequirementsResponse`` so the user
# still gets the credential setup card instead of a generic error.
# ---------------------------------------------------------------------------


def test_is_credential_validation_error_message_recognises_credential_strings():
    """Shared helper should match all credential error strings emitted by
    ``backend.executor.utils._validate_node_input_credentials``."""
    assert is_credential_validation_error_message("These credentials are required")
    assert is_credential_validation_error_message("THESE CREDENTIALS ARE REQUIRED")
    assert is_credential_validation_error_message("Invalid credentials: not found")
    assert is_credential_validation_error_message("Credentials not available: github")
    assert is_credential_validation_error_message("Unknown credentials #abc-123")


def test_is_credential_validation_error_message_rejects_non_credential_strings():
    """Shared helper should ignore unrelated graph validation messages."""
    assert not is_credential_validation_error_message("Input field 'url' is required")
    assert not is_credential_validation_error_message("Block configuration invalid")
    assert not is_credential_validation_error_message("")
    assert not is_credential_validation_error_message("credentials are fine")


@pytest.mark.asyncio(loop_scope="session")
async def test_build_setup_requirements_from_credential_validation_error(
    setup_firecrawl_test_data,
):
    """When the scheduler raises a credential-flavoured GraphValidationError,
    the helper should rebuild the inline setup card from the graph schema."""
    graph = setup_firecrawl_test_data["graph"]
    tool = RunAgentTool()

    # Construct an error in the same shape the executor produces.
    error = GraphValidationError(
        message="Graph is invalid",
        node_errors={"some-node-id": {"credentials": "These credentials are required"}},
    )

    # Race path: all credential fields shown as missing.
    response = tool._build_setup_requirements_from_validation_error(
        graph=graph,
        error=error,
        session_id="test-session",
    )

    assert isinstance(response, SetupRequirementsResponse)
    assert response.graph_id == graph.id
    assert response.graph_version == graph.version
    assert response.setup_info.user_readiness.has_all_credentials is False
    assert response.setup_info.user_readiness.ready_to_run is False
    # The firecrawl fixture defines exactly one credential field (firecrawl
    # API key).  Pin the count so fixture drift is caught immediately.
    missing_credentials = response.setup_info.user_readiness.missing_credentials
    assert len(missing_credentials) == 1, (
        f"Expected exactly 1 credential from the firecrawl fixture, "
        f"got {len(missing_credentials)}: {list(missing_credentials.keys())}"
    )
    assert "credentials" in response.message.lower()
    # Message must be action-neutral: this helper is shared by the run
    # path and the schedule path, so hardcoding "scheduling again" would
    # mislead users on the run path.
    assert "scheduling again" not in response.message.lower()


@pytest.mark.asyncio(loop_scope="session")
async def test_build_setup_requirements_shows_all_creds_missing_in_race(
    setup_firecrawl_test_data,
):
    """In the race scenario (prereq passed → creds deleted → executor raised),
    the helper must show ALL credential fields as missing so the user knows
    which accounts need to be reconnected — not an empty missing_credentials map."""
    graph = setup_firecrawl_test_data["graph"]
    tool = RunAgentTool()

    error = GraphValidationError(
        message="Graph is invalid",
        node_errors={"some-node-id": {"credentials": "These credentials are required"}},
    )

    response = tool._build_setup_requirements_from_validation_error(
        graph=graph,
        error=error,
        session_id="test-session",
    )

    assert isinstance(response, SetupRequirementsResponse)
    # missing_credentials and requirements["credentials"] must both be non-empty
    # and share the same field keys (both come from build_missing_credentials_from_graph).
    missing = response.setup_info.user_readiness.missing_credentials
    requirements_creds = response.setup_info.requirements["credentials"]
    assert len(missing) > 0
    assert set(missing.keys()) == {c["id"] for c in requirements_creds}


@pytest.mark.asyncio(loop_scope="session")
async def test_build_setup_requirements_returns_none_for_empty_node_errors(
    setup_firecrawl_test_data,
):
    """Empty node_errors={} should fall through (helper returns None) because
    there are no messages to classify as credential-related."""
    graph = setup_firecrawl_test_data["graph"]
    tool = RunAgentTool()

    error = GraphValidationError(
        message="Graph is invalid",
        node_errors={},
    )

    response = tool._build_setup_requirements_from_validation_error(
        graph=graph,
        error=error,
        session_id="test-session",
    )

    assert response is None


@pytest.mark.asyncio(loop_scope="session")
async def test_build_setup_requirements_returns_none_for_non_credential_error(
    setup_firecrawl_test_data,
):
    """Non-credential validation errors should fall through to the plain
    ErrorResponse path (helper returns None)."""
    graph = setup_firecrawl_test_data["graph"]
    tool = RunAgentTool()

    error = GraphValidationError(
        message="Graph is invalid",
        node_errors={"some-node-id": {"url": "Input field 'url' is required"}},
    )

    response = tool._build_setup_requirements_from_validation_error(
        graph=graph,
        error=error,
        session_id="test-session",
    )

    assert response is None


@pytest.mark.asyncio(loop_scope="session")
async def test_build_setup_requirements_returns_none_for_mixed_errors(
    setup_firecrawl_test_data,
):
    """Mixed credential + structural errors must fall through to the plain
    ErrorResponse path so structural errors are not hidden from the user."""
    graph = setup_firecrawl_test_data["graph"]
    tool = RunAgentTool()

    error = GraphValidationError(
        message="Graph is invalid",
        node_errors={
            "node-a": {"credentials": "These credentials are required"},
            "node-b": {"url": "Input field 'url' is required"},
        },
    )

    response = tool._build_setup_requirements_from_validation_error(
        graph=graph,
        error=error,
        session_id="test-session",
    )

    assert response is None


@pytest.mark.asyncio(loop_scope="session")
async def test_run_agent_schedule_credential_race_returns_setup_card(
    setup_test_data,
):
    """End-to-end: if the scheduler raises a credential GraphValidationError
    after _check_prerequisites passed, the user should still see the
    inline credentials-setup card (not a generic error)."""
    user = setup_test_data["user"]
    store_submission = setup_test_data["store_submission"]

    tool = RunAgentTool()
    agent_marketplace_id = f"{user.email.split('@')[0]}/{store_submission.slug}"
    session = make_session(user_id=user.id)

    fake_scheduler = AsyncMock()
    fake_scheduler.add_execution_schedule.side_effect = GraphValidationError(
        message="Graph is invalid",
        node_errors={"some-node-id": {"credentials": "These credentials are required"}},
    )

    with patch(
        "backend.copilot.tools.run_agent.get_scheduler_client",
        return_value=fake_scheduler,
    ):
        response = await tool.execute(
            user_id=user.id,
            session_id=str(uuid.uuid4()),
            tool_call_id=str(uuid.uuid4()),
            username_agent_slug=agent_marketplace_id,
            inputs={"test_input": "value"},
            schedule_name="My Schedule",
            cron="0 9 * * *",
            dry_run=False,
            session=session,
        )

    assert response is not None
    assert isinstance(response.output, str)
    result_data = orjson.loads(response.output)

    # Should surface the inline credential card, NOT a generic error or a
    # link redirecting to the Builder.
    assert result_data.get("type") == "setup_requirements"
    assert "setup_info" in result_data
    assert result_data["setup_info"]["user_readiness"]["ready_to_run"] is False
    # Verify that missing_credentials is present (may be empty for graphs
    # where the DB-stored credential schema doesn't surface input-embedded
    # credentials — the important thing is that the card renders instead of
    # a generic error or Builder redirect).
    assert "missing_credentials" in result_data["setup_info"]["user_readiness"]


@pytest.mark.asyncio(loop_scope="session")
async def test_run_agent_schedule_structural_error_returns_error_response(
    setup_test_data,
):
    """End-to-end: if the scheduler raises a GraphValidationError with purely
    structural (non-credential) errors after _check_prerequisites passed, the
    tool must return an ErrorResponse with error='graph_validation_failed' —
    not a setup_requirements card."""
    user = setup_test_data["user"]
    store_submission = setup_test_data["store_submission"]

    tool = RunAgentTool()
    agent_marketplace_id = f"{user.email.split('@')[0]}/{store_submission.slug}"
    session = make_session(user_id=user.id)

    fake_scheduler = AsyncMock()
    fake_scheduler.add_execution_schedule.side_effect = GraphValidationError(
        message="Graph is invalid",
        node_errors={"some-node-id": {"url": "Input field 'url' is required"}},
    )

    with patch(
        "backend.copilot.tools.run_agent.get_scheduler_client",
        return_value=fake_scheduler,
    ):
        response = await tool.execute(
            user_id=user.id,
            session_id=str(uuid.uuid4()),
            tool_call_id=str(uuid.uuid4()),
            username_agent_slug=agent_marketplace_id,
            inputs={"test_input": "value"},
            schedule_name="My Schedule",
            cron="0 9 * * *",
            dry_run=False,
            session=session,
        )

    assert response is not None
    assert isinstance(response.output, str)
    result_data = orjson.loads(response.output)

    # Structural errors must fall through to the plain error path — the
    # user should see the validation error, not the credential setup card.
    assert result_data.get("error") == "graph_validation_failed"
    assert result_data.get("type") != "setup_requirements"


@pytest.mark.asyncio(loop_scope="session")
async def test_run_agent_execution_credential_race_returns_setup_card(
    setup_test_data,
):
    """End-to-end: if the executor raises a credential GraphValidationError
    after _check_prerequisites passed, the user should still see the
    inline credentials-setup card (not a generic error)."""
    user = setup_test_data["user"]
    store_submission = setup_test_data["store_submission"]

    tool = RunAgentTool()
    agent_marketplace_id = f"{user.email.split('@')[0]}/{store_submission.slug}"
    session = make_session(user_id=user.id)

    with patch(
        "backend.copilot.tools.run_agent.execution_utils.add_graph_execution",
        new_callable=AsyncMock,
        side_effect=GraphValidationError(
            message="Graph is invalid",
            node_errors={
                "some-node-id": {"credentials": "These credentials are required"}
            },
        ),
    ):
        response = await tool.execute(
            user_id=user.id,
            session_id=str(uuid.uuid4()),
            tool_call_id=str(uuid.uuid4()),
            username_agent_slug=agent_marketplace_id,
            inputs={"test_input": "value"},
            dry_run=False,
            session=session,
        )

    assert response is not None
    assert isinstance(response.output, str)
    result_data = orjson.loads(response.output)

    # Should surface the inline credential card, NOT a generic error or a
    # link redirecting to the Builder.
    assert result_data.get("type") == "setup_requirements"
    assert "setup_info" in result_data
    assert result_data["setup_info"]["user_readiness"]["ready_to_run"] is False


@pytest.mark.asyncio(loop_scope="session")
async def test_run_agent_execution_structural_error_returns_error_response(
    setup_test_data,
):
    """End-to-end: if the executor raises a GraphValidationError with purely
    structural (non-credential) errors after _check_prerequisites passed, the
    tool must return an ErrorResponse with error='graph_validation_failed' —
    not a setup_requirements card and not a silent swallow."""
    user = setup_test_data["user"]
    store_submission = setup_test_data["store_submission"]

    tool = RunAgentTool()
    agent_marketplace_id = f"{user.email.split('@')[0]}/{store_submission.slug}"
    session = make_session(user_id=user.id)

    with patch(
        "backend.copilot.tools.run_agent.execution_utils.add_graph_execution",
        new_callable=AsyncMock,
        side_effect=GraphValidationError(
            message="Graph is invalid",
            node_errors={
                "some-node-id": {"url": "Input field 'url' is required"},
            },
        ),
    ):
        response = await tool.execute(
            user_id=user.id,
            session_id=str(uuid.uuid4()),
            tool_call_id=str(uuid.uuid4()),
            username_agent_slug=agent_marketplace_id,
            inputs={"test_input": "value"},
            dry_run=False,
            session=session,
        )

    assert response is not None
    assert isinstance(response.output, str)
    result_data = orjson.loads(response.output)

    # Structural errors must fall through to the plain error path — the
    # user should see the validation error, not the credential setup card.
    assert result_data.get("error") == "graph_validation_failed"
    assert result_data.get("type") != "setup_requirements"
