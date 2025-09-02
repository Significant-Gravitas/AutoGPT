"""Unit tests for tool execution functions."""

import json
from typing import Any, Dict

import pytest

from backend.server.v2.chat.tools import (
    execute_find_agent,
    execute_get_agent_details,
    execute_setup_agent,
    tools,
)


class TestToolDefinitions:
    """Test tool definitions structure."""

    def test_tools_list_structure(self):
        """Test that tools list is properly structured."""
        assert isinstance(tools, list)
        assert len(tools) > 0

        for tool in tools:
            assert "type" in tool
            assert tool["type"] == "function"
            assert "function" in tool
            assert "name" in tool["function"]
            assert "description" in tool["function"]
            assert "parameters" in tool["function"]

    def test_find_agent_tool_definition(self):
        """Test find_agent tool definition."""
        find_agent_tool = next(
            (t for t in tools if t["function"]["name"] == "find_agent"), None
        )
        assert find_agent_tool is not None

        func = find_agent_tool["function"]
        assert func["description"]
        assert func["parameters"]["type"] == "object"
        assert "properties" in func["parameters"]
        assert "search_query" in func["parameters"]["properties"]

    def test_get_agent_details_tool_definition(self):
        """Test get_agent_details tool definition."""
        get_details_tool = next(
            (t for t in tools if t["function"]["name"] == "get_agent_details"), None
        )
        assert get_details_tool is not None

        func = get_details_tool["function"]
        assert func["description"]
        assert func["parameters"]["type"] == "object"
        assert "properties" in func["parameters"]
        assert "agent_id" in func["parameters"]["properties"]
        assert "agent_version" in func["parameters"]["properties"]
        assert "agent_id" in func["parameters"]["required"]

    def test_setup_agent_tool_definition(self):
        """Test setup_agent tool definition."""
        setup_tool = next(
            (t for t in tools if t["function"]["name"] == "setup_agent"), None
        )
        assert setup_tool is not None

        func = setup_tool["function"]
        assert func["parameters"]["type"] == "object"
        assert "properties" in func["parameters"]
        assert "graph_id" in func["parameters"]["properties"]
        assert "name" in func["parameters"]["properties"]
        assert "cron" in func["parameters"]["properties"]


class TestExecuteFindAgent:
    """Test execute_find_agent function."""

    @pytest.mark.asyncio
    async def test_find_agent_with_query(self):
        """Test finding agents with a search query."""
        parameters = {"search_query": "data analysis"}
        result = await execute_find_agent(parameters, "user-123", "session-456")

        assert isinstance(result, str)
        assert "Found" in result
        assert "agents matching" in result
        assert "data analysis" in result

        # Parse the JSON part
        json_start = result.find("[")
        if json_start != -1:
            json_data = json.loads(result[json_start:])
            assert isinstance(json_data, list)
            assert len(json_data) > 0

            for agent in json_data:
                assert "id" in agent
                assert "name" in agent
                assert "description" in agent
                assert "version" in agent

    @pytest.mark.asyncio
    async def test_find_agent_without_query(self):
        """Test finding agents without search query."""
        parameters: Dict[str, Any] = {}
        result = await execute_find_agent(parameters, "user-123", "session-456")

        assert isinstance(result, str)
        assert "Found" in result
        # Should still return results even with empty query

    @pytest.mark.asyncio
    async def test_find_agent_returns_mock_data(self):
        """Test that find_agent returns consistent mock data structure."""
        parameters = {"search_query": "test"}
        result = await execute_find_agent(parameters, "user-123", "session-456")

        # Extract JSON from result
        json_start = result.find("[")
        json_data = json.loads(result[json_start:])

        # Verify mock data structure
        assert len(json_data) == 2  # Mock returns 2 agents

        agent = json_data[0]
        assert agent["id"] == "agent-123"
        assert agent["name"] == "Data Analysis Agent"
        assert "rating" in agent
        assert "downloads" in agent


class TestExecuteGetAgentDetails:
    """Test execute_get_agent_details function."""

    @pytest.mark.asyncio
    async def test_get_agent_details_with_id(self):
        """Test getting agent details with agent ID."""
        parameters = {"agent_id": "agent-789"}
        result = await execute_get_agent_details(parameters, "user-123", "session-456")

        assert isinstance(result, str)
        assert "Agent Details" in result
        assert "agent-789" in result

        # Parse JSON part
        json_start = result.find("{")
        if json_start != -1:
            json_data = json.loads(result[json_start:])
            assert json_data["id"] == "agent-789"
            assert "name" in json_data
            assert "description" in json_data
            assert "credentials_required" in json_data
            assert "inputs" in json_data
            assert "capabilities" in json_data

    @pytest.mark.asyncio
    async def test_get_agent_details_with_version(self):
        """Test getting agent details with specific version."""
        parameters = {"agent_id": "agent-789", "agent_version": "2.0.0"}
        result = await execute_get_agent_details(parameters, "user-123", "session-456")

        assert isinstance(result, str)
        assert "2.0.0" in result

        json_start = result.find("{")
        json_data = json.loads(result[json_start:])
        assert json_data["version"] == "2.0.0"

    @pytest.mark.asyncio
    async def test_get_agent_details_without_version(self):
        """Test getting agent details defaults to latest version."""
        parameters = {"agent_id": "agent-789"}
        result = await execute_get_agent_details(parameters, "user-123", "session-456")

        json_start = result.find("{")
        json_data = json.loads(result[json_start:])
        assert json_data["version"] == "latest"

    @pytest.mark.asyncio
    async def test_get_agent_details_credentials_structure(self):
        """Test that credentials required structure is correct."""
        parameters = {"agent_id": "test-agent"}
        result = await execute_get_agent_details(parameters, "user-123", "session-456")

        json_start = result.find("{")
        json_data = json.loads(result[json_start:])

        assert isinstance(json_data["credentials_required"], list)
        if len(json_data["credentials_required"]) > 0:
            cred = json_data["credentials_required"][0]
            assert "type" in cred
            assert "provider" in cred
            assert "description" in cred

    @pytest.mark.asyncio
    async def test_get_agent_details_inputs_structure(self):
        """Test that inputs structure is correct."""
        parameters = {"agent_id": "test-agent"}
        result = await execute_get_agent_details(parameters, "user-123", "session-456")

        json_start = result.find("{")
        json_data = json.loads(result[json_start:])

        assert isinstance(json_data["inputs"], list)
        if len(json_data["inputs"]) > 0:
            input_spec = json_data["inputs"][0]
            assert "name" in input_spec
            assert "type" in input_spec
            assert "description" in input_spec
            assert "required" in input_spec


class TestExecuteSetupAgent:
    """Test execute_setup_agent function."""

    @pytest.mark.asyncio
    async def test_setup_agent_with_all_parameters(self):
        """Test setting up agent with all parameters."""
        parameters = {
            "graph_id": "graph-123",
            "graph_version": 3,
            "name": "Daily Report Agent",
            "cron": "0 9 * * *",
            "inputs": {"source": "database", "format": "pdf"},
        }
        result = await execute_setup_agent(parameters, "user-123", "session-456")

        assert isinstance(result, str)
        assert "Agent Setup Complete" in result
        assert "success" in result.lower()

        # Parse JSON
        json_start = result.find("{")
        json_data = json.loads(result[json_start:])

        assert json_data["status"] == "success"
        assert json_data["graph_id"] == "graph-123"
        assert json_data["graph_version"] == 3
        assert json_data["name"] == "Daily Report Agent"
        assert json_data["cron"] == "0 9 * * *"
        assert json_data["inputs"] == {"source": "database", "format": "pdf"}
        assert "schedule_id" in json_data
        assert "next_run" in json_data
        assert "message" in json_data

    @pytest.mark.asyncio
    async def test_setup_agent_with_minimal_parameters(self):
        """Test setting up agent with minimal parameters."""
        parameters: Dict[str, Any] = {"graph_id": "graph-456"}
        result = await execute_setup_agent(parameters, "user-123", "session-456")

        json_start = result.find("{")
        json_data = json.loads(result[json_start:])

        assert json_data["status"] == "success"
        assert json_data["graph_id"] == "graph-456"
        assert json_data["graph_version"] == "latest"  # Default
        assert json_data["name"] == "Unnamed Schedule"  # Default
        assert json_data["cron"] == ""  # Default empty
        assert json_data["inputs"] == {}  # Default empty

    @pytest.mark.asyncio
    async def test_setup_agent_without_version(self):
        """Test that setup defaults to latest version when not specified."""
        parameters = {
            "graph_id": "graph-789",
            "name": "Test Agent",
            "cron": "*/5 * * * *",
        }
        result = await execute_setup_agent(parameters, "user-123", "session-456")

        json_start = result.find("{")
        json_data = json.loads(result[json_start:])

        assert json_data["graph_version"] == "latest"

    @pytest.mark.asyncio
    async def test_setup_agent_success_message(self):
        """Test that setup returns proper success message."""
        parameters = {
            "graph_id": "graph-999",
            "name": "Hourly Check",
            "cron": "0 * * * *",
        }
        result = await execute_setup_agent(parameters, "user-123", "session-456")

        json_start = result.find("{")
        json_data = json.loads(result[json_start:])

        assert "Successfully scheduled" in json_data["message"]
        assert "Hourly Check" in json_data["message"]
        assert "0 * * * *" in json_data["message"]

    @pytest.mark.asyncio
    async def test_setup_agent_returns_schedule_id(self):
        """Test that setup returns a schedule ID."""
        parameters = {"graph_id": "test-graph"}
        result = await execute_setup_agent(parameters, "user-123", "session-456")

        json_start = result.find("{")
        json_data = json.loads(result[json_start:])

        assert "schedule_id" in json_data
        assert json_data["schedule_id"]  # Should not be empty
        assert isinstance(json_data["schedule_id"], str)


class TestToolIntegration:
    """Integration tests for tools."""

    @pytest.mark.asyncio
    async def test_all_tools_have_executors(self):
        """Test that all defined tools have corresponding executor functions."""
        import backend.server.v2.chat.tools as tools_module

        for tool in tools:
            tool_name = tool["function"]["name"]
            executor_name = f"execute_{tool_name}"

            # Check that executor function exists
            assert hasattr(
                tools_module, executor_name
            ), f"Tool '{tool_name}' missing executor function '{executor_name}'"

            # Check that it's callable
            executor = getattr(tools_module, executor_name)
            assert callable(executor), f"'{executor_name}' is not callable"

    @pytest.mark.asyncio
    async def test_tool_executors_return_strings(self):
        """Test that all tool executors return strings."""
        test_params: Dict[str, Any] = {"test": "value"}

        # Test each executor
        result1 = await execute_find_agent(test_params, "user", "session")
        assert isinstance(result1, str)

        result2 = await execute_get_agent_details(
            {"agent_id": "test"}, "user", "session"
        )
        assert isinstance(result2, str)

        result3 = await execute_setup_agent(test_params, "user", "session")
        assert isinstance(result3, str)

    @pytest.mark.asyncio
    async def test_tool_executors_handle_empty_params(self):
        """Test that tool executors handle empty parameters gracefully."""
        empty_params: Dict[str, Any] = {}

        # None should raise exceptions
        result1 = await execute_find_agent(empty_params, "user", "session")
        assert isinstance(result1, str)

        result2 = await execute_get_agent_details(empty_params, "user", "session")
        assert isinstance(result2, str)

        result3 = await execute_setup_agent(empty_params, "user", "session")
        assert isinstance(result3, str)
