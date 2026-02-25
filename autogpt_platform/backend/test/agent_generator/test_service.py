"""
Tests for the Agent Generator external service client.

This test suite verifies the external Agent Generator service integration,
including service detection, async polling, and error handling.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from backend.copilot.tools.agent_generator import service


class TestServiceConfiguration:
    """Test service configuration detection."""

    def setup_method(self):
        """Reset settings singleton before each test."""
        service._settings = None
        service._client = None

    def test_external_service_not_configured_when_host_empty(self):
        """Test that external service is not configured when host is empty."""
        mock_settings = MagicMock()
        mock_settings.config.agentgenerator_host = ""
        mock_settings.config.agentgenerator_use_dummy = False

        with patch.object(service, "_get_settings", return_value=mock_settings):
            assert service.is_external_service_configured() is False

    def test_external_service_configured_when_host_set(self):
        """Test that external service is configured when host is set."""
        mock_settings = MagicMock()
        mock_settings.config.agentgenerator_host = "agent-generator.local"

        with patch.object(service, "_get_settings", return_value=mock_settings):
            assert service.is_external_service_configured() is True

    def test_get_base_url(self):
        """Test base URL construction."""
        mock_settings = MagicMock()
        mock_settings.config.agentgenerator_host = "agent-generator.local"
        mock_settings.config.agentgenerator_port = 8000

        with patch.object(service, "_get_settings", return_value=mock_settings):
            url = service._get_base_url()
            assert url == "http://agent-generator.local:8000"


class TestSubmitAndPoll:
    """Test the _submit_and_poll helper that handles async job polling."""

    def setup_method(self):
        service._settings = None
        service._client = None

    @pytest.mark.asyncio
    async def test_successful_submit_and_poll(self):
        """Test normal submit -> poll -> completed flow."""
        submit_resp = MagicMock()
        submit_resp.json.return_value = {"job_id": "job-123", "status": "accepted"}
        submit_resp.raise_for_status = MagicMock()

        poll_resp = MagicMock()
        poll_resp.json.return_value = {
            "job_id": "job-123",
            "status": "completed",
            "result": {"type": "instructions", "steps": ["Step 1"]},
        }
        poll_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = submit_resp
        mock_client.get.return_value = poll_resp

        with (
            patch.object(service, "_get_client", return_value=mock_client),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await service._submit_and_poll("/api/test", {"key": "value"})

        assert result == {"type": "instructions", "steps": ["Step 1"]}
        mock_client.post.assert_called_once_with("/api/test", json={"key": "value"})
        mock_client.get.assert_called_once_with("/api/jobs/job-123")

    @pytest.mark.asyncio
    async def test_poll_returns_failed_job(self):
        """Test submit -> poll -> failed flow."""
        submit_resp = MagicMock()
        submit_resp.json.return_value = {"job_id": "job-456", "status": "accepted"}
        submit_resp.raise_for_status = MagicMock()

        poll_resp = MagicMock()
        poll_resp.json.return_value = {
            "job_id": "job-456",
            "status": "failed",
            "error": "Generation failed",
        }
        poll_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = submit_resp
        mock_client.get.return_value = poll_resp

        with (
            patch.object(service, "_get_client", return_value=mock_client),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await service._submit_and_poll("/api/test", {})

        assert result["type"] == "error"
        assert result["error_type"] == "job_failed"
        assert "Generation failed" in result["error"]

    @pytest.mark.asyncio
    async def test_submit_http_error(self):
        """Test HTTP error during job submission."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.HTTPStatusError(
            "Server error", request=MagicMock(), response=mock_response
        )

        with patch.object(service, "_get_client", return_value=mock_client):
            result = await service._submit_and_poll("/api/test", {})

        assert result["type"] == "error"
        assert result["error_type"] == "http_error"

    @pytest.mark.asyncio
    async def test_submit_connection_error(self):
        """Test connection error during job submission."""
        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.RequestError("Connection failed")

        with patch.object(service, "_get_client", return_value=mock_client):
            result = await service._submit_and_poll("/api/test", {})

        assert result["type"] == "error"
        assert result["error_type"] == "connection_error"

    @pytest.mark.asyncio
    async def test_no_job_id_in_submit_response(self):
        """Test submit response missing job_id."""
        submit_resp = MagicMock()
        submit_resp.json.return_value = {"status": "accepted"}  # no job_id
        submit_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = submit_resp

        with patch.object(service, "_get_client", return_value=mock_client):
            result = await service._submit_and_poll("/api/test", {})

        assert result["type"] == "error"
        assert result["error_type"] == "invalid_response"

    @pytest.mark.asyncio
    async def test_poll_retries_on_transient_network_error(self):
        """Test that transient network errors during polling are retried."""
        submit_resp = MagicMock()
        submit_resp.json.return_value = {"job_id": "job-789"}
        submit_resp.raise_for_status = MagicMock()

        ok_poll_resp = MagicMock()
        ok_poll_resp.json.return_value = {
            "job_id": "job-789",
            "status": "completed",
            "result": {"data": "ok"},
        }
        ok_poll_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = submit_resp
        # First poll fails with transient error, second succeeds
        mock_client.get.side_effect = [
            httpx.RequestError("transient"),
            ok_poll_resp,
        ]

        with (
            patch.object(service, "_get_client", return_value=mock_client),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await service._submit_and_poll("/api/test", {})

        assert result == {"data": "ok"}
        assert mock_client.get.call_count == 2

    @pytest.mark.asyncio
    async def test_poll_returns_404_for_expired_job(self):
        """Test that 404 during polling returns job_not_found error."""
        submit_resp = MagicMock()
        submit_resp.json.return_value = {"job_id": "job-expired"}
        submit_resp.raise_for_status = MagicMock()

        mock_404_response = MagicMock()
        mock_404_response.status_code = 404

        mock_client = AsyncMock()
        mock_client.post.return_value = submit_resp
        mock_client.get.side_effect = httpx.HTTPStatusError(
            "Not Found", request=MagicMock(), response=mock_404_response
        )

        with (
            patch.object(service, "_get_client", return_value=mock_client),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await service._submit_and_poll("/api/test", {})

        assert result["type"] == "error"
        assert result["error_type"] == "job_not_found"

    @pytest.mark.asyncio
    async def test_poll_timeout(self):
        """Test that polling times out after MAX_POLL_TIME_SECONDS."""
        submit_resp = MagicMock()
        submit_resp.json.return_value = {"job_id": "job-slow"}
        submit_resp.raise_for_status = MagicMock()

        running_resp = MagicMock()
        running_resp.json.return_value = {"job_id": "job-slow", "status": "running"}
        running_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = submit_resp
        mock_client.get.return_value = running_resp

        # Simulate time passing: first call returns 0.0 (start), then jumps past limit
        monotonic_values = iter([0.0, 0.0, 100.0])

        with (
            patch.object(service, "_get_client", return_value=mock_client),
            patch.object(service, "MAX_POLL_TIME_SECONDS", 50.0),
            patch.object(service, "POLL_INTERVAL_SECONDS", 0.01),
            patch("asyncio.sleep", new_callable=AsyncMock),
            patch("backend.copilot.tools.agent_generator.service.time") as mock_time,
        ):
            mock_time.monotonic.side_effect = monotonic_values
            result = await service._submit_and_poll("/api/test", {})

        assert result["type"] == "error"
        assert result["error_type"] == "timeout"

    @pytest.mark.asyncio
    async def test_poll_gives_up_after_consecutive_transient_errors(self):
        """Test that polling gives up after MAX_CONSECUTIVE_POLL_ERRORS."""
        submit_resp = MagicMock()
        submit_resp.json.return_value = {"job_id": "job-flaky"}
        submit_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = submit_resp
        mock_client.get.side_effect = httpx.RequestError("network down")

        # Ensure monotonic always returns 0 so timeout doesn't kick in
        with (
            patch.object(service, "_get_client", return_value=mock_client),
            patch.object(service, "MAX_POLL_TIME_SECONDS", 9999.0),
            patch.object(service, "POLL_INTERVAL_SECONDS", 0.01),
            patch("asyncio.sleep", new_callable=AsyncMock),
            patch("backend.copilot.tools.agent_generator.service.time") as mock_time,
        ):
            mock_time.monotonic.return_value = 0.0
            result = await service._submit_and_poll("/api/test", {})

        assert result["type"] == "error"
        assert result["error_type"] == "poll_error"
        assert mock_client.get.call_count == service.MAX_CONSECUTIVE_POLL_ERRORS


class TestDecomposeGoalExternal:
    """Test decompose_goal_external function."""

    def setup_method(self):
        """Reset client singleton before each test."""
        service._settings = None
        service._client = None

    @pytest.mark.asyncio
    async def test_decompose_goal_returns_instructions(self):
        """Test successful decomposition returning instructions."""
        with (
            patch.object(service, "_is_dummy_mode", return_value=False),
            patch.object(
                service, "_submit_and_poll", new_callable=AsyncMock
            ) as mock_poll,
        ):
            mock_poll.return_value = {
                "type": "instructions",
                "steps": ["Step 1", "Step 2"],
            }
            result = await service.decompose_goal_external("Build a chatbot")

        assert result == {"type": "instructions", "steps": ["Step 1", "Step 2"]}
        mock_poll.assert_called_once_with(
            "/api/decompose-description",
            {"description": "Build a chatbot"},
        )

    @pytest.mark.asyncio
    async def test_decompose_goal_returns_clarifying_questions(self):
        """Test decomposition returning clarifying questions."""
        with (
            patch.object(service, "_is_dummy_mode", return_value=False),
            patch.object(
                service, "_submit_and_poll", new_callable=AsyncMock
            ) as mock_poll,
        ):
            mock_poll.return_value = {
                "type": "clarifying_questions",
                "questions": ["What platform?", "What language?"],
            }
            result = await service.decompose_goal_external("Build something")

        assert result == {
            "type": "clarifying_questions",
            "questions": ["What platform?", "What language?"],
        }

    @pytest.mark.asyncio
    async def test_decompose_goal_with_context(self):
        """Test decomposition with additional context enriched into description."""
        with (
            patch.object(service, "_is_dummy_mode", return_value=False),
            patch.object(
                service, "_submit_and_poll", new_callable=AsyncMock
            ) as mock_poll,
        ):
            mock_poll.return_value = {"type": "instructions", "steps": ["Step 1"]}
            await service.decompose_goal_external(
                "Build a chatbot", context="Use Python"
            )

        expected_description = (
            "Build a chatbot\n\nAdditional context from user:\nUse Python"
        )
        mock_poll.assert_called_once_with(
            "/api/decompose-description",
            {"description": expected_description},
        )

    @pytest.mark.asyncio
    async def test_decompose_goal_returns_unachievable_goal(self):
        """Test decomposition returning unachievable goal response."""
        with (
            patch.object(service, "_is_dummy_mode", return_value=False),
            patch.object(
                service, "_submit_and_poll", new_callable=AsyncMock
            ) as mock_poll,
        ):
            mock_poll.return_value = {
                "type": "unachievable_goal",
                "reason": "Cannot do X",
                "suggested_goal": "Try Y instead",
            }
            result = await service.decompose_goal_external("Do something impossible")

        assert result == {
            "type": "unachievable_goal",
            "reason": "Cannot do X",
            "suggested_goal": "Try Y instead",
        }

    @pytest.mark.asyncio
    async def test_decompose_goal_handles_poll_error(self):
        """Test that errors from _submit_and_poll are passed through."""
        with (
            patch.object(service, "_is_dummy_mode", return_value=False),
            patch.object(
                service, "_submit_and_poll", new_callable=AsyncMock
            ) as mock_poll,
        ):
            mock_poll.return_value = {
                "type": "error",
                "error": "HTTP error calling Agent Generator: Server error",
                "error_type": "http_error",
            }
            result = await service.decompose_goal_external("Build a chatbot")

        assert result is not None
        assert result.get("type") == "error"
        assert result.get("error_type") == "http_error"

    @pytest.mark.asyncio
    async def test_decompose_goal_handles_unexpected_exception(self):
        """Test that unexpected exceptions are caught and returned as errors."""
        with (
            patch.object(service, "_is_dummy_mode", return_value=False),
            patch.object(
                service, "_submit_and_poll", new_callable=AsyncMock
            ) as mock_poll,
        ):
            mock_poll.side_effect = RuntimeError("unexpected")
            result = await service.decompose_goal_external("Build a chatbot")

        assert result is not None
        assert result.get("type") == "error"
        assert result.get("error_type") == "unexpected_error"


class TestGenerateAgentExternal:
    """Test generate_agent_external function."""

    def setup_method(self):
        """Reset client singleton before each test."""
        service._settings = None
        service._client = None

    @pytest.mark.asyncio
    async def test_generate_agent_success(self):
        """Test successful agent generation."""
        agent_json = {
            "name": "Test Agent",
            "nodes": [],
            "links": [],
        }

        with (
            patch.object(service, "_is_dummy_mode", return_value=False),
            patch.object(
                service, "_submit_and_poll", new_callable=AsyncMock
            ) as mock_poll,
        ):
            mock_poll.return_value = {"success": True, "agent_json": agent_json}

            instructions = {"type": "instructions", "steps": ["Step 1"]}
            result = await service.generate_agent_external(instructions)

        assert result == agent_json
        mock_poll.assert_called_once_with(
            "/api/generate-agent",
            {"instructions": instructions},
        )

    @pytest.mark.asyncio
    async def test_generate_agent_handles_error(self):
        """Test agent generation handles errors gracefully."""
        with (
            patch.object(service, "_is_dummy_mode", return_value=False),
            patch.object(
                service, "_submit_and_poll", new_callable=AsyncMock
            ) as mock_poll,
        ):
            mock_poll.return_value = {
                "type": "error",
                "error": "Connection failed",
                "error_type": "connection_error",
            }
            result = await service.generate_agent_external({"steps": []})

        assert result is not None
        assert result.get("type") == "error"
        assert result.get("error_type") == "connection_error"


class TestGenerateAgentPatchExternal:
    """Test generate_agent_patch_external function."""

    def setup_method(self):
        """Reset client singleton before each test."""
        service._settings = None
        service._client = None

    @pytest.mark.asyncio
    async def test_generate_patch_returns_updated_agent(self):
        """Test successful patch generation returning updated agent."""
        updated_agent = {
            "name": "Updated Agent",
            "nodes": [{"id": "1", "block_id": "test"}],
            "links": [],
        }

        with (
            patch.object(service, "_is_dummy_mode", return_value=False),
            patch.object(
                service, "_submit_and_poll", new_callable=AsyncMock
            ) as mock_poll,
        ):
            mock_poll.return_value = {"success": True, "agent_json": updated_agent}

            current_agent = {"name": "Old Agent", "nodes": [], "links": []}
            result = await service.generate_agent_patch_external(
                "Add a new node", current_agent
            )

        assert result == updated_agent
        mock_poll.assert_called_once_with(
            "/api/update-agent",
            {
                "update_request": "Add a new node",
                "current_agent_json": current_agent,
            },
        )

    @pytest.mark.asyncio
    async def test_generate_patch_returns_clarifying_questions(self):
        """Test patch generation returning clarifying questions."""
        with (
            patch.object(service, "_is_dummy_mode", return_value=False),
            patch.object(
                service, "_submit_and_poll", new_callable=AsyncMock
            ) as mock_poll,
        ):
            mock_poll.return_value = {
                "type": "clarifying_questions",
                "questions": ["What type of node?"],
            }
            result = await service.generate_agent_patch_external(
                "Add something", {"nodes": []}
            )

        assert result == {
            "type": "clarifying_questions",
            "questions": ["What type of node?"],
        }


class TestHealthCheck:
    """Test health_check function."""

    def setup_method(self):
        """Reset singletons before each test."""
        service._settings = None
        service._client = None

    @pytest.mark.asyncio
    async def test_health_check_returns_false_when_not_configured(self):
        """Test health check returns False when service not configured."""
        with patch.object(
            service, "is_external_service_configured", return_value=False
        ):
            result = await service.health_check()
            assert result is False

    @pytest.mark.asyncio
    async def test_health_check_returns_true_when_healthy(self):
        """Test health check returns True when service is healthy."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "healthy",
            "blocks_loaded": True,
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        with (
            patch.object(service, "is_external_service_configured", return_value=True),
            patch.object(service, "_is_dummy_mode", return_value=False),
            patch.object(service, "_get_client", return_value=mock_client),
        ):
            result = await service.health_check()

        assert result is True
        mock_client.get.assert_called_once_with("/health")

    @pytest.mark.asyncio
    async def test_health_check_returns_false_when_not_healthy(self):
        """Test health check returns False when service is not healthy."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "unhealthy",
            "blocks_loaded": False,
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        with (
            patch.object(service, "is_external_service_configured", return_value=True),
            patch.object(service, "_is_dummy_mode", return_value=False),
            patch.object(service, "_get_client", return_value=mock_client),
        ):
            result = await service.health_check()

        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_returns_false_on_error(self):
        """Test health check returns False on connection error."""
        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.RequestError("Connection failed")

        with (
            patch.object(service, "is_external_service_configured", return_value=True),
            patch.object(service, "_is_dummy_mode", return_value=False),
            patch.object(service, "_get_client", return_value=mock_client),
        ):
            result = await service.health_check()

        assert result is False


class TestGetBlocksExternal:
    """Test get_blocks_external function."""

    def setup_method(self):
        """Reset client singleton before each test."""
        service._settings = None
        service._client = None

    @pytest.mark.asyncio
    async def test_get_blocks_success(self):
        """Test successful blocks retrieval."""
        blocks = [
            {"id": "block1", "name": "Block 1"},
            {"id": "block2", "name": "Block 2"},
        ]
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "blocks": blocks,
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        with (
            patch.object(service, "_is_dummy_mode", return_value=False),
            patch.object(service, "_get_client", return_value=mock_client),
        ):
            result = await service.get_blocks_external()

        assert result == blocks
        mock_client.get.assert_called_once_with("/api/blocks")

    @pytest.mark.asyncio
    async def test_get_blocks_handles_error(self):
        """Test blocks retrieval handles errors gracefully."""
        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.RequestError("Connection failed")

        with (
            patch.object(service, "_is_dummy_mode", return_value=False),
            patch.object(service, "_get_client", return_value=mock_client),
        ):
            result = await service.get_blocks_external()

        assert result is None


class TestLibraryAgentsPassthrough:
    """Test that library_agents are passed correctly in all requests."""

    def setup_method(self):
        """Reset client singleton before each test."""
        service._settings = None
        service._client = None

    @pytest.mark.asyncio
    async def test_decompose_goal_passes_library_agents(self):
        """Test that library_agents are included in decompose goal payload."""
        library_agents = [
            {
                "graph_id": "agent-123",
                "graph_version": 1,
                "name": "Email Sender",
                "description": "Sends emails",
                "input_schema": {"properties": {"to": {"type": "string"}}},
                "output_schema": {"properties": {"sent": {"type": "boolean"}}},
            },
        ]

        with (
            patch.object(service, "_is_dummy_mode", return_value=False),
            patch.object(
                service, "_submit_and_poll", new_callable=AsyncMock
            ) as mock_poll,
        ):
            mock_poll.return_value = {"type": "instructions", "steps": ["Step 1"]}
            await service.decompose_goal_external(
                "Send an email",
                library_agents=library_agents,
            )

        # Verify library_agents was passed in the payload
        call_args = mock_poll.call_args
        payload = call_args[0][1]
        assert payload["library_agents"] == library_agents

    @pytest.mark.asyncio
    async def test_generate_agent_passes_library_agents(self):
        """Test that library_agents are included in generate agent payload."""
        library_agents = [
            {
                "graph_id": "agent-456",
                "graph_version": 2,
                "name": "Data Fetcher",
                "description": "Fetches data from API",
                "input_schema": {"properties": {"url": {"type": "string"}}},
                "output_schema": {"properties": {"data": {"type": "object"}}},
            },
        ]

        with (
            patch.object(service, "_is_dummy_mode", return_value=False),
            patch.object(
                service, "_submit_and_poll", new_callable=AsyncMock
            ) as mock_poll,
        ):
            mock_poll.return_value = {
                "agent_json": {"name": "Test Agent", "nodes": []},
            }
            await service.generate_agent_external(
                {"steps": ["Step 1"]},
                library_agents=library_agents,
            )

        # Verify library_agents was passed in the payload
        call_args = mock_poll.call_args
        payload = call_args[0][1]
        assert payload["library_agents"] == library_agents

    @pytest.mark.asyncio
    async def test_generate_agent_patch_passes_library_agents(self):
        """Test that library_agents are included in patch generation payload."""
        library_agents = [
            {
                "graph_id": "agent-789",
                "graph_version": 1,
                "name": "Slack Notifier",
                "description": "Sends Slack messages",
                "input_schema": {"properties": {"message": {"type": "string"}}},
                "output_schema": {"properties": {"success": {"type": "boolean"}}},
            },
        ]

        with (
            patch.object(service, "_is_dummy_mode", return_value=False),
            patch.object(
                service, "_submit_and_poll", new_callable=AsyncMock
            ) as mock_poll,
        ):
            mock_poll.return_value = {
                "agent_json": {"name": "Updated Agent", "nodes": []},
            }
            await service.generate_agent_patch_external(
                "Add error handling",
                {"name": "Original Agent", "nodes": []},
                library_agents=library_agents,
            )

        # Verify library_agents was passed in the payload
        call_args = mock_poll.call_args
        payload = call_args[0][1]
        assert payload["library_agents"] == library_agents

    @pytest.mark.asyncio
    async def test_decompose_goal_without_library_agents(self):
        """Test that decompose goal works without library_agents."""
        with (
            patch.object(service, "_is_dummy_mode", return_value=False),
            patch.object(
                service, "_submit_and_poll", new_callable=AsyncMock
            ) as mock_poll,
        ):
            mock_poll.return_value = {"type": "instructions", "steps": ["Step 1"]}
            await service.decompose_goal_external("Build a workflow")

        # Verify library_agents was NOT passed when not provided
        call_args = mock_poll.call_args
        payload = call_args[0][1]
        assert "library_agents" not in payload


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
