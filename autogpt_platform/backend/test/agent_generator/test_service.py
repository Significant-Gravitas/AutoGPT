"""
Tests for the Agent Generator external service client.

This test suite verifies the external Agent Generator service integration,
including service detection, API calls, and error handling.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from backend.api.features.chat.tools.agent_generator import service


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


class TestDecomposeGoalExternal:
    """Test decompose_goal_external function."""

    def setup_method(self):
        """Reset client singleton before each test."""
        service._settings = None
        service._client = None

    @pytest.mark.asyncio
    async def test_decompose_goal_returns_instructions(self):
        """Test successful decomposition returning instructions."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "type": "instructions",
            "steps": ["Step 1", "Step 2"],
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        with patch.object(service, "_get_client", return_value=mock_client):
            result = await service.decompose_goal_external("Build a chatbot")

        assert result == {"type": "instructions", "steps": ["Step 1", "Step 2"]}
        mock_client.post.assert_called_once_with(
            "/api/decompose-description", json={"description": "Build a chatbot"}
        )

    @pytest.mark.asyncio
    async def test_decompose_goal_returns_clarifying_questions(self):
        """Test decomposition returning clarifying questions."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "type": "clarifying_questions",
            "questions": ["What platform?", "What language?"],
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        with patch.object(service, "_get_client", return_value=mock_client):
            result = await service.decompose_goal_external("Build something")

        assert result == {
            "type": "clarifying_questions",
            "questions": ["What platform?", "What language?"],
        }

    @pytest.mark.asyncio
    async def test_decompose_goal_with_context(self):
        """Test decomposition with additional context."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "type": "instructions",
            "steps": ["Step 1"],
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        with patch.object(service, "_get_client", return_value=mock_client):
            await service.decompose_goal_external(
                "Build a chatbot", context="Use Python"
            )

        mock_client.post.assert_called_once_with(
            "/api/decompose-description",
            json={"description": "Build a chatbot", "user_instruction": "Use Python"},
        )

    @pytest.mark.asyncio
    async def test_decompose_goal_returns_unachievable_goal(self):
        """Test decomposition returning unachievable goal response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "type": "unachievable_goal",
            "reason": "Cannot do X",
            "suggested_goal": "Try Y instead",
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        with patch.object(service, "_get_client", return_value=mock_client):
            result = await service.decompose_goal_external("Do something impossible")

        assert result == {
            "type": "unachievable_goal",
            "reason": "Cannot do X",
            "suggested_goal": "Try Y instead",
        }

    @pytest.mark.asyncio
    async def test_decompose_goal_handles_http_error(self):
        """Test decomposition handles HTTP errors gracefully."""
        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.HTTPStatusError(
            "Server error", request=MagicMock(), response=MagicMock()
        )

        with patch.object(service, "_get_client", return_value=mock_client):
            result = await service.decompose_goal_external("Build a chatbot")

        assert result is None

    @pytest.mark.asyncio
    async def test_decompose_goal_handles_request_error(self):
        """Test decomposition handles request errors gracefully."""
        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.RequestError("Connection failed")

        with patch.object(service, "_get_client", return_value=mock_client):
            result = await service.decompose_goal_external("Build a chatbot")

        assert result is None

    @pytest.mark.asyncio
    async def test_decompose_goal_handles_service_error(self):
        """Test decomposition handles service returning error."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": False,
            "error": "Internal error",
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        with patch.object(service, "_get_client", return_value=mock_client):
            result = await service.decompose_goal_external("Build a chatbot")

        assert result is None


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
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "agent_json": agent_json,
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        instructions = {"type": "instructions", "steps": ["Step 1"]}

        with patch.object(service, "_get_client", return_value=mock_client):
            result = await service.generate_agent_external(instructions)

        assert result == agent_json
        mock_client.post.assert_called_once_with(
            "/api/generate-agent", json={"instructions": instructions}
        )

    @pytest.mark.asyncio
    async def test_generate_agent_handles_error(self):
        """Test agent generation handles errors gracefully."""
        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.RequestError("Connection failed")

        with patch.object(service, "_get_client", return_value=mock_client):
            result = await service.generate_agent_external({"steps": []})

        assert result is None


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
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "agent_json": updated_agent,
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        current_agent = {"name": "Old Agent", "nodes": [], "links": []}

        with patch.object(service, "_get_client", return_value=mock_client):
            result = await service.generate_agent_patch_external(
                "Add a new node", current_agent
            )

        assert result == updated_agent
        mock_client.post.assert_called_once_with(
            "/api/update-agent",
            json={
                "update_request": "Add a new node",
                "current_agent_json": current_agent,
            },
        )

    @pytest.mark.asyncio
    async def test_generate_patch_returns_clarifying_questions(self):
        """Test patch generation returning clarifying questions."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "type": "clarifying_questions",
            "questions": ["What type of node?"],
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        with patch.object(service, "_get_client", return_value=mock_client):
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

        with patch.object(service, "is_external_service_configured", return_value=True):
            with patch.object(service, "_get_client", return_value=mock_client):
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

        with patch.object(service, "is_external_service_configured", return_value=True):
            with patch.object(service, "_get_client", return_value=mock_client):
                result = await service.health_check()

        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_returns_false_on_error(self):
        """Test health check returns False on connection error."""
        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.RequestError("Connection failed")

        with patch.object(service, "is_external_service_configured", return_value=True):
            with patch.object(service, "_get_client", return_value=mock_client):
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

        with patch.object(service, "_get_client", return_value=mock_client):
            result = await service.get_blocks_external()

        assert result == blocks
        mock_client.get.assert_called_once_with("/api/blocks")

    @pytest.mark.asyncio
    async def test_get_blocks_handles_error(self):
        """Test blocks retrieval handles errors gracefully."""
        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.RequestError("Connection failed")

        with patch.object(service, "_get_client", return_value=mock_client):
            result = await service.get_blocks_external()

        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
