"""
Tests for the Agent Generator core module.

This test suite verifies that the core functions correctly delegate to
the external Agent Generator service.
"""

from unittest.mock import AsyncMock, patch

import pytest

from backend.api.features.chat.tools.agent_generator import core
from backend.api.features.chat.tools.agent_generator.core import (
    AgentGeneratorNotConfiguredError,
)


class TestServiceNotConfigured:
    """Test that functions raise AgentGeneratorNotConfiguredError when service is not configured."""

    @pytest.mark.asyncio
    async def test_decompose_goal_raises_when_not_configured(self):
        """Test that decompose_goal raises error when service not configured."""
        with patch.object(core, "is_external_service_configured", return_value=False):
            with pytest.raises(AgentGeneratorNotConfiguredError):
                await core.decompose_goal("Build a chatbot")

    @pytest.mark.asyncio
    async def test_generate_agent_raises_when_not_configured(self):
        """Test that generate_agent raises error when service not configured."""
        with patch.object(core, "is_external_service_configured", return_value=False):
            with pytest.raises(AgentGeneratorNotConfiguredError):
                await core.generate_agent({"steps": []})

    @pytest.mark.asyncio
    async def test_generate_agent_patch_raises_when_not_configured(self):
        """Test that generate_agent_patch raises error when service not configured."""
        with patch.object(core, "is_external_service_configured", return_value=False):
            with pytest.raises(AgentGeneratorNotConfiguredError):
                await core.generate_agent_patch("Add a node", {"nodes": []})


class TestDecomposeGoal:
    """Test decompose_goal function service delegation."""

    @pytest.mark.asyncio
    async def test_calls_external_service(self):
        """Test that decompose_goal calls the external service."""
        expected_result = {"type": "instructions", "steps": ["Step 1"]}

        with patch.object(
            core, "is_external_service_configured", return_value=True
        ), patch.object(
            core, "decompose_goal_external", new_callable=AsyncMock
        ) as mock_external:
            mock_external.return_value = expected_result

            result = await core.decompose_goal("Build a chatbot")

            mock_external.assert_called_once_with("Build a chatbot", "")
            assert result == expected_result

    @pytest.mark.asyncio
    async def test_passes_context_to_external_service(self):
        """Test that decompose_goal passes context to external service."""
        expected_result = {"type": "instructions", "steps": ["Step 1"]}

        with patch.object(
            core, "is_external_service_configured", return_value=True
        ), patch.object(
            core, "decompose_goal_external", new_callable=AsyncMock
        ) as mock_external:
            mock_external.return_value = expected_result

            await core.decompose_goal("Build a chatbot", "Use Python")

            mock_external.assert_called_once_with("Build a chatbot", "Use Python")

    @pytest.mark.asyncio
    async def test_returns_none_on_service_failure(self):
        """Test that decompose_goal returns None when external service fails."""
        with patch.object(
            core, "is_external_service_configured", return_value=True
        ), patch.object(
            core, "decompose_goal_external", new_callable=AsyncMock
        ) as mock_external:
            mock_external.return_value = None

            result = await core.decompose_goal("Build a chatbot")

            assert result is None


class TestGenerateAgent:
    """Test generate_agent function service delegation."""

    @pytest.mark.asyncio
    async def test_calls_external_service(self):
        """Test that generate_agent calls the external service."""
        expected_result = {"name": "Test Agent", "nodes": [], "links": []}

        with patch.object(
            core, "is_external_service_configured", return_value=True
        ), patch.object(
            core, "generate_agent_external", new_callable=AsyncMock
        ) as mock_external:
            mock_external.return_value = expected_result

            instructions = {"type": "instructions", "steps": ["Step 1"]}
            result = await core.generate_agent(instructions)

            mock_external.assert_called_once_with(instructions)
            # Result should have id, version, is_active added if not present
            assert result is not None
            assert result["name"] == "Test Agent"
            assert "id" in result
            assert result["version"] == 1
            assert result["is_active"] is True

    @pytest.mark.asyncio
    async def test_preserves_existing_id_and_version(self):
        """Test that external service result preserves existing id and version."""
        expected_result = {
            "id": "existing-id",
            "version": 3,
            "is_active": False,
            "name": "Test Agent",
        }

        with patch.object(
            core, "is_external_service_configured", return_value=True
        ), patch.object(
            core, "generate_agent_external", new_callable=AsyncMock
        ) as mock_external:
            mock_external.return_value = expected_result.copy()

            result = await core.generate_agent({"steps": []})

            assert result is not None
            assert result["id"] == "existing-id"
            assert result["version"] == 3
            assert result["is_active"] is False

    @pytest.mark.asyncio
    async def test_returns_none_when_external_service_fails(self):
        """Test that generate_agent returns None when external service fails."""
        with patch.object(
            core, "is_external_service_configured", return_value=True
        ), patch.object(
            core, "generate_agent_external", new_callable=AsyncMock
        ) as mock_external:
            mock_external.return_value = None

            result = await core.generate_agent({"steps": []})

            assert result is None


class TestGenerateAgentPatch:
    """Test generate_agent_patch function service delegation."""

    @pytest.mark.asyncio
    async def test_calls_external_service(self):
        """Test that generate_agent_patch calls the external service."""
        expected_result = {"name": "Updated Agent", "nodes": [], "links": []}

        with patch.object(
            core, "is_external_service_configured", return_value=True
        ), patch.object(
            core, "generate_agent_patch_external", new_callable=AsyncMock
        ) as mock_external:
            mock_external.return_value = expected_result

            current_agent = {"nodes": [], "links": []}
            result = await core.generate_agent_patch("Add a node", current_agent)

            mock_external.assert_called_once_with("Add a node", current_agent)
            assert result == expected_result

    @pytest.mark.asyncio
    async def test_returns_clarifying_questions(self):
        """Test that generate_agent_patch returns clarifying questions."""
        expected_result = {
            "type": "clarifying_questions",
            "questions": [{"question": "What type of node?"}],
        }

        with patch.object(
            core, "is_external_service_configured", return_value=True
        ), patch.object(
            core, "generate_agent_patch_external", new_callable=AsyncMock
        ) as mock_external:
            mock_external.return_value = expected_result

            result = await core.generate_agent_patch("Add a node", {"nodes": []})

            assert result == expected_result

    @pytest.mark.asyncio
    async def test_returns_none_when_external_service_fails(self):
        """Test that generate_agent_patch returns None when service fails."""
        with patch.object(
            core, "is_external_service_configured", return_value=True
        ), patch.object(
            core, "generate_agent_patch_external", new_callable=AsyncMock
        ) as mock_external:
            mock_external.return_value = None

            result = await core.generate_agent_patch("Add a node", {"nodes": []})

            assert result is None


class TestJsonToGraph:
    """Test json_to_graph function."""

    def test_converts_agent_json_to_graph(self):
        """Test conversion of agent JSON to Graph model."""
        agent_json = {
            "id": "test-id",
            "version": 2,
            "is_active": True,
            "name": "Test Agent",
            "description": "A test agent",
            "nodes": [
                {
                    "id": "node1",
                    "block_id": "block1",
                    "input_default": {"key": "value"},
                    "metadata": {"x": 100},
                }
            ],
            "links": [
                {
                    "id": "link1",
                    "source_id": "node1",
                    "sink_id": "output",
                    "source_name": "result",
                    "sink_name": "input",
                    "is_static": False,
                }
            ],
        }

        graph = core.json_to_graph(agent_json)

        assert graph.id == "test-id"
        assert graph.version == 2
        assert graph.is_active is True
        assert graph.name == "Test Agent"
        assert graph.description == "A test agent"
        assert len(graph.nodes) == 1
        assert graph.nodes[0].id == "node1"
        assert graph.nodes[0].block_id == "block1"
        assert len(graph.links) == 1
        assert graph.links[0].source_id == "node1"

    def test_generates_ids_if_missing(self):
        """Test that missing IDs are generated."""
        agent_json = {
            "name": "Test Agent",
            "nodes": [{"block_id": "block1"}],
            "links": [],
        }

        graph = core.json_to_graph(agent_json)

        assert graph.id is not None
        assert graph.nodes[0].id is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
