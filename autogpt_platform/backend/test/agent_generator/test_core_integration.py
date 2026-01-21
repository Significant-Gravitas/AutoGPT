"""
Tests for the Agent Generator core module's external service integration.

This test suite verifies that the core functions correctly delegate to
the external service when configured, and fall back to built-in
implementation when not configured.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.api.features.chat.tools.agent_generator import core


class TestDecomposeGoal:
    """Test decompose_goal function service delegation."""

    @pytest.mark.asyncio
    async def test_uses_external_service_when_configured(self):
        """Test that decompose_goal uses external service when configured."""
        expected_result = {"type": "instructions", "steps": ["Step 1"]}

        with patch.object(
            core, "is_external_service_configured", return_value=True
        ) as mock_is_configured:
            with patch.object(
                core, "decompose_goal_external", new_callable=AsyncMock
            ) as mock_external:
                mock_external.return_value = expected_result

                result = await core.decompose_goal("Build a chatbot")

                mock_is_configured.assert_called_once()
                mock_external.assert_called_once_with("Build a chatbot", "")
                assert result == expected_result

    @pytest.mark.asyncio
    async def test_uses_external_service_with_context(self):
        """Test that decompose_goal passes context to external service."""
        expected_result = {"type": "instructions", "steps": ["Step 1"]}

        with patch.object(core, "is_external_service_configured", return_value=True):
            with patch.object(
                core, "decompose_goal_external", new_callable=AsyncMock
            ) as mock_external:
                mock_external.return_value = expected_result

                await core.decompose_goal("Build a chatbot", "Use Python")

                mock_external.assert_called_once_with("Build a chatbot", "Use Python")

    @pytest.mark.asyncio
    async def test_falls_back_to_builtin_when_not_configured(self):
        """Test that decompose_goal uses built-in when external not configured."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            '{"type": "instructions", "steps": ["Step 1"]}'
        )
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch.object(core, "is_external_service_configured", return_value=False):
            with patch.object(
                core, "_get_builtin_client", return_value=(mock_client, "test-model")
            ):
                with patch.object(
                    core,
                    "_get_builtin_prompts",
                    return_value=("prompt1", "prompt2", "prompt3"),
                ):
                    with patch.object(
                        core, "get_block_summaries", return_value="block summaries"
                    ):
                        result = await core.decompose_goal("Build a chatbot")

                        # Verify built-in client was used
                        mock_client.chat.completions.create.assert_called_once()
                        assert result == {"type": "instructions", "steps": ["Step 1"]}


class TestGenerateAgent:
    """Test generate_agent function service delegation."""

    @pytest.mark.asyncio
    async def test_uses_external_service_when_configured(self):
        """Test that generate_agent uses external service when configured."""
        expected_result = {"name": "Test Agent", "nodes": [], "links": []}

        with patch.object(core, "is_external_service_configured", return_value=True):
            with patch.object(
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
    async def test_external_result_preserves_existing_id(self):
        """Test that external service result preserves existing id."""
        expected_result = {
            "id": "existing-id",
            "version": 3,
            "is_active": False,
            "name": "Test Agent",
        }

        with patch.object(core, "is_external_service_configured", return_value=True):
            with patch.object(
                core, "generate_agent_external", new_callable=AsyncMock
            ) as mock_external:
                mock_external.return_value = expected_result.copy()

                result = await core.generate_agent({"steps": []})

                assert result is not None
                assert result["id"] == "existing-id"
                assert result["version"] == 3
                assert result["is_active"] is False

    @pytest.mark.asyncio
    async def test_falls_back_to_builtin_when_not_configured(self):
        """Test that generate_agent uses built-in when external not configured."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            '{"name": "Generated Agent", "nodes": [], "links": []}'
        )
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch.object(core, "is_external_service_configured", return_value=False):
            with patch.object(
                core, "_get_builtin_client", return_value=(mock_client, "test-model")
            ):
                with patch.object(
                    core,
                    "_get_builtin_prompts",
                    return_value=("prompt1", "prompt2", "prompt3"),
                ):
                    with patch.object(
                        core, "get_block_summaries", return_value="block summaries"
                    ):
                        result = await core.generate_agent({"steps": []})

                        mock_client.chat.completions.create.assert_called_once()
                        assert result is not None
                        assert result["name"] == "Generated Agent"
                        assert "id" in result

    @pytest.mark.asyncio
    async def test_returns_none_when_external_service_fails(self):
        """Test that generate_agent returns None when external service fails."""
        with patch.object(core, "is_external_service_configured", return_value=True):
            with patch.object(
                core, "generate_agent_external", new_callable=AsyncMock
            ) as mock_external:
                mock_external.return_value = None

                result = await core.generate_agent({"steps": []})

                assert result is None


class TestGenerateAgentPatch:
    """Test generate_agent_patch function service delegation."""

    @pytest.mark.asyncio
    async def test_uses_external_service_when_configured(self):
        """Test that generate_agent_patch uses external service when configured."""
        expected_result = {"patches": [{"type": "modify", "node_id": "1"}]}

        with patch.object(core, "is_external_service_configured", return_value=True):
            with patch.object(
                core, "generate_agent_patch_external", new_callable=AsyncMock
            ) as mock_external:
                mock_external.return_value = expected_result

                current_agent = {"nodes": [], "links": []}
                result = await core.generate_agent_patch("Add a node", current_agent)

                mock_external.assert_called_once_with("Add a node", current_agent)
                assert result == expected_result

    @pytest.mark.asyncio
    async def test_falls_back_to_builtin_when_not_configured(self):
        """Test that generate_agent_patch uses built-in when not configured."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"patches": []}'
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch.object(core, "is_external_service_configured", return_value=False):
            with patch.object(
                core, "_get_builtin_client", return_value=(mock_client, "test-model")
            ):
                with patch.object(
                    core,
                    "_get_builtin_prompts",
                    return_value=("prompt1", "prompt2", "prompt3"),
                ):
                    with patch.object(
                        core, "get_block_summaries", return_value="block summaries"
                    ):
                        current_agent = {"nodes": [], "links": []}
                        result = await core.generate_agent_patch(
                            "Add a node", current_agent
                        )

                        mock_client.chat.completions.create.assert_called_once()
                        assert result == {"patches": []}


class TestApplyAgentPatch:
    """Test apply_agent_patch function."""

    def test_modify_patch(self):
        """Test applying a modify patch."""
        current_agent = {
            "nodes": [{"id": "1", "input_default": {"key": "old_value"}}],
            "links": [],
        }
        patch = {
            "patches": [
                {
                    "type": "modify",
                    "node_id": "1",
                    "changes": {"input_default": {"key": "new_value"}},
                }
            ]
        }

        result = core.apply_agent_patch(current_agent, patch)

        assert result["nodes"][0]["input_default"]["key"] == "new_value"
        # Original should not be modified
        assert current_agent["nodes"][0]["input_default"]["key"] == "old_value"

    def test_add_patch(self):
        """Test applying an add patch."""
        current_agent = {"nodes": [], "links": []}
        patch = {
            "patches": [
                {
                    "type": "add",
                    "new_nodes": [{"id": "new1", "block_id": "test"}],
                    "new_links": [
                        {"id": "link1", "source_id": "new1", "sink_id": "out"}
                    ],
                }
            ]
        }

        result = core.apply_agent_patch(current_agent, patch)

        assert len(result["nodes"]) == 1
        assert result["nodes"][0]["id"] == "new1"
        assert len(result["links"]) == 1
        assert result["links"][0]["id"] == "link1"

    def test_remove_patch(self):
        """Test applying a remove patch."""
        current_agent = {
            "nodes": [
                {"id": "1", "block_id": "test1"},
                {"id": "2", "block_id": "test2"},
            ],
            "links": [
                {"id": "link1", "source_id": "1", "sink_id": "2"},
                {"id": "link2", "source_id": "2", "sink_id": "out"},
            ],
        }
        patch = {"patches": [{"type": "remove", "node_ids": ["1"], "link_ids": []}]}

        result = core.apply_agent_patch(current_agent, patch)

        # Node 1 should be removed
        assert len(result["nodes"]) == 1
        assert result["nodes"][0]["id"] == "2"
        # Link referencing node 1 should also be removed
        assert len(result["links"]) == 1
        assert result["links"][0]["id"] == "link2"


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
