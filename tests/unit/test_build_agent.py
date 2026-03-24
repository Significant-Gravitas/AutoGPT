"""Unit tests for the BuildAgent and AgentBuilder classes."""
from unittest.mock import MagicMock, patch

import pytest

from autogpt.agent.agent import Agent
from autogpt.agent.agent_builder import AgentBuilder, AGENT_TEMPLATES
from autogpt.agent.build_agent import BuildAgent


class TestBuildAgent:
    """Tests for the BuildAgent class."""

    def test_build_agent_inherits_from_agent(self):
        memory = MagicMock()
        agent = BuildAgent(
            ai_name="TestBuilder",
            memory=memory,
            full_message_history=[],
            next_action_count=0,
            system_prompt="test prompt",
            triggering_prompt="test trigger",
        )
        assert isinstance(agent, Agent)

    def test_build_agent_stores_project_dir(self):
        memory = MagicMock()
        agent = BuildAgent(
            ai_name="TestBuilder",
            memory=memory,
            full_message_history=[],
            next_action_count=0,
            system_prompt="test prompt",
            triggering_prompt="test trigger",
            project_dir="/tmp/project",
        )
        assert agent.project_dir == "/tmp/project"

    def test_build_agent_stores_build_config(self):
        memory = MagicMock()
        config = {"test_command": "pytest"}
        agent = BuildAgent(
            ai_name="TestBuilder",
            memory=memory,
            full_message_history=[],
            next_action_count=0,
            system_prompt="test prompt",
            triggering_prompt="test trigger",
            build_config=config,
        )
        assert agent.build_config == config

    def test_build_agent_default_build_config(self):
        memory = MagicMock()
        agent = BuildAgent(
            ai_name="TestBuilder",
            memory=memory,
            full_message_history=[],
            next_action_count=0,
            system_prompt="test prompt",
            triggering_prompt="test trigger",
        )
        assert agent.build_config == {}

    @patch("autogpt.agent.build_agent.Config")
    def test_get_build_prompt_contains_name_and_role(self, mock_config):
        mock_config.return_value.execute_local_commands = False
        prompt = BuildAgent.get_build_prompt(
            ai_name="BuildGPT",
            ai_role="a build agent",
            ai_goals=["Build the project"],
        )
        assert "BuildGPT" in prompt
        assert "a build agent" in prompt
        assert "Build the project" in prompt

    @patch("autogpt.agent.build_agent.Config")
    def test_get_build_prompt_contains_project_dir(self, mock_config):
        mock_config.return_value.execute_local_commands = False
        prompt = BuildAgent.get_build_prompt(
            ai_name="BuildGPT",
            ai_role="a build agent",
            ai_goals=["Build"],
            project_dir="/my/project",
        )
        assert "/my/project" in prompt

    @patch("autogpt.agent.build_agent.Config")
    def test_get_build_prompt_contains_build_commands(self, mock_config):
        mock_config.return_value.execute_local_commands = False
        prompt = BuildAgent.get_build_prompt(
            ai_name="BuildGPT",
            ai_role="a build agent",
            ai_goals=["Build"],
        )
        assert "evaluate_code" in prompt
        assert "improve_code" in prompt
        assert "write_tests" in prompt
        assert "execute_python_file" in prompt

    @patch("autogpt.agent.build_agent.Config")
    def test_get_build_prompt_includes_shell_when_enabled(self, mock_config):
        mock_config.return_value.execute_local_commands = True
        prompt = BuildAgent.get_build_prompt(
            ai_name="BuildGPT",
            ai_role="a build agent",
            ai_goals=["Build"],
        )
        assert "execute_shell" in prompt


class TestAgentBuilder:
    """Tests for the AgentBuilder class."""

    def test_list_templates(self):
        templates = AgentBuilder.list_templates()
        assert "build" in templates
        assert "research" in templates
        assert "code-review" in templates

    def test_from_template_sets_values(self):
        builder = AgentBuilder().from_template("build")
        assert builder._ai_name == "BuildGPT"
        assert builder._ai_role != ""
        assert len(builder._ai_goals) > 0

    def test_from_template_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown template"):
            AgentBuilder().from_template("nonexistent")

    def test_with_name_overrides(self):
        builder = AgentBuilder().from_template("build").with_name("CustomName")
        assert builder._ai_name == "CustomName"

    def test_with_role_overrides(self):
        builder = AgentBuilder().from_template("build").with_role("custom role")
        assert builder._ai_role == "custom role"

    def test_with_goals_overrides(self):
        goals = ["Goal 1", "Goal 2"]
        builder = AgentBuilder().from_template("build").with_goals(goals)
        assert builder._ai_goals == goals

    def test_with_project_dir(self):
        builder = AgentBuilder().with_project_dir("/tmp/project")
        assert builder._project_dir == "/tmp/project"

    def test_with_build_config(self):
        config = {"key": "value"}
        builder = AgentBuilder().with_build_config(config)
        assert builder._build_config == config

    def test_fluent_interface(self):
        builder = (
            AgentBuilder()
            .with_name("Test")
            .with_role("tester")
            .with_goals(["test"])
            .with_project_dir("/tmp")
            .with_build_config({"a": "b"})
            .with_memory_type("local")
            .with_next_action_count(5)
        )
        assert builder._ai_name == "Test"
        assert builder._ai_role == "tester"
        assert builder._ai_goals == ["test"]
        assert builder._project_dir == "/tmp"
        assert builder._build_config == {"a": "b"}
        assert builder._memory_type == "local"
        assert builder._next_action_count == 5

    @patch("autogpt.agent.agent_builder.get_memory")
    @patch("autogpt.agent.agent_builder.Config")
    @patch("autogpt.agent.build_agent.Config")
    def test_build_agent_returns_build_agent(
        self, mock_ba_config, mock_config, mock_memory
    ):
        mock_ba_config.return_value.execute_local_commands = False
        mock_memory.return_value = MagicMock()
        agent = (
            AgentBuilder()
            .from_template("build")
            .with_project_dir("/tmp")
            .build_agent(agent_type="build")
        )
        assert isinstance(agent, BuildAgent)
        assert agent.project_dir == "/tmp"

    @patch("autogpt.agent.agent_builder.get_memory")
    @patch("autogpt.agent.agent_builder.Config")
    @patch("autogpt.prompt.Config")
    def test_build_agent_returns_default_agent(
        self, mock_prompt_config, mock_config, mock_memory
    ):
        mock_memory.return_value = MagicMock()
        mock_prompt_config.return_value.execute_local_commands = False
        mock_prompt_config.return_value.huggingface_audio_to_text_model = None
        mock_prompt_config.return_value.allow_downloads = False
        agent = (
            AgentBuilder()
            .with_name("TestAgent")
            .with_role("a test agent")
            .with_goals(["Test"])
            .build_agent(agent_type="default")
        )
        assert isinstance(agent, Agent)
        assert not isinstance(agent, BuildAgent)

    def test_templates_have_required_keys(self):
        for name, template in AGENT_TEMPLATES.items():
            assert "ai_name" in template, f"Template '{name}' missing ai_name"
            assert "ai_role" in template, f"Template '{name}' missing ai_role"
            assert "ai_goals" in template, f"Template '{name}' missing ai_goals"
            assert len(template["ai_goals"]) > 0, f"Template '{name}' has no goals"
