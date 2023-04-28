import pytest
from autogpt.projects.agent_model import AgentModel


def test_agent_config_init():
    agent_config = AgentModel(
        agent_name="Agent1",
        agent_role="Role1",
        agent_goals=["Goal1", "Goal2"],
    )
    assert agent_config.agent_name == "Agent1"
    assert agent_config.agent_role == "Role1"
    assert agent_config.agent_goals == ["Goal1", "Goal2"]
    assert agent_config.agent_model is None
    assert agent_config.agent_model_type is None
    assert agent_config.prompt_generator is None
    assert agent_config.command_registry is None

    agent_config_with_model = AgentModel(
        agent_name="Agent2",
        agent_role="Role2",
        agent_goals=["Goal3", "Goal4"],
        agent_model="Model1",
        agent_model_type="ModelType1",
    )
    assert agent_config_with_model.agent_name == "Agent2"
    assert agent_config_with_model.agent_role == "Role2"
    assert agent_config_with_model.agent_goals == ["Goal3", "Goal4"]
    assert agent_config_with_model.agent_model == "Model1"
    assert agent_config_with_model.agent_model_type == "ModelType1"
    assert agent_config_with_model.prompt_generator is None
    assert agent_config_with_model.command_registry is None


if __name__ == "__main__":
    pytest.main([__file__])
