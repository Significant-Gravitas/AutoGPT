import pytest
from autogpt.config.project.agent.config import AgentConfig
from autogpt.config.project import Project


@pytest.fixture
def lead_agent():
    return AgentConfig(agent_name="Lead Agent", agent_role="Leader", agent_goals=["Goal1"])


@pytest.fixture
def delegated_agent():
    return AgentConfig(agent_name="Delegated Agent", agent_role="Delegated", agent_goals=["Goal1"])


def test_project_init():
    lead_agent = AgentConfig(
        agent_name="Agent1",
        agent_role="Role1",
        agent_goals=["Goal1", "Goal2"],
    )
    delegated_agent = AgentConfig(
        agent_name="Agent2",
        agent_role="Role2",
        agent_goals=["Goal3", "Goal4"],
    )
    project = Project(
        project_name="Test Project",
        api_budget=100.0,
        lead_agent=lead_agent,
        delegated_agents=[delegated_agent],
    )
    assert project.project_name == "Test Project"
    assert project.api_budget == 100.0
    assert project.lead_agent == lead_agent
    assert project.delegated_agents == [delegated_agent]


def test_project_to_dict():
    lead_agent = AgentConfig(
        agent_name="Agent1",
        agent_role="Role1",
        agent_goals=["Goal1", "Goal2"],
    )
    delegated_agent = AgentConfig(
        agent_name="Agent2",
        agent_role="Role2",
        agent_goals=["Goal3", "Goal4"],
    )
    project = Project(
        project_name="Test Project",
        api_budget=100.0,
        lead_agent=lead_agent,
        delegated_agents=[delegated_agent],
    )
    project_dict = project.to_dict()
    assert project_dict["project_name"] == "Test Project"
    assert project_dict["api_budget"] == 100.0
    assert project_dict["lead_agent"]["agent_name"] == "Agent1"
    assert project_dict["lead_agent"]["agent_role"] == "Role1"
    assert project_dict["lead_agent"]["agent_goals"] == ["Goal1", "Goal2"]
    assert project_dict["delegated_agents"][0]["agent_name"] == "Agent2"
    assert project_dict["delegated_agents"][0]["agent_role"] == "Role2"
    assert project_dict["delegated_agents"][0]["agent_goals"] == ["Goal3", "Goal4"]


def test_project_with_fixture(lead_agent, delegated_agent):
    project = Project("Test Project", 100.0, lead_agent, [delegated_agent])

    assert project.project_name == "Test Project"
    assert project.api_budget == 100.0
    assert project.lead_agent == lead_agent
    assert project.delegated_agents == [delegated_agent]


def test_project_to_dict_with_fixture(lead_agent, delegated_agent):
    project = Project("Test Project", 100.0, lead_agent, [delegated_agent])
    project_dict = project.to_dict()

    assert project_dict["project_name"] == "Test Project"
    assert project_dict["api_budget"] == 100.0
    assert project_dict["lead_agent"] == lead_agent.to_dict()
    assert project_dict["delegated_agents"] == [delegated_agent.to_dict()]


def test_project_without_delegated_agents(lead_agent):
    project = Project("Test Project", 100.0, lead_agent)

    assert project.project_name == "Test Project"
    assert project.api_budget == 100.

def test_project_str(lead_agent, delegated_agent):
    project = Project("Test Project", 100.0, lead_agent, [delegated_agent])

    assert str(project) == str(project.to_dict())

if __name__ == "__main__":
    pytest.main([__file__])
