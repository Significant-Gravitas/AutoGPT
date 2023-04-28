import pytest
from autogpt.projects.agent_model import AgentModel
from autogpt.projects import Project


@pytest.fixture
def lead_agent():
    return AgentModel(agent_name="Lead Agent", agent_role="Leader", agent_goals=["Goal1"])


@pytest.fixture
def delegated_agent():
    return AgentModel(agent_name="Delegated Agent", agent_role="Delegated", agent_goals=["Goal1"])


def test_project_init():
    lead_agent = AgentModel(
        agent_name="Agent1",
        agent_role="Role1",
        agent_goals=["Goal1", "Goal2"],
    )
    delegated_agent = AgentModel(
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
    lead_agent = AgentModel(
        agent_name="Agent1",
        agent_role="Role1",
        agent_goals=["Goal1", "Goal2"],
    )
    delegated_agent = AgentModel(
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



def test_project_init_unittest_style():
    lead_agent = AgentModel(
        agent_name="Lead Agent",
        agent_role="Lead",
        agent_goals=["Goal 1", "Goal 2"],
        agent_model="Model 1",
        agent_model_type="Type 1",
    )
    delegated_agents = [
        AgentModel(
            agent_name="Delegated Agent 1",
            agent_role="Delegated",
            agent_goals=["Goal 3"],
            agent_model="Model 2",
            agent_model_type="Type 2",
        ),
        AgentModel(
            agent_name="Delegated Agent 2",
            agent_role="Delegated",
            agent_goals=["Goal 4", "Goal 5"],
            agent_model="Model 3",
            agent_model_type="Type 3",
        ),
    ]

    project = Project(
        project_name="Project 1",
        api_budget=500.0,
        lead_agent=lead_agent,
        delegated_agents=delegated_agents,
    )

    assert project.project_name == "Project 1"
    assert project.api_budget == 500.0
    assert project.lead_agent.agent_name == "Lead Agent"
    assert project.lead_agent.agent_role == "Lead"
    assert project.lead_agent.agent_goals == ["Goal 1", "Goal 2"]
    assert project.lead_agent.agent_model == "Model 1"
    assert project.lead_agent.agent_model_type == "Type 1"
    assert project.delegated_agents[0].agent_name == "Delegated Agent 1"
    assert project.delegated_agents[0].agent_role == "Delegated"
    assert project.delegated_agents[0].agent_goals == ["Goal 3"]
    assert project.delegated_agents[0].agent_model == "Model 2"
    assert project.delegated_agents[0].agent_model_type == "Type 2"
    assert project.delegated_agents[1].agent_name == "Delegated Agent 2"
    assert project.delegated_agents[1].agent_role == "Delegated"
    assert project.delegated_agents[1].agent_goals == ["Goal 4", "Goal 5"]

    
if __name__ == "__main__":
    pytest.main([__file__])