import os
import pytest
from pathlib import Path
from autogpt.projects.project import Project
from autogpt.projects.agent_model import AgentModel
from autogpt.projects.projects_broker import ProjectsBroker

CONFIG_FILE = str(Path(os.getcwd()) / "test_ai_settings.yaml")

@pytest.fixture
def project_config_broker():
    return ProjectsBroker(config_file=CONFIG_FILE)


@pytest.fixture
def lead_agent():
    return AgentModel(
        agent_name="Lead Agent",
        agent_role="Lead",
        agent_goals=["Lead goal 1", "Lead goal 2"],
        agent_model="Lead model",
        agent_model_type="Lead type",
        prompt_generator=None,
        command_registry=None,
    )


@pytest.fixture
def delegated_agent1():
    return AgentModel(
        agent_name="Delegated Agent 1",
        agent_role="Delegated",
        agent_goals=["Delegated goal 1", "Delegated goal 2"],
        agent_model="Delegated model",
        agent_model_type="Delegated type",
        prompt_generator=None,
        command_registry=None,
    )


@pytest.fixture
def delegated_agent2():
    return AgentModel(
        agent_name="Delegated Agent 2",
        agent_role="Delegated",
        agent_goals=["Delegated goal 1", "Delegated goal 2"],
        agent_model="Delegated model",
        agent_model_type="Delegated type",
        prompt_generator=None,
        command_registry=None,
    )


# Test whether the set_project_name() method sets the project name correctly
def test_set_project_name(project_config_broker):
    project_name = "Project1"
    project_config_broker.set_project_name(project_name)
    assert project_config_broker.get_project_name() == project_name


# Test whether the set_lead_agent() method sets the lead agent correctly
def test_set_lead_agent(project_config_broker, lead_agent):
    project_config_broker.set_lead_agent(lead_agent)
    assert project_config_broker.get_lead_agent() == lead_agent


# Test whether the add_delegated_agent() method adds a delegated agent correctly
def test_add_delegated_agent(project_config_broker, delegated_agent1):
    project_config_broker.add_delegated_agent(delegated_agent1)
    assert delegated_agent1 in project_config_broker.get_delegated_agents()


# Test whether the get_total_api_budget() method calculates the total API budget correctly
def test_get_total_api_budget(project_config_broker, lead_agent, delegated_agent1, delegated_agent2):
    project = Project(
        project_name="Project1",
        api_budget=1000.0,
        lead_agent=lead_agent,
        delegated_agents=[delegated_agent1, delegated_agent2],
    )
    project_config_broker.set_project(project)
    assert project_config_broker.get_total_api_budget() == 3000.0  # 1000.0 + 1000.0 + 1000.0


# Test whether the get_delegated_agent_by_name() method returns the correct delegated agent
def test_get_delegated_agent_by_name(project_config_broker, delegated_agent1, delegated_agent2):
    project_config_broker.add_delegated_agent(delegated_agent1)
    project_config_broker.add_delegated_agent(delegated_agent2)
    assert project_config_broker.get_delegated_agent_by_name("Delegated Agent 1") == delegated_agent1
    assert project_config_broker.get_delegated_agent_by_name("Delegated Agent 2") == delegated_agent2
    assert project_config_broker.get_delegated_agent_by_name("Invalid agent name") is None
