import os
import pytest
from pathlib import Path
from autogpt.config.project.config import Project
from autogpt.config.project.agent.config import AgentConfig
from autogpt.config.project import ProjectConfigBroker

CONFIG_FILE = str(Path(os.getcwd()) / "test_agent_settings.yaml")

@pytest.fixture
def project_config_broker():
    return ProjectConfigBroker(config_file=CONFIG_FILE)


def test_load(project_config_broker):
    projects = project_config_broker._load(CONFIG_FILE)
    assert isinstance(projects, list)


def test_save(project_config_broker):
    project_config_broker._save(CONFIG_FILE)
    assert os.path.exists(CONFIG_FILE)


def test_create_project(project_config_broker):
    result = project_config_broker.create_project(
        project_id=0,
        api_budget=1.0,
        agent_name="Test Agent",
        agent_role="Test Role",
        agent_goals=["Test Goal 1", "Test Goal 2"],
        prompt_generator="Test Prompt Generator",
        command_registry="Test Command Registry",
        overwrite=True,
        project_name="Test Project"
    )
    assert result

    # Test creating a project with the same ID without overwrite
    with pytest.raises(ValueError):
        project_config_broker.create_project(
            project_id=0,
            api_budget=1.0,
            agent_name="Test Agent",
            agent_role="Test Role",
            agent_goals=["Test Goal 1", "Test Goal 2"],
            prompt_generator="Test Prompt Generator",
            command_registry="Test Command Registry",
            overwrite=False,
            project_name="Test Project"
        )

    # Test creating a project with an invalid ID
    with pytest.raises(ValueError):
        project_config_broker.create_project(
            project_id=-1,
            api_budget=1.0,
            agent_name="Test Agent",
            agent_role="Test Role",
            agent_goals=["Test Goal 1", "Test Goal 2"],
            prompt_generator="Test Prompt Generator",
            command_registry="Test Command Registry",
            overwrite=True,
            project_name="Test Project"
        )


def test_set_project_number(project_config_broker):
    result = project_config_broker.set_project_number(new_project_id=0)
    assert result

    # Test setting an invalid project number
    with pytest.raises(ValueError):
        project_config_broker.set_project_number(new_project_id=-1)


def test_get_current_project_id(project_config_broker):
    project_id = project_config_broker.get_current_project_id()
    assert isinstance(project_id, int)


def test_get_current_project(project_config_broker):
    project = project_config_broker.get_current_project()
    assert isinstance(project, Project)


def test_get_project(project_config_broker):
    project = project_config_broker.get_project(project_number=0)
    assert isinstance(project, AgentConfig)

    # Test getting a project with an invalid number
    with pytest.raises(ValueError):
        project_config_broker.get_project(project_number=-1)


def test_get_projects(project_config_broker):
    projects = project_config_broker.get_projects()
    assert isinstance(projects, list)


def test_delete_project(project_config_broker):
    project_count = len(project_config_broker.get_projects())

    # Test deleting a project
    project_config_broker.delete_project(project_number=0)
    new_project_count = len(project_config_broker.get_projects())
    assert new_project_count == project_count - 1

    # Test deleting a project with an invalid number
    with pytest.raises(ValueError):
        project_config_broker.delete_project(project_number=-1)
