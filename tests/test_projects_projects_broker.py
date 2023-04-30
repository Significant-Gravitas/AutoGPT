"""
This module contains tests for the `ProjectsBroker` class in the `autogpt` package. The `ProjectsBroker` class manages the creation, loading, and saving of `Project` objects.

The tests in this module cover various functionalities of the `ProjectsBroker` class, including:

1. Initializing a `ProjectsBroker` object
2. Loading saved projects
3. Creating and saving a new project
4. Getting a project by ID
5. Getting all projects
6. Setting and getting the current project
7. Formatting a project directory name

The tests use `pytest` as the testing framework and define several fixtures that are used in the tests, including the `ProjectsBroker` object itself. 

Data for the tests:

- `test_agent`: An instance of the `AIConfig` class with goals, name, role, and budget information for a project.
- `test_agent2`: An instance of the `AIConfig` class with goals, name, role, and budget information for a project.
- `test_agent3`: An instance of the `AIConfig` class with goals, name, role, and budget information for a project.
- `test_agent4`: An instance of the `AIConfig` class with goals, name, role, and budget information for a project.

Tests:
- `test_init`: Tests that a `ProjectsBroker` object can be initialized.
- `test_load`: Tests that saved projects can be loaded.
- `test_save`: Tests that a new project can be created and saved.
- `test_create_project`: Tests that a new project can be created.
- `test_project_dir_name_formater`: Tests that the project directory name is correctly formatted.
- `test_set_project_position_number`: Tests that the current project can be set by ID.
- `test_get_current_project_id`: Tests that the ID of the current project can be retrieved.
- `test_get_current_project`: Tests that the current project can be retrieved.
- `test_get_project`: Tests that a project can be retrieved by ID.
- `test_get_projects`: Tests that all projects can be retrieved.
"""

import pytest
import os
from autogpt.projects.projects_broker import ProjectsBroker
from autogpt.projects.agent_model import AgentModel
from autogpt.projects.project import Project
from autogpt.config.ai_config import AIConfig

ai_goals =  ['Goal 1: Make a sandwich','Goal 2, Eat the sandwich','Goal 3 - Go to sleep', 'Goal 4: Wake up']
ai_name= 'McFamished',
ai_role= 'A hungry AI',
api_budget= 10.0

test_agent =  AIConfig( ai_goals = ai_goals,
                        ai_name= ai_name,
                        ai_role= ai_role,
                        api_budget= api_budget
                    )

ai_goals2 = ['Goal 1: Peel the apple', 'Goal 2: Slice the apple', 'Goal 3: Eat the apple', 'Goal 4: Drink water']
ai_name2 = 'FruityBot'
ai_role2 = 'A fruit-loving AI'
api_budget2 = 3.5
test_agent2 =  AIConfig( ai_goals = ai_goals2,
                        ai_name= ai_name2,
                        ai_role= ai_role2,
                        api_budget= api_budget2
                    )

ai_goals3 = ['Goal 1: Pick the ripe mango', 'Goal 2: Cut the mango into pieces', 'Goal 3: Blend the mango into a smoothie', 'Goal 4: Drink the mango smoothie']
ai_name3 = 'MangoMaster'
ai_role3 = 'An AI with a taste for mangoes'
api_budget3 = 3.0
test_agent3 =  AIConfig( ai_goals = ai_goals3,
                        ai_name= ai_name3,
                        ai_role= ai_role3,
                        api_budget= api_budget3
                    )
test_agent4 =  AIConfig( ai_goals = ai_goals3,
                        ai_name= ai_name3,
                        ai_role= ai_role3,
                        api_budget= api_budget3
                    )


@pytest.fixture
def projects_broker():
    return ProjectsBroker()


def test_init(projects_broker : ProjectsBroker):
    assert isinstance(projects_broker, ProjectsBroker)


def test_load(projects_broker : ProjectsBroker):
    projects = projects_broker.load()
    assert isinstance(projects, list)


def test_save(projects_broker : ProjectsBroker):
    # create a new project
    lead_agent = test_agent
    project = projects_broker.create_project(
        project_position_number=0,
        lead_agent=lead_agent,
        project_budget=100.0,
        project_name="test_project",
        version="0.0.1"
    )
    # save the project
    projects_broker._save(project_position_number=0)
    # check if the project file exists
    assert os.path.exists(f"{ProjectsBroker.PROJECT_DIR}/test_project/settings.yaml")

def test_create_project(projects_broker : ProjectsBroker):
    lead_agent = test_agent
    project = projects_broker.create_project(
        project_position_number=0,
        lead_agent=lead_agent,
        project_budget=100.0,
        project_name="test_project",
        version="0.0.1"
    )
    assert isinstance(project, Project)


def test_project_dir_name_formater():
    assert ProjectsBroker.project_dir_name_formater("Test Project") == "testproject"


def test_set_project_position_number(projects_broker : ProjectsBroker):
    # create a new project
    lead_agent = test_agent
    project = projects_broker.create_project(
        project_position_number=0,
        lead_agent=lead_agent,
        project_budget=100.0,
        project_name="test_project",
        version="0.0.1"
    )
    # set the project position number to 0
    projects_broker.set_project_position_number(new_project_id=0)
    # check if the current project is the newly created project
    assert projects_broker.get_current_project() == project


def test_get_current_project_id(projects_broker : ProjectsBroker):
    # create a new project and set it as the current project
    lead_agent = test_agent
    project = projects_broker.create_project(
        project_position_number=0,
        lead_agent=lead_agent,
        project_budget=100.0,
        project_name="test_project",
        version="0.0.1"
    )
    projects_broker.set_project_position_number(new_project_id=0)
    # check if the current project ID is 0
    assert projects_broker.get_current_project_id() == 0


def test_get_current_project(projects_broker : ProjectsBroker):
    # create a new project and set it as the current project
    lead_agent = test_agent
    project = projects_broker.create_project(
        project_position_number=0,
        lead_agent=lead_agent,
        project_budget=100.0,
        project_name="test_project",
        version="0.0.1"
    )
    projects_broker.set_project_position_number(new_project_id=0)
    # check if the current project is the newly created project
    assert projects_broker.get_current_project() == project


def test_get_project(projects_broker : ProjectsBroker):
    # create a new project
    lead_agent = test_agent
    project = projects_broker.create_project(
        project_position_number=0,
        lead_agent=lead_agent,
        project_budget=100.0,
        project_name="test_project",
        version="0.0.1"
    )
    # get the newly created project
    assert projects_broker.get_project(project_position_number=0) == project


def test_get_projects(projects_broker : ProjectsBroker):
    # create two new projects
    lead_agent1 = test_agent
    project1 = projects_broker.create_project(
        project_position_number=0,
        lead_agent=lead_agent1,
        project_budget=100.0,
        project_name="test_project1",
        version="0.0.1"
    )
    lead_agent2 = test_agent2
    project2 = projects_broker.create_project(
        project_position_number=1,
        lead_agent=lead_agent2,
        project_budget=200.0,
        project_name="test_project2",
        version="0.0.2"
    )
    # get all projects
    projects = projects_broker.get_projects()
    # check if both projects are in the list
    assert project1 in projects and project2 in projects

def test_create_project(projects_broker : ProjectsBroker):
    lead_agent = test_agent
    project = projects_broker.create_project(
        project_position_number=0,
        lead_agent=lead_agent,
        project_budget=100.0,
        project_name="test_project",
        version="0.0.1"
    )
    assert isinstance(project, Project)


def test_project_dir_name_formater():
    assert ProjectsBroker.project_dir_name_formater("Test Project") == "testproject"


def test_set_project_position_number(projects_broker : ProjectsBroker):
    # create a new project
    lead_agent = test_agent
    project = projects_broker.create_project(
        project_position_number=0,
        lead_agent=lead_agent,
        project_budget=100.0,
        project_name="test_project",
        version="0.0.1"
    )
    # set the project position number to 0
    projects_broker.set_project_position_number(new_project_id=0)
    # check if the current project is the newly created project
    assert projects_broker.get_current_project() == project


def test_get_current_project_id(projects_broker : ProjectsBroker):
    # create a new project and set it as the current project
    lead_agent = test_agent
    project = projects_broker.create_project(
        project_position_number=0,
        lead_agent=lead_agent,
        project_budget=100.0,
        project_name="test_project",
        version="0.0.1"
    )
    projects_broker.set_project_position_number(new_project_id=0)
    # check if the current project ID is 0
    assert projects_broker.get_current_project_id() == 0


def test_get_current_project(projects_broker : ProjectsBroker):
    # create a new project and set it as the current project
    lead_agent = test_agent
    project = projects_broker.create_project(
        project_position_number=0,
        lead_agent=lead_agent,
        project_budget=100.0,
        project_name="test_project",
        version="0.0.1"
    )
    projects_broker.set_project_position_number(new_project_id=0)
    # check if the current project is the newly created project
    assert projects_broker.get_current_project() == project


def test_get_project(projects_broker : ProjectsBroker):
    # create a new project
    lead_agent = test_agent
    project = projects_broker.create_project(
        project_position_number=0,
        lead_agent=lead_agent,
        project_budget=100.0,
        project_name="test_project",
        version="0.0.1"
    )
    # get the newly created project
    assert projects_broker.get_project(project_position_number=0) == project


def test_get_projects(projects_broker : ProjectsBroker):
    # create two new projects
    lead_agent1 = test_agent
    project1 = projects_broker.create_project(
        project_position_number=0,
        lead_agent=lead_agent1,
        project_budget=100.0,
        project_name="test_project1",
        version="0.0.1"
    )
    lead_agent2 = test_agent2
    project2 = projects_broker.create_project(
        project_position_number=1,
        lead_agent=lead_agent2,
        project_budget=200.0,
        project_name="test_project2",
        version="0.0.2"
    )
    # get all projects
    projects = projects_broker.get_projects()
    # check if both projects are in the list
    assert project1 in projects and project2 in projects
