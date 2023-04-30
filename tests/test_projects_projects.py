import pytest
from autogpt.projects.agent_model import AgentModel
from autogpt.projects.project import Project
from autogpt.config.ai_config import AIConfig
import yaml


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

test_agent5 = AIConfig(
ai_goals = ['Goal 1: Create a personalized workout plan', 'Goal 2: Track progress towards fitness goals', 'Goal 3: Provide healthy meal suggestions', 'Goal 4: Motivate users to stay on track'],
ai_name= 'FitAI',
ai_role= 'An AI dedicated to fitness and health',
api_budget= 5.0
)

@pytest.fixture()
def project():
    lead_agent = test_agent #,AgentModel("McFamished", "openai", "default")
    delegated_agents = [test_agent2, test_agent3]
    #delegated_agents = [AgentModel("ai_name2", "openai", "default"), AgentModel("davinci", "openai", "default")]
    my_project = Project("My Project", 1000.0, lead_agent, delegated_agents)
    return my_project

def test_project_to_dict(project : Project):
    project_dict = project.to_dict()
    assert isinstance(project_dict, dict)
    assert project_dict["project_name"] == "My Project"
    assert project_dict["project_budget"] == 1000.0

def test_project_load(project : Project):
    project_dict = project.to_dict()
    loaded_project = Project.load(project_dict)
    assert isinstance(loaded_project, Project)
    assert loaded_project.project_name == "My Project"
    assert loaded_project.project_budget == 1000.0
    assert isinstance(loaded_project.get_lead(), AgentModel)
    assert isinstance(loaded_project.get_delegated_agents()[0], AgentModel)

def test_project_save(project, tmpdir):
    project_dict = project.to_dict()
    project_dir = tmpdir.mkdir("My Project")
    project_file = project_dir.join("settings.yaml")
    project.save(str(project_file))
    assert project_file.exists()
    with open(str(project_file), "r") as f:
        saved_project_dict = yaml.safe_load(f)
    assert saved_project_dict == project_dict

def test_project_get_lead(project : Project):
    lead_agent = project.get_lead()
    assert isinstance(lead_agent, AgentModel)
    assert lead_agent.name == "McFamished"

def test_project_get_delegated_agents(project : Project):
    delegated_agents = project.get_delegated_agents()
    assert isinstance(delegated_agents, list)
    assert len(delegated_agents) == 2
    assert isinstance(delegated_agents[0], AgentModel)
    assert delegated_agents[0].name == ai_name2


""" # TODO
def test_project_delete_delegated_agents(project : Project):
    success = project.delete_delegated_agent(0)
    assert success
    delegated_agents = project.get_delegated_agents()
    assert len(delegated_agents) == 1
    assert delegated_agents[0].name == "davinci"

def test_project_add_delegated_agent(project : Project):
    new_agent = test_agent4 #AgentModel("McFamished-codex", "openai", "default")
    project.add_delegated_agent(new_agent)
    delegated_agents = project.get_delegated_agents()
    assert len(delegated_agents) == 3
    assert delegated_agents[-1].name == "MangoMaster"
"""
