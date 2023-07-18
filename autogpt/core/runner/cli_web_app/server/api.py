import uuid
from pathlib import Path
import yaml
from fastapi import APIRouter, FastAPI, Request
from autogpt.core.runner.client_lib.shared_click_commands import (
    DEFAULT_SETTINGS_FILE,
    make_settings,
)

from autogpt.core.runner.cli_web_app.server.schema import SimpleAgentMessageRequestBody
from autogpt.core.runner.client_lib.workspacebuilder import workspace_loader , get_settings_from_file, get_logger_and_workspace
from autogpt.core.agent import AgentSettings, SimpleAgent
from autogpt.core.runner.client_lib.parser import parse_agent_name_and_goals , parse_ability_result , parse_agent_plan , parse_next_ability



router = APIRouter()

@router.get("/agents")
async def get_agents(request: Request):
    """
    Description: Returns a list of all agents.
    Purpose: Having a list of agents/chat is crucial for a web UI.
    Prerequisites:
    Agents need a UUID.
    Agents need a name.
    Ideally, agents should have a "status" property (Active, Inactive, Deleted) to avoid concurrent access.
    Ideally, agents should have a boolean property ("retrievable", "front", or "clientfacing") to exclude sub-agents. Default to False, but SimpleAgents should have it set to "True."
    Comment: Ideally returns all client_facing, active and inactive agents.
    """
    settings = get_settings_from_file()
    client_logger, agent_workspace = get_logger_and_workspace(settings)
    # workspace =  workspace_loader(settings, client_logger, agent_workspace)
    agent = {}
    if agent_workspace : 
        agent = SimpleAgent.from_workspace(
            agent_workspace, 
            client_logger,
            )
        agent = agent.__dict__
        # @Todo place holder for elements
        # @Todo filter on client_facing
        # @Todo filter on inactive
        agent['agent_id' ]= uuid.uuid4().hex
        agent['client_facing'] = True
        agent['status'] = 0
    
    # Mock a list of agent with a lenght of 1 instead of returning an agent
    return {"agents": [agent]}


@router.post("/agent")
async def create_agent(request: Request):
    """
    Create a new agent.
    Minimal requirements : None 
    """
    agent_id = uuid.uuid4().hex
    return {"agent_id": agent_id}


@router.get("/agent/{agent_id}")
async def get_agent_by_id(request: Request, agent_id: str):
    """
    Get an agent from it's ID & return an agent
    """
    settings = get_settings_from_file()
    client_logger, agent_workspace = get_logger_and_workspace(settings)
    
   # workspace =  workspace_loader(settings, client_logger, agent_workspace)
    agent = {}
    if agent_workspace : 
        agent = SimpleAgent.from_workspace(
            agent_workspace,
            client_logger,
            )
        agent = agent.__dict__
        # @Todo place holder for elements
        # @Todo filter on client_facing
        # @Todo filter on inactive
        agent['agent_id' ]= uuid.uuid4().hex
        agent['client_facing'] = True
        agent['status'] = 0

    return {"agent": agent}


@router.post("/agent/{agent_id}/start")
async def start_simple_agent_main_loop(request: Request, agent_id: str):
    """
    Senf a message to an agent 
    """
    user_configuration = get_settings_from_file()
    # Get the logger and the workspace movec to a function
    # Because almost every API end-point will excecute this piece of code
    client_logger, agent_workspace = get_logger_and_workspace(user_configuration)

    # Get the logger and the workspace moved to a function
    # Because API end-point will treat this as an error & break
    if not agent_workspace : 
        raise BaseException

    # launch agent interaction loop
    agent = SimpleAgent.from_workspace(
        agent_workspace,
        client_logger,
    )

    plan = await agent.build_initial_plan()

    current_task, next_ability = await agent.determine_next_ability(plan)

    return {
        'plan' : plan.__dict__,
        'current_task' : current_task.__dict__ , 
        'next_ability' : next_ability.__dict__
        }


@router.post("/agent/{agent_id}/message")
async def message_simple_agent(request: Request, agent_id: str, body: SimpleAgentMessageRequestBody):
    """
    Description: Sends a message to the agent.
    Arg : 
    - Required body.message  (str): The message sent from the client
    - Optional body.start (bool): Start a new loop
    Comment: Only works if the agent is inactive. Works for both client-facing = true and false, enabling sub-agents.
    """
    user_configuration = get_settings_from_file()
    # Get the logger and the workspace movec to a function
    # Because almost every API end-point will excecute this piece of code
    client_logger, agent_workspace = get_logger_and_workspace(user_configuration)

    # Get the logger and the workspace moved to a function
    # Because API end-point will treat this as an error & break
    if not agent_workspace : 
        raise BaseException

    # launch agent interaction loop
    agent = SimpleAgent.from_workspace(
        agent_workspace,
        client_logger,
    )

    plan = await agent.build_initial_plan()
    print(parse_agent_plan(plan))

    current_task, next_ability = await agent.determine_next_ability(plan)
    
    result = {}
    ability_result = await agent.execute_next_ability(body.message)
    result["ability_result"] = ability_result.__dict__

    if (body.start == True ) : 
            result['current_task'], result['next_ability'] = await agent.determine_next_ability(plan)

    return result

app = FastAPI()
app.include_router(router, prefix="/api/v1")
