"""
/agents (GET): Returns a list of all agents.
AP alias /agent/tasks (GET): Returns a list of all agents.
    => Need Agent (Parent) to define this properties : agent_id, client_facing, status

    
/agent (POST): Create a new agent.
AP alias /agent/tasks (POST): Create a new agent.
    => TODO : Save an Agent (or PlannerAgent) in the workspace (can't find the method still reading the code)

    
/agent/{agent_id} (GET): Get an agent from its ID & return an agent.
AP alias /agent/tasks/{agent_id} (GET): Get an agent from its ID & return an agent.


/agent/{agent_id}/start (POST): Send a message to an agent.
/agent/{agent_id}/message (POST): Sends a message to the agent.
/agent/{agent_id}/messagehistory (GET): Get message history for an agent.
    => need PlannerAgent to provide a method 
/agent/{agent_id}/lastmessage (GET): Get the last message for an agent.
    => need PlannerAgent to provide a method 
"""
import uuid
from pathlib import Path
import yaml
from fastapi import APIRouter, FastAPI, Request
from autogpts.autogpt.autogpt.core.runner.client_lib.shared_click_commands import (
    DEFAULT_SETTINGS_FILE,
    make_settings,
)

from autogpts.autogpt.autogpt.core.runner.cli_web_app.server.schema import AgentMessageRequestBody
from autogpts.autogpt.autogpt.core.runner.client_lib.workspacebuilder import (
    workspace_loader,
    get_settings_from_file,
    get_logger_and_workspace,
)
from autogpts.autogpt.autogpt.core.agents import PlannerAgent
from autogpts.autogpt.autogpt.core.runner.client_lib.parser import (
    parse_agent_name_and_goals,
    parse_ability_result,
    parse_agent_plan,
    parse_next_tool,
)


router = APIRouter()

# def get_settings_logger_workspace():
#     settings = get_settings_from_file()
#     client_logger, agent_workspace = get_logger_and_workspace(settings)
#     return settings, client_logger, agent_workspace


@router.get("/agents")
@router.get("/agent/tasks")
async def get_agents(request: Request):
    """
    Description: Returns a list of all agents.
    Purpose: Having a list of agents/chat is crucial for a web UI.
    Prerequisites:
    Agents need a UUID.
    Agents need a name.
    Ideally, agents should have a "status" property (Active, Inactive, Deleted) to avoid concurrent access.
    Ideally, agents should have a boolean property ("retrievable", "front", or "clientfacing") to exclude sub-agents. Default to False, but PlannerAgents should have it set to "True."
    Comment: Ideally returns all client_facing, active and inactive agents.
    """
    settings = get_settings_from_file()
    client_logger, agent_workspace = get_logger_and_workspace(settings)
    # workspace =  workspace_loader(settings, client_logger, agent_workspace)
    agent = {}
    if agent_workspace:
        agent = PlannerAgent.from_workspace(
            agent_workspace,
            client_logger,
        )
        agent = agent.__dict__
        # TODO place holder for elements
        # TODO filter on client_facing
        # TODO filter on inactive
        agent["agent_id"] = uuid.uuid4().hex
        agent["client_facing"] = True
        agent["status"] = 0

    # Mock a list of agent with a lenght of 1 instead of returning an agent
    return {"agents": [agent]}


@router.post("/agent")
@router.post("/agent/tasks")
async def create_agent(request: Request):
    """
    Create a new agent.
    Minimal requirements : None
    """
    agent_id = uuid.uuid4().hex

    # TODO : Save the Agent, idealy managed in Agent() not Simple Agent.
    return {"agent_id": agent_id}


@router.get("/agent/{agent_id}")
@router.get("/agent/tasks/{agent_id}")
async def get_agent_by_id(request: Request, agent_id: str):
    """
    Get an agent from it's ID & return an agent
    """
    settings = get_settings_from_file()
    client_logger, agent_workspace = get_logger_and_workspace(settings)

    # workspace =  workspace_loader(settings, client_logger, agent_workspace)
    agent = {}
    if agent_workspace:
        agent = PlannerAgent.from_workspace(
            agent_workspace,
            client_logger,
        )
        agent = agent.__dict__
        # TODO place holder for elements
        # TODO filter on client_facing
        # TODO filter on inactive
        agent["agent_id"] = uuid.uuid4().hex
        agent["client_facing"] = True
        agent["status"] = 0

    return {"agent": agent}


@router.post("/agent/{agent_id}/start")
async def start_simple_agent_main_loop(request: Request, agent_id: str):
    """
    Senf a message to an agent
    """

    # Get the logger and the workspace movec to a function
    # Because almost every API end-point will excecute this piece of code
    user_configuration = get_settings_from_file()
    client_logger, agent_workspace = get_logger_and_workspace(user_configuration)

    # Get the logger and the workspace moved to a function
    # Because API end-point will treat this as an error & break
    if not agent_workspace:
        raise BaseException

    # launch agent interaction loop
    agent = PlannerAgent.from_workspace(
        agent_workspace,
        client_logger,
    )

    plan = await agent.build_initial_plan()

    current_task, next_ability = await agent.determine_next_ability(plan)

    return {
        "plan": plan.__dict__,
        "current_task": current_task.__dict__,
        "next_ability": next_ability.__dict__,
    }


@router.post("/agent/{agent_id}/message")
async def message_simple_agent(
    request: Request, agent_id: str, body: AgentMessageRequestBody
):
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
    if not agent_workspace:
        print(f"request.body: {request.body}")
        print(f"agent_id: {agent_id}")
        print(f"body: {body}")
        raise BaseException

    # launch agent interaction loop
    agent = PlannerAgent.from_workspace(
        agent_workspace,
        client_logger,
    )

    plan = await agent.build_initial_plan()

    current_task, next_ability = await agent.determine_next_ability(plan)

    result = {}
    ability_result = await agent.execute_next_ability(body.message)
    result["ability_result"] = ability_result.__dict__

    if body.start == True:
        (
            result["current_task"],
            result["next_ability"],
        ) = await agent.determine_next_ability(plan)

    return result


@router.get("/agent/{agent_id}/messagehistory")
def get_message_history(request: Request, agent_id: str):
    # TODO : Define structure of the list
    return {"messages": ["message 1", "message 2", "message 3", "message 4"]}


@router.get("/agent/{agent_id}/lastmessage")
def get_last_message(request: Request, agent_id: str):
    return {"message": "my last message"}


app = FastAPI()
app.include_router(router, prefix="/api/v1")
