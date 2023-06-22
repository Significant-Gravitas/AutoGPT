from datetime import datetime
from typing import Optional

from fastapi import APIRouter

# There is a catch with dynamic creation of agents based on json request
# All classes need to have been imported to be able to be found this needs
# to be looked at again to see if we can find a cleaner way to do this
from autogpt.core.agent.sample_agent import SampleAgent
from autogpt.core.messaging.queue_channel import QueueChannel
from autogpt.core.server.models.agent import (
    AgentDefinition,
    AgentListResponse,
    StartAgentResponse,
    StopAgentReq,
    StopAgentResponse,
)
from autogpt.core.server.services import agent_service

router = APIRouter()

channel = QueueChannel(id="channel1", name="channel1", host="localhost", port=8080)
agents = []


@router.post("/start", response_model=StartAgentResponse)
async def start_agent(agent_definition: AgentDefinition) -> StartAgentResponse:
    """Create an agent from a json definition of the agent"""
    print(agent_definition.data)
    agent = SampleAgent.parse_raw(agent_definition.data)

    task = await agent_service.create_agent(agent, channel)
    agents.append(
        {
            "name": agent.name,
            "id": agent.uid,
            "session_id": agent.session_id,
            "task": task,
        }
    )

    return StartAgentResponse(
        name=agent.name, id=agent.uid, session_id=agent.session_id
    )


@router.post("/stop", response_model=StopAgentResponse)
async def stop_agent(stop_request: StopAgentReq) -> StopAgentResponse:
    """Stops an agent"""
    task = None
    for agent in agents:
        if agent["id"] == stop_request.agent_id:
            task = agent["task"]
            remove_running_agent = agent

    response = await agent_service.stop_agent(
        stop_request=stop_request, message_queue=channel, task=task
    )

    agents.remove(remove_running_agent)

    return response


@router.get("/list", response_model=AgentListResponse)
async def list_agents():
    """Lists all running agents"""
    return AgentListResponse(agents=agents)


@router.get("/{id}/thoughts")
async def listen_to_agent(
    id: str, start_thought_id: int, end_thought_id: Optional[datetime] = None
):
    agent = [agent for agent in agents if agent.id == id]
    thoughts = []
    # Add your code here to get the agent's thoughts
    return {"agent": agent, "thoughts": thoughts}


@router.get("/{id}/step/{session_id}")
async def run_agent_step(id: str, session_id: str):
    agent = [agent for agent in agents if agent.id == id]
    thoughts = []
    # Add your code here to run the agent step
    return {"agent": agent, "thoughts": thoughts}
