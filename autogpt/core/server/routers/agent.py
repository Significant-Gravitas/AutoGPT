from typing import Optional
from datetime import datetime
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException

from autogpt.core.server.services import agent_service



from autogpt.core.agent import SampleAgent
from autogpt.core.LLM.openai_povider import OpenAIProvider 
from autogpt.core.messaging.message_broker import MessageBroker

import json

router = APIRouter()


class AgentDefinition(BaseModel):
    data: str

class StartAgentResponse(BaseModel):
    name: str
    id: str
    session_id: str

agents = []


@router.post("/start", response_model=StartAgentResponse)
async def start_agent(agent_definition: AgentDefinition):
    print(agent_definition.data)
    # agent_def_dict = json.loads(agent_definition.data)  # convert the json string to a python dictionary

    agent = SampleAgent.parse_raw(agent_definition.data)
    return StartAgentResponse(name=agent.name, id=agent.uid, session_id=agent.session_id)



@router.post("/stop/{id}")
async def stop_agent(id: str, kill: Optional[bool] = False):
    agent = [agent for agent in agents if agent.id == id]
    if kill:
        # Add your code here to kill the agent
        pass
    else:
        # Add your code here to stop the agent
        pass
    return {"message": "Agent stopped"}

@router.get("/list")
async def list_agents():
    return agents

@router.get("/{id}/thoughts")
async def listen_to_agent(id: str, from_timestamp: datetime, to_timestamp: Optional[datetime] = None):
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
