from typing import List

from fastapi import APIRouter, Depends

from autogpt.core.server.models.agent import Agent, AgentCreate, AgentUpdate
from autogpt.core.server.models.user import User, UserCreate, UserUpdate
from autogpt.core.server.services import agent_service, user_service
from autogpt.core.server.services.auth_service import get_current_user

router = APIRouter()


@router.get("/{user_id}", response_model=User)
async def get_user(user_id: str, current_user=Depends(get_current_user)):
    return await user_service.get_user(user_id)


@router.post("/", response_model=UserCreate)
async def create_user(user: UserCreate, current_user=Depends(get_current_user)):
    return await user_service.create_user(user)


@router.put("/{user_id}", response_model=UserUpdate)
async def update_user(
    user_id: str, user: UserUpdate, current_user=Depends(get_current_user)
):
    return await user_service.update_user(user_id, user)


@router.delete("/{user_id}")
async def delete_user(user_id: str, current_user=Depends(get_current_user)):
    return await user_service.delete_user(user_id)


## User's Agents Management


@router.get("/{user_id}/agents", response_model=List[Agent])
async def get_user_agents(user_id: str, current_user=Depends(get_current_user)):
    return await agent_service.get_user_agents(user_id)


@router.get("/{user_id}/agents/{agent_id}", response_model=Agent)
async def get_user_agent(
    user_id: str, agent_id: str, current_user=Depends(get_current_user)
):
    return await agent_service.get_user_agent(user_id, agent_id)


@router.post("/{user_id}/agents", response_model=AgentCreate)
async def create_user_agent(
    user_id: str, agent: AgentCreate, current_user=Depends(get_current_user)
):
    return await agent_service.create_user_agent(user_id, agent)


@router.put("/{user_id}/agents/{agent_id}", response_model=AgentUpdate)
async def update_user_agent(
    user_id: str,
    agent_id: str,
    agent: AgentUpdate,
    current_user=Depends(get_current_user),
):
    return await agent_service.update_user_agent(user_id, agent_id, agent)


@router.delete("/{user_id}/agents/{agent_id}")
async def delete_user_agent(
    user_id: str, agent_id: str, current_user=Depends(get_current_user)
):
    return await agent_service.delete_user_agent(user_id, agent_id)


## User-Agent Detailed Information and Metrics


@router.get("/{user_id}/agents/{agent_id}/details")
async def get_user_agent_details(
    user_id: str, agent_id: str, current_user=Depends(get_current_user)
):
    return await agent_service.get_user_agent_details(user_id, agent_id)


@router.get("/{user_id}/agents/{agent_id}/metrics")
async def get_user_agent_metrics(
    user_id: str, agent_id: str, current_user=Depends(get_current_user)
):
    return await agent_service.get_user_agent_metrics(user_id, agent_id)


@router.get("/{user_id}/agents/{agent_id}/logs")
async def get_user_agent_logs(
    user_id: str, agent_id: str, current_user=Depends(get_current_user)
):
    return await agent_service.get_user_agent_logs(user_id, agent_id)


@router.get("/{user_id}/agents/{agent_id}/costs")
async def get_user_agent_costs(
    user_id: str, agent_id: str, current_user=Depends(get_current_user)
):
    return await agent_service.get_user_agent_costs(user_id, agent_id)
