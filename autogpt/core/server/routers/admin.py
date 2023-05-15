from typing import List

from fastapi import APIRouter, Depends, HTTPException, status

from autogpt.core.server.models.agent import Agent, AgentCreate, AgentUpdate
from autogpt.core.server.models.user import User, UserCreate, UserUpdate
from autogpt.core.server.services import agent_service, user_service
from autogpt.core.server.services.auth_service import get_current_user

router = APIRouter()


def check_admin(current_user):
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Not enough privileges"
        )


@router.get("/users", response_model=List[User])
async def get_all_users(current_user=Depends(get_current_user)):
    check_admin(current_user)
    return await user_service.get_all_users()


@router.get("/users/{user_id}", response_model=User)
async def get_user(user_id: str, current_user=Depends(get_current_user)):
    check_admin(current_user)
    return await user_service.get_user(user_id)


@router.post("/users", response_model=UserCreate)
async def create_user(user: UserCreate, current_user=Depends(get_current_user)):
    check_admin(current_user)
    return await user_service.create_user(user)


@router.put("/users/{user_id}", response_model=UserUpdate)
async def update_user(
    user_id: str, user: UserUpdate, current_user=Depends(get_current_user)
):
    check_admin(current_user)
    return await user_service.update_user(user_id, user)


@router.delete("/users/{user_id}")
async def delete_user(user_id: str, current_user=Depends(get_current_user)):
    check_admin(current_user)
    return await user_service.delete_user(user_id)


@router.get("/agents", response_model=List[Agent])
async def get_all_agents(current_user=Depends(get_current_user)):
    check_admin(current_user)
    return await agent_service.get_all_agents()


@router.get("/agents/{agent_id}", response_model=Agent)
async def get_agent(agent_id: str, current_user=Depends(get_current_user)):
    check_admin(current_user)
    return await agent_service.get_agent(agent_id)


@router.post("/agents", response_model=AgentCreate)
async def create_agent(agent: AgentCreate, current_user=Depends(get_current_user)):
    check_admin(current_user)
    return await agent_service.create_agent(agent)


@router.put("/agents/{agent_id}", response_model=AgentUpdate)
async def update_agent(
    agent_id: str, agent: AgentUpdate, current_user=Depends(get_current_user)
):
    check_admin(current_user)
    return await agent_service.update_agent(agent_id, agent)


@router.delete("/agents/{agent_id}")
async def delete_agent(agent_id: str, current_user=Depends(get_current_user)):
    check_admin(current_user)
    return await agent_service.delete_agent(agent_id)


@router.get("/agents/{agent_id}/details")
async def get_agent_details(agent_id: str, current_user=Depends(get_current_user)):
    check_admin(current_user)
    return await agent_service.get_agent_details(agent_id)


@router.get("/agents/{agent_id}/metrics")
async def get_agent_metrics(agent_id: str, current_user=Depends(get_current_user)):
    check_admin(current_user)
    return await agent_service.get_agent_metrics(agent_id)


@router.get("/agents/{agent_id}/logs")
async def get_agent_logs(agent_id: str, current_user=Depends(get_current_user)):
    check_admin(current_user)
    return await agent_service.get_agent_logs(agent_id)


@router.get("/agents/{agent_id}/costs")
async def get_agent_costs(agent_id: str, current_user=Depends(get_current_user)):
    check_admin(current_user)
    return await agent_service.get_agent_costs(agent_id)


@router.get("/metrics")
async def get_all_metrics(current_user=Depends(get_current_user)):
    check_admin(current_user)
    return await agent_service.get_all_metrics()


@router.get("/costs")
async def get_all_costs(current_user=Depends(get_current_user)):
    check_admin(current_user)
    return await agent_service.get_all_costs()
