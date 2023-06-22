from fastapi import APIRouter, Depends, HTTPException

from autogpt.core.server.models.interaction import Interaction
from autogpt.core.server.models.session import SessionCreate, SessionUpdate
from autogpt.core.server.services import agent_service
from autogpt.core.server.services.auth_service import get_current_user

router = APIRouter()


@router.post("/{agent_id}/start", response_model=str)
async def start_agebt(agent_id: str, current_user=Depends(get_current_user)):
    return await agent_service.start_session(agent_id, current_user)


@router.get("/{agent_id}/sessions/{session_id}")
async def get_session(
    agent_id: str, session_id: str, current_user=Depends(get_current_user)
):
    return await agent_service.get_session(agent_id, session_id, current_user)


@router.post(
    "/{agent_id}/sessions/{session_id}/interactions", response_model=Interaction
)
async def interact_with_agent(
    agent_id: str,
    session_id: str,
    interaction: Interaction,
    current_user=Depends(get_current_user),
):
    return await agent_service.interact_with_agent(
        agent_id, session_id, interaction, current_user
    )


@router.put("/{agent_id}/sessions/{session_id}", response_model=SessionUpdate)
async def update_session(
    agent_id: str,
    session_id: str,
    session_update: SessionUpdate,
    current_user=Depends(get_current_user),
):
    return await agent_service.update_session(
        agent_id, session_id, session_update, current_user
    )


@router.delete("/{agent_id}/sessions/{session_id}")
async def end_session(
    agent_id: str, session_id: str, current_user=Depends(get_current_user)
):
    return await agent_service.end_session(agent_id, session_id, current_user)
