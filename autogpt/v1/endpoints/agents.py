import sys
sys.path.append("scripts/")
from starlette.responses import JSONResponse
from typing import Dict
from fastapi import APIRouter
from scripts.ai_config import AIConfig

router = APIRouter()

@router.post("/agents")
async def create_agent(body: Dict): # TODO pydantify this when we have a clear definition of the request pattern.
    agent = AIConfig(
        ai_name=body["ai_name"],
        ai_role=body["ai_role"],
        ai_goals=["ai_goals"])
    agent.save()

    return JSONResponse(
        content={"message": "Agent created", "id": 1})  # only one main agent is supported for now
