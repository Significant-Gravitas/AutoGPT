from app.sdk.errors import *
from AFAAS.app.sdk.forge_log import ForgeLogger
from app.sdk.schema import *
from fastapi import APIRouter, Depends, Query, Request, Response, UploadFile

LOG = ForgeLogger(__name__)
from fastapi import APIRouter, FastAPI, Request

from AFAAS.app.core.agents import PlannerAgent


def get_agent(request: Request, agent_id: str) -> PlannerAgent:
    agent: PlannerAgent = PlannerAgent.get_agent_from_memory(
        agent_id=agent_id,
        user_id=request.state.user_id,
        logger=LOG,
    )
    if agent is None:
        raise NotFoundError
    else:
        return agent
