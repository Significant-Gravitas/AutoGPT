from AFAAS.lib.sdk.errors import *
from AFAAS.lib.sdk.logger import AFAASLogger
from AFAAS.lib.sdk.schema import *
from fastapi import APIRouter, Depends, Query, Request, Response, UploadFile

LOG = AFAASLogger(name=__name__)
from fastapi import APIRouter, FastAPI, Request

from AFAAS.core.agents import PlannerAgent


def get_agent(request: Request, agent_id: str) -> PlannerAgent:
    agent: PlannerAgent = PlannerAgent.get_agent_from_memory(
        agent_id=agent_id, user_id=request.state.user_id
    )
    if agent is None:
        raise NotFoundError
    else:
        return agent
