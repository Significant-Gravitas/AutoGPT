
import json
from typing import Optional

from fastapi import APIRouter, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse

from app.sdk.errors import *
from app.sdk.forge_log import ForgeLogger
from app.sdk.schema import *


LOG = ForgeLogger(__name__)

import uuid
from pathlib import Path
import yaml
from fastapi import APIRouter, FastAPI, Request

from autogpts.autogpt.autogpt.core.agents import PlannerAgent


afaas_user_router = APIRouter()
user_router = APIRouter()

@afaas_user_router.get("/agents", tags=["agent"], response_model=AgentListResponse)
@user_router.get("/agent/tasks", tags=["agent"], response_model=AgentListResponse)
async def list_agents(
    request: Request,
    page: Optional[int] = Query(1, ge=1),
    page_size: Optional[int] = Query(10, ge=1),
) -> AgentListResponse:
    """
    Retrieves a paginated list of all agents.

    Args:
        request (Request): FastAPI request object.
        page (int, optional): The page number for pagination. Defaults to 1.
        page_size (int, optional): The number of agents per page for pagination. Defaults to 10.

    Returns:
        AgentListResponse: A response object containing a list of agents and pagination details.

    Example:
        Request:
            GET /agent/tasks?page=1&pageSize=10

        Response (AgentListResponse defined in schema.py):
            {
                "items": [
                    {
                        "input": "Write the word 'Washington' to a .txt file",
                        "additional_input": null,
                        "task_id": "50da533e-3904-4401-8a07-c49adf88b5eb",
                        "artifacts": [],
                        "steps": []
                    },
                    ...
                ],
                "pagination": {
                    "total": 100,
                    "pages": 10,
                    "current": 1,
                    "pageSize": 10
                }
            }
    """
    LOG.info("Getting agent settings")

    try:
        agents : list[PlannerAgent]= PlannerAgent.get_agentsetting_list_from_memory(
            user_id=request.state.user_id
        ) # (page, page_size)
        return Response(
            content=BaseModel(agents).json(),
            status_code=200,
            media_type="application/json",
        )
    except NotFoundError:
        LOG.exception("Error whilst trying to list agents")
        return Response(
            content=json.dumps({"error": "Agents not found"}),
            status_code=404,
            media_type="application/json",
        )
    except Exception:
        LOG.exception("Error whilst trying to list agents")
        return Response(
            content=json.dumps({"error": "Internal server error"}),
            status_code=500,
            media_type="application/json",
        )


@afaas_user_router.post("/agent", tags=["agent"], response_model=Agent)
@user_router.post("/agent/tasks", tags=["agent"], response_model=Agent)
async def create_agent(request: Request, task_request: AgentRequestBody) -> Agent:
    """
    Creates a new task using the provided AgentRequestBody and returns a Agent.

    Args:
        request (Request): FastAPI request object.
        task (AgentRequestBody): The task request containing input and additional input data.

    Returns:
        Agent: A new task with task_id, input, additional_input, and empty lists for artifacts and stask.

    Example:
        Request (AgentRequestBody defined in schema.py):
            {
                "input": "Write the words you receive to the file 'output.txt'.",
                "additional_input": "python/code"
            }

        Response (Agent defined in schema.py):
            {
                "task_id": "50da533e-3904-4401-8a07-c49adf88b5eb",
                "input": "Write the word 'Washington' to a .txt file",
                "additional_input": "python/code",
                "artifacts": [],
            }
    """
    agent = request["agent"]

    try:
        task_request = await agent.create_task(task_request)
        return Response(
            content=task_request.json(),
            status_code=200,
            media_type="application/json",
        )
    except Exception:
        LOG.exception(f"Error whilst trying to create a task: {task_request}")
        return Response(
            content=json.dumps({"error": "Internal server error"}),
            status_code=500,
            media_type="application/json",
        )


