import json
import os
from io import BytesIO
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Body, Depends, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from AFAAS.core.agents.planner.main import PlannerAgent
from AFAAS.lib.sdk.artifacts import Artifact
from AFAAS.lib.sdk.errors import *
from AFAAS.lib.sdk.logger import AFAASLogger
from AFAAS.lib.sdk.schema import *

from .dependencies.agents import get_agent

afaas_artifact_router = APIRouter()
artifact_router = APIRouter()

LOG = AFAASLogger(name=__name__)


@afaas_artifact_router.get(
    "/agent/{agent_id}/artifacts",
    tags=["artifacts"],
    response_model=AgentArtifactsListResponse,
)
@artifact_router.get(
    "/agent/tasks/{agent_id}/artifacts",
    tags=["artifacts"],
    response_model=AgentArtifactsListResponse,
)
async def list_agent_artifacts(
    request: Request,
    agent_id: str,
    page: Optional[int] = Query(1, ge=1),
    page_size: Optional[int] = Query(10, ge=1, alias="pageSize"),
    agent: PlannerAgent = Depends(get_agent),
) -> AgentArtifactsListResponse:
    """
    Retrieves a paginated list of artifacts associated with a specific task.

    Args:
        request (Request): FastAPI request object.
        agent_id (str): The ID of the task.
        page (int, optional): The page number for pagination. Defaults to 1.
        page_size (int, optional): The number of items per page for pagination. Defaults to 10.

    Returns:
        AgentArtifactsListResponse: A response object containing a list of artifacts and pagination details.

    Example:
        Request:
            GET /agent/tasks/50da533e-3904-4401-8a07-c49adf88b5eb/artifacts?page=1&pageSize=10

        Response (AgentArtifactsListResponse defined in schema.py):
            {
                "items": [
                    {"artifact_id": "artifact1_id", ...},
                    {"artifact_id": "artifact2_id", ...},
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
    request["agent"]
    try:
        artifacts: AgentArtifactsListResponse = await list_artifacts(
            agent_id, page, page_size
        )
        return artifacts
    except NotFoundError:
        LOG.exception("Error whilst trying to list artifacts")
        return Response(
            content=json.dumps({"error": "Artifacts not found for agent_id"}),
            status_code=404,
            media_type="application/json",
        )
    except Exception:
        LOG.exception("Error whilst trying to list artifacts")
        return Response(
            content=json.dumps({"error": "Internal server error"}),
            status_code=500,
            media_type="application/json",
        )


@afaas_artifact_router.post(
    "/agent/{agent_id}/artifacts",
    tags=["artifacts"],
    response_model=Artifact,
)
@artifact_router.post(
    "/agent/tasks/{agent_id}/artifacts", tags=["artifacts"], response_model=Artifact
)
async def upload_agentartifacts(
    request: Request,
    agent_id: str,
    file: UploadFile = Body(...),
    relative_path: Optional[str] = "",
    agent: PlannerAgent = Depends(get_agent),
) -> Artifact:
    """
    This endpoint is used to upload an artifact associated with a specific task. The artifact is provided as a file.

    Args:
        request (Request): The FastAPI request object.
        agent_id (str): The unique identifier of the task for which the artifact is being uploaded.
        file (UploadFile): The file being uploaded as an artifact.
        relative_path (str): The relative path for the file. This is a query parameter.

    Returns:
        Artifact: An object containing metadata of the uploaded artifact, including its unique identifier.

    Example:
        Request:
            POST /agent/tasks/50da533e-3904-4401-8a07-c49adf88b5eb/artifacts?relative_path=my_folder/my_other_folder
            File: <uploaded_file>

        Response:
            {
                "artifact_id": "b225e278-8b4c-4f99-a696-8facf19f0e56",
                "created_at": "2023-01-01T00:00:00Z",
                "modified_at": "2023-01-01T00:00:00Z",
                "agent_created": false,
                "relative_path": "/my_folder/my_other_folder/",
                "file_name": "main.py"
            }
    """
    request["agent"]

    if file is None:
        return Response(
            content=json.dumps({"error": "File must be specified"}),
            status_code=404,
            media_type="application/json",
        )
    try:
        artifact = await create_artifact(agent_id, file, relative_path)
        return Response(
            content=artifact.json(),
            status_code=200,
            media_type="application/json",
        )
    except Exception:
        LOG.exception(f"Error whilst trying to upload artifact: {agent_id}")
        return Response(
            content=json.dumps({"error": "Internal server error"}),
            status_code=500,
            media_type="application/json",
        )


@afaas_artifact_router.post(
    "/agent/{agent_id}/artifacts/{artifact_id}",
    tags=["artifacts"],
    response_model=str,
)
@artifact_router.get(
    "/agent/tasks/{agent_id}/artifacts/{artifact_id}",
    tags=["artifacts"],
    response_model=str,
)
async def download_agent_artifact(
    request: Request,
    agent_id: str,
    artifact_id: str,
    agent: PlannerAgent = Depends(get_agent),
) -> FileResponse:
    """
    Downloads an artifact associated with a specific task.

    Args:
        request (Request): FastAPI request object.
        agent_id (str): The ID of the task.
        artifact_id (str): The ID of the artifact.

    Returns:
        FileResponse: The downloaded artifact file.

    Example:
        Request:
            GET /agent/tasks/50da533e-3904-4401-8a07-c49adf88b5eb/artifacts/artifact1_id

        Response:
            <file_content_of_artifact>
    """
    request["agent"]
    try:
        return await get_artifact(agent_id, artifact_id)
    except NotFoundError:
        LOG.exception(f"Error whilst trying to download artifact: {agent_id}")
        return Response(
            content=json.dumps(
                {
                    "error": f"Artifact not found - agent_id: {agent_id}, artifact_id: {artifact_id}"
                }
            ),
            status_code=404,
            media_type="application/json",
        )
    except Exception:
        LOG.exception(f"Error whilst trying to download artifact: {agent_id}")
        return Response(
            content=json.dumps(
                {
                    "error": f"Internal server error - agent_id: {agent_id}, artifact_id: {artifact_id}"
                }
            ),
            status_code=500,
            media_type="application/json",
        )


async def list_artifacts(
    agent, agent_id: str, page: int = 1, pageSize: int = 10
) -> AgentArtifactsListResponse:
    """
    List the artifacts that the task has created.
    """
    try:
        artifacts, pagination = await agent.db.list_artifacts(agent_id, page, pageSize)
        return AgentArtifactsListResponse(artifacts=artifacts, pagination=pagination)

    except Exception:
        raise


async def create_artifact(
    agent, agent_id: str, file: UploadFile, relative_path: str
) -> Artifact:
    """
    Create an artifact for the task.
    """
    data = None
    file_name = file.filename or str(uuid4())
    try:
        data = b""
        while contents := file.file.read(1024 * 1024):
            data += contents
        # Check if relative path ends with filename
        if relative_path.endswith(file_name):
            file_path = relative_path
        else:
            file_path = os.path.join(relative_path, file_name)

        agent.workspace.write(agent_id, file_path, data)

        artifact = await agent.db.create_artifact(
            agent_id=agent_id,
            file_name=file_name,
            relative_path=relative_path,
            agent_created=False,
        )
    except Exception:
        raise
    return artifact


async def get_artifact(agent, agent_id: str, artifact_id: str) -> Artifact:
    """
    Get an artifact by ID.
    """
    try:
        artifact = await agent.db.get_artifact(artifact_id)
        if artifact.file_name not in artifact.relative_path:
            file_path = os.path.join(artifact.relative_path, artifact.file_name)
        else:
            file_path = artifact.relative_path
        retrieved_artifact = agent.workspace.read(agent_id=agent_id, path=file_path)
    except NotFoundError:
        raise
    except FileNotFoundError:
        raise
    except Exception:
        raise

    return StreamingResponse(
        BytesIO(retrieved_artifact),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={artifact.file_name}"},
    )
