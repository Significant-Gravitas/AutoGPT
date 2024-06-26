"""
Routes for the Agent Service.

This module defines the API routes for the Agent service.

Developers and contributors should be especially careful when making modifications
to these routes to ensure consistency and correctness in the system's behavior.
"""
import logging
from typing import TYPE_CHECKING, Optional

from fastapi import APIRouter, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import StreamingResponse

from .models import (
    Artifact,
    Step,
    StepRequestBody,
    Task,
    TaskArtifactsListResponse,
    TaskListResponse,
    TaskRequestBody,
    TaskStepsListResponse,
)

if TYPE_CHECKING:
    from .agent import ProtocolAgent

base_router = APIRouter()
logger = logging.getLogger(__name__)


@base_router.get("/", tags=["root"])
async def root():
    """
    Root endpoint that returns a welcome message.
    """
    return Response(content="Welcome to the AutoGPT Forge")


@base_router.get("/heartbeat", tags=["server"])
async def check_server_status():
    """
    Check if the server is running.
    """
    return Response(content="Server is running.", status_code=200)


@base_router.post("/agent/tasks", tags=["agent"], response_model=Task)
async def create_agent_task(request: Request, task_request: TaskRequestBody) -> Task:
    """
    Creates a new task using the provided TaskRequestBody and returns a Task.

    Args:
        request (Request): FastAPI request object.
        task (TaskRequestBody): The task request containing input data.

    Returns:
        Task: A new task with task_id, input, and additional_input set.

    Example:
        Request (TaskRequestBody defined in schema.py):
            {
                "input": "Write the words you receive to the file 'output.txt'.",
                "additional_input": "python/code"
            }

        Response (Task defined in schema.py):
            {
                "task_id": "50da533e-3904-4401-8a07-c49adf88b5eb",
                "input": "Write the word 'Washington' to a .txt file",
                "additional_input": "python/code",
                "artifacts": [],
            }
    """
    agent: "ProtocolAgent" = request["agent"]

    try:
        task = await agent.create_task(task_request)
        return task
    except Exception:
        logger.exception(f"Error whilst trying to create a task: {task_request}")
        raise


@base_router.get("/agent/tasks", tags=["agent"], response_model=TaskListResponse)
async def list_agent_tasks(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1),
) -> TaskListResponse:
    """
    Retrieves a paginated list of all tasks.

    Args:
        request (Request): FastAPI request object.
        page (int, optional): Page number for pagination. Default: 1
        page_size (int, optional): Number of tasks per page for pagination. Default: 10

    Returns:
        TaskListResponse: A list of tasks, and pagination details.

    Example:
        Request:
            GET /agent/tasks?page=1&pageSize=10

        Response (TaskListResponse defined in schema.py):
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
    agent: "ProtocolAgent" = request["agent"]
    try:
        tasks = await agent.list_tasks(page, page_size)
        return tasks
    except Exception:
        logger.exception("Error whilst trying to list tasks")
        raise


@base_router.get("/agent/tasks/{task_id}", tags=["agent"], response_model=Task)
async def get_agent_task(request: Request, task_id: str) -> Task:
    """
    Gets the details of a task by ID.

    Args:
        request (Request): FastAPI request object.
        task_id (str): The ID of the task.

    Returns:
        Task: The task with the given ID.

    Example:
        Request:
            GET /agent/tasks/50da533e-3904-4401-8a07-c49adf88b5eb

        Response (Task defined in schema.py):
            {
                "input": "Write the word 'Washington' to a .txt file",
                "additional_input": null,
                "task_id": "50da533e-3904-4401-8a07-c49adf88b5eb",
                "artifacts": [
                    {
                        "artifact_id": "7a49f31c-f9c6-4346-a22c-e32bc5af4d8e",
                        "file_name": "output.txt",
                        "agent_created": true,
                        "relative_path": "file://50da533e-3904-4401-8a07-c49adf88b5eb/output.txt"
                    }
                ],
                "steps": [
                    {
                        "task_id": "50da533e-3904-4401-8a07-c49adf88b5eb",
                        "step_id": "6bb1801a-fd80-45e8-899a-4dd723cc602e",
                        "input": "Write the word 'Washington' to a .txt file",
                        "additional_input": "challenge:write_to_file",
                        "name": "Write to file",
                        "status": "completed",
                        "output": "I am going to use the write_to_file command and write Washington to a file called output.txt <write_to_file('output.txt', 'Washington')>",
                        "additional_output": "Do you want me to continue?",
                        "artifacts": [
                            {
                                "artifact_id": "7a49f31c-f9c6-4346-a22c-e32bc5af4d8e",
                                "file_name": "output.txt",
                                "agent_created": true,
                                "relative_path": "file://50da533e-3904-4401-8a07-c49adf88b5eb/output.txt"
                            }
                        ],
                        "is_last": true
                    }
                ]
            }
    """  # noqa: E501
    agent: "ProtocolAgent" = request["agent"]
    try:
        task = await agent.get_task(task_id)
        return task
    except Exception:
        logger.exception(f"Error whilst trying to get task: {task_id}")
        raise


@base_router.get(
    "/agent/tasks/{task_id}/steps",
    tags=["agent"],
    response_model=TaskStepsListResponse,
)
async def list_agent_task_steps(
    request: Request,
    task_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, alias="pageSize"),
) -> TaskStepsListResponse:
    """
    Retrieves a paginated list of steps associated with a specific task.

    Args:
        request (Request): FastAPI request object.
        task_id (str): The ID of the task.
        page (int, optional): The page number for pagination. Defaults to 1.
        page_size (int, optional): Number of steps per page for pagination. Default: 10.

    Returns:
        TaskStepsListResponse: A list of steps, and pagination details.

    Example:
        Request:
            GET /agent/tasks/50da533e-3904-4401-8a07-c49adf88b5eb/steps?page=1&pageSize=10

        Response (TaskStepsListResponse defined in schema.py):
            {
                "items": [
                    {
                        "task_id": "50da533e-3904-4401-8a07-c49adf88b5eb",
                        "step_id": "step1_id",
                        ...
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
    """  # noqa: E501
    agent: "ProtocolAgent" = request["agent"]
    try:
        steps = await agent.list_steps(task_id, page, page_size)
        return steps
    except Exception:
        logger.exception("Error whilst trying to list steps")
        raise


@base_router.post("/agent/tasks/{task_id}/steps", tags=["agent"], response_model=Step)
async def execute_agent_task_step(
    request: Request, task_id: str, step_request: Optional[StepRequestBody] = None
) -> Step:
    """
    Executes the next step for a specified task based on the current task status and
    returns the executed step with additional feedback fields.

    This route is significant because this is where the agent actually performs work.
    The function handles executing the next step for a task based on its current state,
    and it requires careful implementation to ensure all scenarios (like the presence
    or absence of steps or a step marked as `last_step`) are handled correctly.

    Depending on the current state of the task, the following scenarios are possible:
    1. No steps exist for the task.
    2. There is at least one step already for the task, and the task does not have a
       completed step marked as `last_step`.
    3. There is a completed step marked as `last_step` already on the task.

    In each of these scenarios, a step object will be returned with two additional
    fields: `output` and `additional_output`.
    - `output`: Provides the primary response or feedback to the user.
    - `additional_output`: Supplementary information or data. Its specific content is
      not strictly defined and can vary based on the step or agent's implementation.

    Args:
        request (Request): FastAPI request object.
        task_id (str): The ID of the task.
        step (StepRequestBody): The details for executing the step.

    Returns:
        Step: Details of the executed step with additional feedback.

    Example:
        Request:
            POST /agent/tasks/50da533e-3904-4401-8a07-c49adf88b5eb/steps
            {
                "input": "Step input details...",
                ...
            }

        Response:
            {
                "task_id": "50da533e-3904-4401-8a07-c49adf88b5eb",
                "step_id": "step1_id",
                "output": "Primary feedback...",
                "additional_output": "Supplementary details...",
                ...
            }
    """
    agent: "ProtocolAgent" = request["agent"]
    try:
        # An empty step request represents a yes to continue command
        if not step_request:
            step_request = StepRequestBody(input="y")

        step = await agent.execute_step(task_id, step_request)
        return step
    except Exception:
        logger.exception(f"Error whilst trying to execute a task step: {task_id}")
        raise


@base_router.get(
    "/agent/tasks/{task_id}/steps/{step_id}", tags=["agent"], response_model=Step
)
async def get_agent_task_step(request: Request, task_id: str, step_id: str) -> Step:
    """
    Retrieves the details of a specific step for a given task.

    Args:
        request (Request): FastAPI request object.
        task_id (str): The ID of the task.
        step_id (str): The ID of the step.

    Returns:
        Step: Details of the specific step.

    Example:
        Request:
            GET /agent/tasks/50da533e-3904-4401-8a07-c49adf88b5eb/steps/step1_id

        Response:
            {
                "task_id": "50da533e-3904-4401-8a07-c49adf88b5eb",
                "step_id": "step1_id",
                ...
            }
    """
    agent: "ProtocolAgent" = request["agent"]
    try:
        step = await agent.get_step(task_id, step_id)
        return step
    except Exception:
        logger.exception(f"Error whilst trying to get step: {step_id}")
        raise


@base_router.get(
    "/agent/tasks/{task_id}/artifacts",
    tags=["agent"],
    response_model=TaskArtifactsListResponse,
)
async def list_agent_task_artifacts(
    request: Request,
    task_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, alias="pageSize"),
) -> TaskArtifactsListResponse:
    """
    Retrieves a paginated list of artifacts associated with a specific task.

    Args:
        request (Request): FastAPI request object.
        task_id (str): The ID of the task.
        page (int, optional): The page number for pagination. Defaults to 1.
        page_size (int, optional): Number of items per page for pagination. Default: 10.

    Returns:
        TaskArtifactsListResponse: A list of artifacts, and pagination details.

    Example:
        Request:
            GET /agent/tasks/50da533e-3904-4401-8a07-c49adf88b5eb/artifacts?page=1&pageSize=10

        Response (TaskArtifactsListResponse defined in schema.py):
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
    """  # noqa: E501
    agent: "ProtocolAgent" = request["agent"]
    try:
        artifacts = await agent.list_artifacts(task_id, page, page_size)
        return artifacts
    except Exception:
        logger.exception("Error whilst trying to list artifacts")
        raise


@base_router.post(
    "/agent/tasks/{task_id}/artifacts", tags=["agent"], response_model=Artifact
)
async def upload_agent_task_artifacts(
    request: Request, task_id: str, file: UploadFile, relative_path: str = ""
) -> Artifact:
    """
    This endpoint is used to upload an artifact (file) associated with a specific task.

    Args:
        request (Request): The FastAPI request object.
        task_id (str): The ID of the task for which the artifact is being uploaded.
        file (UploadFile): The file being uploaded as an artifact.
        relative_path (str): The relative path for the file. This is a query parameter.

    Returns:
        Artifact: Metadata object for the uploaded artifact, including its ID and path.

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
    """  # noqa: E501
    agent: "ProtocolAgent" = request["agent"]

    if file is None:
        raise HTTPException(status_code=400, detail="File must be specified")
    try:
        artifact = await agent.create_artifact(task_id, file, relative_path)
        return artifact
    except Exception:
        logger.exception(f"Error whilst trying to upload artifact: {task_id}")
        raise


@base_router.get(
    "/agent/tasks/{task_id}/artifacts/{artifact_id}",
    tags=["agent"],
    response_model=str,
)
async def download_agent_task_artifact(
    request: Request, task_id: str, artifact_id: str
) -> StreamingResponse:
    """
    Downloads an artifact associated with a specific task.

    Args:
        request (Request): FastAPI request object.
        task_id (str): The ID of the task.
        artifact_id (str): The ID of the artifact.

    Returns:
        FileResponse: The downloaded artifact file.

    Example:
        Request:
            GET /agent/tasks/50da533e-3904-4401-8a07-c49adf88b5eb/artifacts/artifact1_id

        Response:
            <file_content_of_artifact>
    """
    agent: "ProtocolAgent" = request["agent"]
    try:
        return await agent.get_artifact(task_id, artifact_id)
    except Exception:
        logger.exception(f"Error whilst trying to download artifact: {task_id}")
        raise
