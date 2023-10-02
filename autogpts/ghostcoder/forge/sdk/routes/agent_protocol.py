"""
Routes for the Agent Service.

This module defines the API routes for the Agent service. While there are multiple endpoints provided by the service,
the ones that require special attention due to their complexity are:

1. `execute_agent_task_step`:
   This route is significant because this is where the agent actually performs the work. The function handles
   executing the next step for a task based on its current state, and it requires careful implementation to ensure
   all scenarios (like the presence or absence of steps or a step marked as `last_step`) are handled correctly.

2. `upload_agent_task_artifacts`:
   This route allows for the upload of artifacts, supporting various URI types (e.g., s3, gcs, ftp, http).
   The support for different URI types makes it a bit more complex, and it's important to ensure that all
   supported URI types are correctly managed. NOTE: The AutoGPT team will eventually handle the most common
   uri types for you.

3. `create_agent_task`:
   While this is a simpler route, it plays a crucial role in the workflow, as it's responsible for the creation
   of a new task.

Developers and contributors should be especially careful when making modifications to these routes to ensure
consistency and correctness in the system's behavior.
"""
import json
from typing import Optional

from fastapi import APIRouter, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse

from forge.sdk.errors import *
from forge.sdk.forge_log import ForgeLogger
from forge.sdk.schema import *

base_router = APIRouter()

LOG = ForgeLogger(__name__)


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
        task (TaskRequestBody): The task request containing input and additional input data.

    Returns:
        Task: A new task with task_id, input, additional_input, and empty lists for artifacts and steps.

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


@base_router.get("/agent/tasks", tags=["agent"], response_model=TaskListResponse)
async def list_agent_tasks(
    request: Request,
    page: Optional[int] = Query(1, ge=1),
    page_size: Optional[int] = Query(10, ge=1),
) -> TaskListResponse:
    """
    Retrieves a paginated list of all tasks.

    Args:
        request (Request): FastAPI request object.
        page (int, optional): The page number for pagination. Defaults to 1.
        page_size (int, optional): The number of tasks per page for pagination. Defaults to 10.

    Returns:
        TaskListResponse: A response object containing a list of tasks and pagination details.

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
    agent = request["agent"]
    try:
        tasks = await agent.list_tasks(page, page_size)
        return Response(
            content=tasks.json(),
            status_code=200,
            media_type="application/json",
        )
    except NotFoundError:
        LOG.exception("Error whilst trying to list tasks")
        return Response(
            content=json.dumps({"error": "Tasks not found"}),
            status_code=404,
            media_type="application/json",
        )
    except Exception:
        LOG.exception("Error whilst trying to list tasks")
        return Response(
            content=json.dumps({"error": "Internal server error"}),
            status_code=500,
            media_type="application/json",
        )


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
                        "uri": "file://50da533e-3904-4401-8a07-c49adf88b5eb/output.txt"
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
                                "uri": "file://50da533e-3904-4401-8a07-c49adf88b5eb/output.txt"
                            }
                        ],
                        "is_last": true
                    }
                ]
            }
    """
    agent = request["agent"]
    try:
        task = await agent.get_task(task_id)
        return Response(
            content=task.json(),
            status_code=200,
            media_type="application/json",
        )
    except NotFoundError:
        LOG.exception(f"Error whilst trying to get task: {task_id}")
        return Response(
            content=json.dumps({"error": "Task not found"}),
            status_code=404,
            media_type="application/json",
        )
    except Exception:
        LOG.exception(f"Error whilst trying to get task: {task_id}")
        return Response(
            content=json.dumps({"error": "Internal server error"}),
            status_code=500,
            media_type="application/json",
        )


@base_router.get(
    "/agent/tasks/{task_id}/steps", tags=["agent"], response_model=TaskStepsListResponse
)
async def list_agent_task_steps(
    request: Request,
    task_id: str,
    page: Optional[int] = Query(1, ge=1),
    page_size: Optional[int] = Query(10, ge=1, alias="pageSize"),
) -> TaskStepsListResponse:
    """
    Retrieves a paginated list of steps associated with a specific task.

    Args:
        request (Request): FastAPI request object.
        task_id (str): The ID of the task.
        page (int, optional): The page number for pagination. Defaults to 1.
        page_size (int, optional): The number of steps per page for pagination. Defaults to 10.

    Returns:
        TaskStepsListResponse: A response object containing a list of steps and pagination details.

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
    """
    agent = request["agent"]
    try:
        steps = await agent.list_steps(task_id, page, page_size)
        return Response(
            content=steps.json(),
            status_code=200,
            media_type="application/json",
        )
    except NotFoundError:
        LOG.exception("Error whilst trying to list steps")
        return Response(
            content=json.dumps({"error": "Steps not found"}),
            status_code=404,
            media_type="application/json",
        )
    except Exception:
        LOG.exception("Error whilst trying to list steps")
        return Response(
            content=json.dumps({"error": "Internal server error"}),
            status_code=500,
            media_type="application/json",
        )


@base_router.post("/agent/tasks/{task_id}/steps", tags=["agent"], response_model=Step)
async def execute_agent_task_step(
    request: Request, task_id: str, step: Optional[StepRequestBody] = None
) -> Step:
    """
    Executes the next step for a specified task based on the current task status and returns the
    executed step with additional feedback fields.

    Depending on the current state of the task, the following scenarios are supported:

    1. No steps exist for the task.
    2. There is at least one step already for the task, and the task does not have a completed step marked as `last_step`.
    3. There is a completed step marked as `last_step` already on the task.

    In each of these scenarios, a step object will be returned with two additional fields: `output` and `additional_output`.
    - `output`: Provides the primary response or feedback to the user.
    - `additional_output`: Supplementary information or data. Its specific content is not strictly defined and can vary based on the step or agent's implementation.

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
    agent = request["agent"]
    try:
        # An empty step request represents a yes to continue command
        if not step:
            step = StepRequestBody(input="y")
        step = await agent.execute_step(task_id, step)
        return Response(
            content=step.json(),
            status_code=200,
            media_type="application/json",
        )
    except NotFoundError:
        LOG.exception(f"Error whilst trying to execute a task step: {task_id}")
        return Response(
            content=json.dumps({"error": f"Task not found {task_id}"}),
            status_code=404,
            media_type="application/json",
        )
    except Exception as e:
        LOG.exception(f"Error whilst trying to execute a task step: {task_id}")
        return Response(
            content=json.dumps({"error": "Internal server error"}),
            status_code=500,
            media_type="application/json",
        )


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
    agent = request["agent"]
    try:
        step = await agent.get_step(task_id, step_id)
        return Response(content=step.json(), status_code=200)
    except NotFoundError:
        LOG.exception(f"Error whilst trying to get step: {step_id}")
        return Response(
            content=json.dumps({"error": "Step not found"}),
            status_code=404,
            media_type="application/json",
        )
    except Exception:
        LOG.exception(f"Error whilst trying to get step: {step_id}")
        return Response(
            content=json.dumps({"error": "Internal server error"}),
            status_code=500,
            media_type="application/json",
        )


@base_router.get(
    "/agent/tasks/{task_id}/artifacts",
    tags=["agent"],
    response_model=TaskArtifactsListResponse,
)
async def list_agent_task_artifacts(
    request: Request,
    task_id: str,
    page: Optional[int] = Query(1, ge=1),
    page_size: Optional[int] = Query(10, ge=1, alias="pageSize"),
) -> TaskArtifactsListResponse:
    """
    Retrieves a paginated list of artifacts associated with a specific task.

    Args:
        request (Request): FastAPI request object.
        task_id (str): The ID of the task.
        page (int, optional): The page number for pagination. Defaults to 1.
        page_size (int, optional): The number of items per page for pagination. Defaults to 10.

    Returns:
        TaskArtifactsListResponse: A response object containing a list of artifacts and pagination details.

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
    """
    agent = request["agent"]
    try:
        artifacts: TaskArtifactsListResponse = await agent.list_artifacts(
            task_id, page, page_size
        )
        return artifacts
    except NotFoundError:
        LOG.exception("Error whilst trying to list artifacts")
        return Response(
            content=json.dumps({"error": "Artifacts not found for task_id"}),
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


@base_router.post(
    "/agent/tasks/{task_id}/artifacts", tags=["agent"], response_model=Artifact
)
async def upload_agent_task_artifacts(
    request: Request, task_id: str, file: UploadFile, relative_path: Optional[str] = ""
) -> Artifact:
    """
    This endpoint is used to upload an artifact associated with a specific task. The artifact is provided as a file.

    Args:
        request (Request): The FastAPI request object.
        task_id (str): The unique identifier of the task for which the artifact is being uploaded.
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
    agent = request["agent"]

    if file is None:
        return Response(
            content=json.dumps({"error": "File must be specified"}),
            status_code=404,
            media_type="application/json",
        )
    try:
        artifact = await agent.create_artifact(task_id, file, relative_path)
        return Response(
            content=artifact.json(),
            status_code=200,
            media_type="application/json",
        )
    except Exception:
        LOG.exception(f"Error whilst trying to upload artifact: {task_id}")
        return Response(
            content=json.dumps({"error": "Internal server error"}),
            status_code=500,
            media_type="application/json",
        )


@base_router.get(
    "/agent/tasks/{task_id}/artifacts/{artifact_id}", tags=["agent"], response_model=str
)
async def download_agent_task_artifact(
    request: Request, task_id: str, artifact_id: str
) -> FileResponse:
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
    agent = request["agent"]
    try:
        return await agent.get_artifact(task_id, artifact_id)
    except NotFoundError:
        LOG.exception(f"Error whilst trying to download artifact: {task_id}")
        return Response(
            content=json.dumps(
                {
                    "error": f"Artifact not found - task_id: {task_id}, artifact_id: {artifact_id}"
                }
            ),
            status_code=404,
            media_type="application/json",
        )
    except Exception:
        LOG.exception(f"Error whilst trying to download artifact: {task_id}")
        return Response(
            content=json.dumps(
                {
                    "error": f"Internal server error - task_id: {task_id}, artifact_id: {artifact_id}"
                }
            ),
            status_code=500,
            media_type="application/json",
        )
