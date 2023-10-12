
import json
from typing import Optional

from fastapi import APIRouter, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse

from app.sdk.errors import *
from app.sdk.forge_log import ForgeLogger
from app.sdk.schema import *

agent_router = APIRouter()

LOG = ForgeLogger(__name__)

"""
/agents (GET): Returns a list of all agents.
AP alias /agent/tasks (GET): Returns a list of all agents.
    => Need Agent (Parent) to define this properties : agent_id, client_facing, status

    
/agent (POST): Create a new agent.
AP alias /agent/tasks (POST): Create a new agent.
    => TODO : Save an Agent (or PlannerAgent) in the workspace (can't find the method still reading the code)

    
/agent/{agent_id} (GET): Get an agent from its ID & return an agent.
AP alias /agent/tasks/{agent_id} (GET): Get an agent from its ID & return an agent.


/agent/{agent_id}/start (POST): Send a message to an agent.
/agent/{agent_id}/message (POST): Sends a message to the agent.
/agent/{agent_id}/messagehistory (GET): Get message history for an agent.
    => need PlannerAgent to provide a method 
/agent/{agent_id}/lastmessage (GET): Get the last message for an agent.
    => need PlannerAgent to provide a method 
"""
import uuid
from pathlib import Path
import yaml
from fastapi import APIRouter, FastAPI, Request


from autogpts.autogpt.autogpt.core.runner.cli_web_app.server.schema import PlannerAgentMessageRequestBody
from autogpts.autogpt.autogpt.core.runner.client_lib.workspacebuilder import (
    workspace_loader,
    get_settings_from_file,
    get_logger_and_workspace,
)
from autogpts.autogpt.autogpt.core.agents import PlannerAgent
from autogpts.autogpt.autogpt.core.runner.client_lib.parser import (
    parse_agent_name_and_goals,
    parse_ability_result,
    parse_agent_plan,
    parse_next_ability,
)


router = APIRouter()
user_id = uuid.UUID("a1621e69-970a-4340-86e7-778d82e2137b")



@agent_router.get("/agents", tags=["agent"], response_model=TaskListResponse)
@agent_router.get("/agent/tasks", tags=["agent"], response_model=TaskListResponse)
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

    from autogpts.autogpt.autogpt.core.agents import PlannerAgent
    LOG.info("Getting agent settings")

    # Step 1. Collate the user's settings with the default system settings.
    agent_settings: PlannerAgent.SystemSettings = PlannerAgent.SystemSettings()


    #
    # We support multiple users however since there is no UI to enforce that we will be using a user with ID : a1621e69-970a-4340-86e7-778d82e2137b
    #
    agent_settings.user_id = user_id

    # NOTE : Real world scenario, this user_id will be passed as an argument
    agent_dict_list = PlannerAgent.get_agentsetting_list_from_memory(
        user_id=user_id, logger=LOG
    )

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

@agent_router.post("/agent")
@agent_router.post("/agent/tasks")
async def create_agent(request: Request):
    """
    Create a new agent.
    Minimal requirements : None
    """
    agent_id = uuid.uuid4().hex

    # TODO : Save the Agent, idealy managed in Agent() not Simple Agent.
    return {"agent_id": agent_id}


@agent_router.get("/agent/{agent_id}")
@agent_router.get("/agent/tasks/{agent_id}")
async def get_agent_by_id(request: Request, agent_id: str):
    """
    Get an agent from it's ID & return an agent
    """
    settings = get_settings_from_file()
    LOG, agent_workspace = get_logger_and_workspace(settings)

    # workspace =  workspace_loader(settings, LOG, agent_workspace)
    agent = {}
    if agent_workspace:
        agent = PlannerAgent.from_workspace(
            agent_workspace,
            LOG,
        )
        agent = agent.__dict__
        # TODO place holder for elements
        # TODO filter on client_facing
        # TODO filter on inactive
        agent["agent_id"] = uuid.uuid4().hex
        agent["client_facing"] = True
        agent["status"] = 0

    return {"agent": agent}


@agent_router.post("/agent/{agent_id}/start")
async def start_simple_agent_main_loop(request: Request, agent_id: str):
    """
    Senf a message to an agent
    """

    # Get the logger and the workspace movec to a function
    # Because almost every API end-point will excecute this piece of code
    user_configuration = get_settings_from_file()
    LOG, agent_workspace = get_logger_and_workspace(user_configuration)

    # Get the logger and the workspace moved to a function
    # Because API end-point will treat this as an error & break
    if not agent_workspace:
        raise BaseException

    # launch agent interaction loop
    agent = PlannerAgent.from_workspace(
        agent_workspace,
        LOG,
    )

    plan = await agent.build_initial_plan()

    current_task, next_ability = await agent.determine_next_ability(plan)

    return {
        "plan": plan.__dict__,
        "current_task": current_task.__dict__,
        "next_ability": next_ability.__dict__,
    }


@agent_router.post("/agent/{agent_id}/message")
async def message_simple_agent(
    request: Request, agent_id: str, body: PlannerAgentMessageRequestBody
):
    """
    Description: Sends a message to the agent.
    Arg :
    - Required body.message  (str): The message sent from the client
    - Optional body.start (bool): Start a new loop
    Comment: Only works if the agent is inactive. Works for both client-facing = true and false, enabling sub-agents.
    """
    user_configuration = get_settings_from_file()
    # Get the logger and the workspace movec to a function
    # Because almost every API end-point will excecute this piece of code
    LOG, agent_workspace = get_logger_and_workspace(user_configuration)

    # Get the logger and the workspace moved to a function
    # Because API end-point will treat this as an error & break
    if not agent_workspace:
        print(f"request.body: {request.body}")
        print(f"agent_id: {agent_id}")
        print(f"body: {body}")
        raise BaseException

    # launch agent interaction loop
    agent = PlannerAgent.from_workspace(
        agent_workspace,
        LOG,
    )

    plan = await agent.build_initial_plan()

    current_task, next_ability = await agent.determine_next_ability(plan)

    result = {}
    ability_result = await agent.execute_next_ability(body.message)
    result["ability_result"] = ability_result.__dict__

    if body.start == True:
        (
            result["current_task"],
            result["next_ability"],
        ) = await agent.determine_next_ability(plan)

    return result


@agent_router.get("/agent/{agent_id}/messagehistory")
def get_message_history(request: Request, agent_id: str):
    # TODO : Define structure of the list
    return {"messages": ["message 1", "message 2", "message 3", "message 4"]}


@agent_router.get("/agent/{agent_id}/lastmessage")
def get_last_message(request: Request, agent_id: str):
    return {"message": "my last message"}



@agent_router.post("/agent/tasks", tags=["agent"], response_model=Task)
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




@agent_router.get("/agent/tasks/{task_id}", tags=["agent"], response_model=Task)
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


@agent_router.get(
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


@agent_router.post("/agent/tasks/{task_id}/steps", tags=["agent"], response_model=Step)
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


@agent_router.get(
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
