import uuid

from fastapi import APIRouter, FastAPI, Request

from autogpt.core.runner.cli_web_app.server.schema import InteractRequestBody

router = APIRouter()


@router.post("/agents")
async def create_agent(request: Request):
    """Create a new agent."""
    agent_id = uuid.uuid4().hex
    return {"agent_id": agent_id}


@router.post("/agents/{agent_id}")
async def interact(request: Request, agent_id: str, body: InteractRequestBody):
    """Interact with an agent."""

    # check headers

    # check if agent_id exists

    # get agent object from somewhere, e.g. a database/disk/global dict

    # continue agent interaction with user input

    return {
        "thoughts": {
            "thoughts": {
                "text": "text",
                "reasoning": "reasoning",
                "plan": "plan",
                "criticism": "criticism",
                "speak": "speak",
            },
            "commands": {
                "name": "name",
                "args": {"arg_1": "value_1", "arg_2": "value_2"},
            },
        },
        "messages": ["message1", agent_id],
    }


app = FastAPI()
app.include_router(router, prefix="/api/v1")
