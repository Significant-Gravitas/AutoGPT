import uuid

from fastapi import APIRouter, FastAPI, Request

from autogpt.core.agent import AgentSettings
from autogpt.core.runner.cli_web_app.server.services.users import USER_SERVICE

router = APIRouter()


@router.post("/agents")
async def create_agent(request: Request, agent_settings: AgentSettings):
    """Create a new agent."""
    user_id = USER_SERVICE.get_user_id(request)

    return {"agent_id": uuid.uuid4().hex}

    user_id = get_user_id(request)
    user_emitter = MESSAGE_BROKER.get_emitter(
        channel_name="autogpt",
        sender_uuid=user_id,
        sender_name="autogpt-user",
        sender_role=Role.USER,
    )

    if agent_configuration.agent_goals.objective:
        await user_emitter.send_message(
            content=agent_configuration,
            message_broker=MESSAGE_BROKER,
            instruction="parse_goals",
        )
        agent_info_message = await APPLICATION_MESSAGE_QUEUE.get("autogpt")

        # 2. Send message with update agent info
        pass

    await USER_EMITTER.send_message(
        content=agent_configuration.dict(),
        additional_metadata={"instruction": "bootstrap_agent"},
    )

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
