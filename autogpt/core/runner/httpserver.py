import asyncio.queues
import uuid
from collections import defaultdict

from fastapi import APIRouter, FastAPI, Request

from autogpt.core.messaging.simple import Message, Role, SimpleMessageBroker
from autogpt.core.runner.agent import agent_context
from autogpt.core.runner.factory import agent_factory_context
from autogpt.core.runner.schema import AgentConfiguration, InteractRequestBody

router = APIRouter()

# Set up MessageBroker shim as a thin intermediary between the HTTP server and the agent.
# This is a critical abstraction to enable robust inter-agent communication later and to
# enable long-running agents. For now, it's just an indirection layer to mock something like
# kafka, redis, or rabbitmq, e.g.


class ApplicationQueue:
    queue = defaultdict(asyncio.queues.Queue)

    class Config:
        arbitrary_types_allowed = True

    async def get(self, channel_name: str):
        """Gets a message from the channel."""
        return await self.queue[channel_name].get()

    async def put(self, channel_name: str, message) -> None:
        await self.queue[channel_name].put(message)


APPLICATION_MESSAGE_QUEUE = ApplicationQueue()


class MessageFilters:
    @staticmethod
    def is_user_message(message: Message):
        return message.sender.role == Role.USER

    @staticmethod
    def is_agent_message(message: Message):
        return message.sender.role == Role.AGENT

    @staticmethod
    def is_agent_factory_message(message: Message):
        return message.sender.role == Role.AGENT_FACTORY

    @staticmethod
    def is_server_message(message: Message):
        return MessageFilters.is_agent_message(
            message
        ) | MessageFilters.is_agent_factory_message(message)

    @staticmethod
    def is_parse_goals_message(message: Message):
        metadata = message.additional_metadata
        return (
            MessageFilters.is_user_message(message) & metadata["instruction"]
            == "parse_goals"
        )

    @staticmethod
    def is_user_bootstrap_message(message: Message):
        metadata = message.additional_metadata
        return (
            MessageFilters.is_user_message(message) & metadata["instruction"]
            == "bootstrap_agent"
        )

    @staticmethod
    def is_user_launch_message(message: Message):
        metadata = message.additional_metadata
        return (
            MessageFilters.is_user_message(message) & metadata["instruction"]
            == "launch_agent"
        )


def get_message_broker() -> SimpleMessageBroker:
    message_channel_name = "autogpt"
    message_broker = SimpleMessageBroker()
    message_broker.create_message_channel(message_channel_name)

    message_broker.register_listener(
        message_channel="autogpt",
        listener=lambda message: APPLICATION_MESSAGE_QUEUE.put("autogpt", message),
        message_filter=MessageFilters.is_server_message,
    )

    message_broker.register_listener(
        message_channel="autogpt",
        listener=agent_factory_context.parse_goals,
        message_filter=MessageFilters.is_parse_goals_message,
    )

    message_broker.register_listener(
        message_channel="autogpt",
        listener=agent_factory_context.bootstrap_agent,
        message_filter=MessageFilters.is_user_bootstrap_message,
    )

    message_broker.register_listener(
        message_channel="autogpt",
        listener=agent_context.launch_agent,
        message_filter=MessageFilters.is_user_launch_message,
    )

    return message_broker


MESSAGE_BROKER = get_message_broker()
APPLICATION_UUID = uuid.uuid4()
APPLICATION_EMITTER = MESSAGE_BROKER.get_emitter(
    channel_name="autogpt",
    sender_uuid=APPLICATION_UUID,
    sender_name="autogpt-server",
    sender_role=Role.APPLICATION_SERVER,
)
APPLICATION_USERS = {}


def get_user_id(request: Request) -> uuid.UUID:
    # TODO: use a real user id
    hostname = request.client.host
    port = request.client.port
    user = f"{hostname}:{port}"
    if user not in APPLICATION_USERS:
        APPLICATION_USERS[user] = uuid.uuid4()
    return APPLICATION_USERS[user]


@router.post("/agents")
async def create_agent(request: Request, agent_configuration: AgentConfiguration):
    """Create a new agent."""
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
# NOTE:
# - start with `uvicorn autogpt.core.runner.server:app --reload --port=8080`
# - see auto-generated API docs: http://localhost:8080/docs
