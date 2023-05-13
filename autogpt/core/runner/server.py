import uuid
from collections import defaultdict
from fastapi import APIRouter, FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional, Any, Dict

from autogpt.core.agent.base import Agent
from autogpt.core.messaging.simple import Message, Role, SimpleMessageBroker
from autogpt.core.runner.factory import bootstrap_agent

##################################################
# Hacking stuff together for an in-process model #
##################################################


class MessageFilters:
    @staticmethod
    def is_user_message(message: Message):
        metadata = message.metadata
        return metadata.sender.role == Role.USER

    @staticmethod
    def is_agent_message(message: Message):
        metadata = message.metadata
        return metadata.sender.role == Role.AGENT

    @staticmethod
    def is_agent_factory_message(message: Message):
        metadata = message.metadata
        return metadata.sender.role == Role.AGENT_FACTORY

    @staticmethod
    def is_server_message(message: Message):
        return MessageFilters.is_agent_message(
            message
        ) | MessageFilters.is_agent_factory_message(message)

    @staticmethod
    def is_user_bootstrap_message(message: Message):
        metadata = message.metadata
        return (
            MessageFilters.is_user_message(message)
            & metadata.additional_metadata["instruction"]
            == "bootstrap_agent"
        )

    @staticmethod
    def is_user_launch_message(message: Message):
        metadata = message.metadata
        return (
            MessageFilters.is_user_message(message)
            & metadata.additional_metadata["instruction"]
            == "launch_agent"
        )


class FakeApplicationServer:
    """The interface to the 'application server' process.

    This could be a restful API or something.

    """

    message_queue = defaultdict(list)

    def __init__(self):
        self._message_broker = self._get_message_broker()

        self._user_emitter = self._message_broker.get_emitter(
            channel_name="autogpt",
            sender_name="autogpt-user",
            sender_role=Role.USER,
        )

    def _get_message_broker(self) -> SimpleMessageBroker:
        message_channel_name = "autogpt"
        message_broker = SimpleMessageBroker()
        message_broker.create_message_channel(message_channel_name)

        message_broker.register_listener(
            message_channel="autogpt",
            listener=self._add_to_queue,
            message_filter=MessageFilters.is_server_message,
        )

        message_broker.register_listener(
            message_channel="autogpt",
            listener=bootstrap_agent,
            message_filter=MessageFilters.is_user_bootstrap_message,
        )

        message_broker.register_listener(
            message_channel="autogpt",
            listener=self.launch_agent,
            message_filter=MessageFilters.is_user_launch_message,
        )

        return message_broker

    def _add_to_queue(self, message: Message):
        self.message_queue[message.metadata.sender.name].append(message)

    def _send_message(
        self,
        request,
        extra_content: dict = None,
        extra_metadata: dict = None,
    ):
        content = {**request.json["content"], **extra_content}
        metadata = {**request.json["metadata"], **extra_metadata}

        success = self._user_emitter.send_message(content, **metadata)
        response = object()
        if success:
            response.status_code = 200
        else:
            response.status_code = 500
        return response

    def list_agents(self, request):
        """List all agents."""
        pass

    def boostrap_new_agent(self, request):
        """Bootstrap a new agent."""
        response = self._send_message(
            request,
            extra_content={"message_broker": self._message_broker},
            extra_metadata={"instruction": "bootstrap_agent"},
        )
        # Collate all responses from the agent factory since we're in-process.
        agent_factory_responses = self.message_queue["autogpt-agent-factory"]
        self.message_queue["autogpt-agent-factory"] = []
        response.json = agent_factory_responses
        return response

    def launch_agent(self, request):
        """Launch an agent."""
        return self._send_message(request)

    def give_agent_feedback(self, request):
        """Give feedback to an agent."""
        response = self._send_message(request)
        response.json = {
            "content": self.message_queue["autogpt-agent"].pop(),
        }

    # def get_agent_plan(self, request):
    #     """Get the plan for an agent."""
    #     # TODO: need a clever hack here to get the agent plan since we'd have natural
    #     #  asynchrony here with a webserver.
    #     pass


application_server = FakeApplicationServer()


def _get_workspace_path_from_agent_name(agent_name: str) -> str:
    # FIXME: Very much a stand-in for later logic. This could be a whole agent registry
    #  system and probably lives on the client side instead of here
    return f"~/autogpt_workspace/{agent_name}"


def launch_agent(message: Message):
    message_content = message.content
    message_broker = message_content["message_broker"]
    agent_name = message_content["agent_name"]
    workspace_path = _get_workspace_path_from_agent_name(agent_name)

    agent = Agent.from_workspace(workspace_path, message_broker)
    agent.run()


###############
# HTTP SERVER #
###############

router = APIRouter()


class CreateAgentRequestBody(BaseModel):
    ai_name: str
    ai_role: str
    ai_goals: List[str]
    # could add more config as needed


class CreateAgentResponseBody(BaseModel):
    agent_id: str


@router.post("/agents")
async def create_agent(request: Request, body: CreateAgentRequestBody):
    """Create a new agent."""

    # validate headers. This is where you would do auth.
    # currently checks for an api key (as an example)
    api_key = request.headers.get("openai_api_key")
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="missing openai_api_key header key",
        )

    # this is where you would do something with the request body
    # ...

    # initialize the agent

    agent_id = uuid.uuid4().hex

    return {"agent_id": agent_id}


class InteractRequestBody(BaseModel):
    user_input: Optional[str] = None


class InteractResponseBody(BaseModel):
    thoughts: Dict[str, str]  # TBD
    messages: List[str]  # for example


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
                "args": {
                    "arg_1": "value_1",
                    "arg_2": "value_2"
                }
            }
        },
        "messages": ["message1", agent_id]
    }


app = FastAPI()
app.include_router(router, prefix="/api/v1")
# NOTE:
# - start with `uvicorn autogpt.core.runner.server:app --reload --port=8080`
# - see auto-generated API docs: http://localhost:8080/docs

