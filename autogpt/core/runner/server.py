from collections import defaultdict

from autogpt.core.messaging.simple import Message, Role, SimpleMessageBroker
from autogpt.core.runner.agent import agent_context
from autogpt.core.runner.factory import agent_factory_context

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
            listener=agent_factory_context.bootstrap_agent,
            message_filter=MessageFilters.is_user_bootstrap_message,
        )

        message_broker.register_listener(
            message_channel="autogpt",
            listener=agent_context.launch_agent,
            message_filter=MessageFilters.is_user_launch_message,
        )

        return message_broker

    async def _add_to_queue(self, message: Message):
        self.message_queue[message.metadata.sender.name].append(message)

    async def _send_message(
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

    async def list_agents(self, request):
        """List all agents."""
        pass

    async def boostrap_new_agent(self, request):
        """Bootstrap a new agent."""
        response = await self._send_message(
            request,
            extra_content={"message_broker": self._message_broker},
            extra_metadata={"instruction": "bootstrap_agent"},
        )
        # Collate all responses from the agent factory since we're in-process.
        agent_factory_responses = self.message_queue["autogpt-agent-factory"]
        self.message_queue["autogpt-agent-factory"] = []
        response.json = agent_factory_responses
        return response

    async def launch_agent(self, request):
        """Launch an agent."""
        return await self._send_message(request)

    async def give_agent_feedback(self, request):
        """Give feedback to an agent."""
        response = await self._send_message(request)
        response.json = {
            "content": self.message_queue["autogpt-agent"].pop(),
        }

    # async def get_agent_plan(self, request):
    #     """Get the plan for an agent."""
    #     # TODO: need a clever hack here to get the agent plan since we'd have natural
    #     #  asynchrony here with a webserver.
    #     pass


application_server = FakeApplicationServer()
