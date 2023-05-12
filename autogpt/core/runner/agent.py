from pathlib import Path

from autogpt.core.agent.simple import SimpleAgent
from autogpt.core.messaging.simple import Message, Role, SimpleMessageBroker


class AgentContext:
    def __init__(self):
        self._agent = None
        self._message_broker = None
        self._emitter = None

    async def launch_agent(self, message: Message):
        message_content = message.content
        self._message_broker: SimpleMessageBroker = message_content["message_broker"]
        agent_name = message_content["agent_name"]
        self._agent_emitter = self._message_broker.get_emitter(
            channel_name="autogpt",
            sender_name=agent_name,
            sender_role=Role.AGENT,
        )

        workspace_path = self._get_workspace_path_from_agent_name(agent_name)

        self._agent = SimpleAgent.from_workspace(workspace_path)
        await self._send_agent_launched_message()

    @staticmethod
    def _get_workspace_path_from_agent_name(agent_name: str) -> Path:
        # FIXME: Very much a stand-in for later logic. This could be a whole agent registry
        #  system and probably lives on the client side instead of here
        return Path.home() / "auto-gpt" / agent_name

    async def _send_agent_launched_message(self):
        await self._emitter.send_message(
            content={"message": "Agent launched and awaiting instructions..."},
            message_type="log",
        )


agent_context = AgentContext()
