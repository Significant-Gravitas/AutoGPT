from collections import defaultdict


from autogpt.core.runner.agent import agent_context
from autogpt.core.runner.factory import agent_factory_context

##################################################
# Hacking stuff together for an in-process model #
##################################################


class FakeApplicationServer:
    """The interface to the 'application server' process.

    This could be a restful API or something.

    """

    message_queue = defaultdict(list)

    def __init__(self):
        self._message_broker = self._get_message_broker()

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


#application_server = FakeApplicationServer()


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

