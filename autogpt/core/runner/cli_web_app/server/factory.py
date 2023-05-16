import logging
import uuid

from autogpt.core.agent.factory import SimpleAgentFactory
from autogpt.core.runner.app_lib.schema import AgentConfiguration, AgentInfo
from autogpt.core.runner.cli_web_app.messaging.simple import (
    Message,
    Role,
    SimpleMessageBroker,
)


class AgentFactoryContext:
    def __init__(self):
        self._uuid = uuid.uuid4()
        self._agent_factory = None
        self._message_broker = None
        self._emitter = None

    @staticmethod
    def configure_agent_factory_logging(agent_factory_logger: logging.Logger):
        agent_factory_logger.setLevel(logging.DEBUG)

    async def start_agent_factory(self, message: Message):
        agent_factory_logger = logging.getLogger("autogpt_agent_factory")
        self.configure_agent_factory_logging(agent_factory_logger)
        self._agent_factory = SimpleAgentFactory(agent_factory_logger)

        message_broker: SimpleMessageBroker = message.additional_metadata[
            "message_broker"
        ]
        self._emitter = message_broker.get_emitter(
            # can get from user config
            channel_name="autogpt",
            sender_uuid=self._uuid,
            sender_name="autogpt-agent-factory",
            sender_role=Role.AGENT_FACTORY,
        )

    async def parse_goals(self, message: Message):
        if self._agent_factory is None:
            await self.start_agent_factory(message)

        message_content: AgentConfiguration = message.content
        user_configuration: dict = message_content.user_configuration
        agent_objective: str = message_content.agent_goals.objective

        await self.send_confirmation_message()
        # Either need to do validation as we're building the configuration, or shortly
        # after. Probably should have systems do their own validation.
        configuration, configuration_errors = self._agent_factory.compile_configuration(
            user_configuration,
        )
        if configuration_errors:
            await self.send_configuration_error_message()
            return
        await self.send_configuration_success_message()

        objective_prompt = (
            self._agent_factory.construct_objective_prompt_from_user_input(
                agent_objective,
                configuration,
            )
        )
        await self.send_user_objective_message(objective_prompt)

        model_response = await self._agent_factory.determine_agent_objective(
            objective_prompt,
            configuration,
        )
        content = model_response.content
        await self.send_agent_objective_message(content)
        # Set the agents goals
        configuration.planner.update(content)

        agent_info = AgentInfo(
            objective=agent_objective,
            name=content["ai_name"],
            role=content["ai_role"],
            goals=content["ai_goals"],
        )
        self._emitter.send_message(
            content=agent_info,
            message_type="parse_goals_complete",
        )

    async def bootstrap_agent(self, message: Message):
        """Provision a new agent by getting an objective from the user and setting up agent resources."""
        if self._agent_factory is None:
            await self.start_agent_factory(message)

        message_content: dict = message.content

        user_configuration: dict = message_content["user_configuration"]
        user_objective: str = message_content["user_objective"]

        self._agent_factory.provision_new_agent(configuration)
        await self.send_agent_setup_complete_message()

    async def parse_user_objective_into_goals(self, user_objective):
        pass

    async def send_confirmation_message(self):
        await self._emitter.send_message(
            content={"message": "Startup request received, Setting up agent..."},
            message_type="log",
        )

    async def send_configuration_error_message(self):
        await self._emitter.send_message(
            content={
                "message": "Configuration errors encountered, aborting agent setup."
            },
            message_type="error",
        )

    async def send_configuration_success_message(self):
        await self._emitter.send_message(
            content={
                "message": (
                    "Agent configuration compiled. Constructing initial "
                    "agent plan from user objective."
                )
            },
            message_type="log",
        )

    async def send_user_objective_message(self, objective_prompt):
        await self._emitter.send_message(
            content={
                "message": "Translated user input into objective prompt.",
                "objective_prompt": objective_prompt,
            },
            message_type="log",
        )

    async def send_agent_objective_message(self, agent_objective):
        await self._emitter.send_message(
            content={
                "message": "Agent objective determined.",
                "agent_objective": agent_objective,
            },
            message_type="log",
        )

    async def send_agent_setup_complete_message(self):
        await self._emitter.send_message(
            content={
                "message": "Agent setup complete.",
            },
            message_type="log",
        )


agent_factory_context = AgentFactoryContext()
