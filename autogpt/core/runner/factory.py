import logging

from autogpt.core.agent.factory import SimpleAgentFactory
from autogpt.core.messaging.simple import Message, Role, SimpleMessageBroker


def configure_agent_factory_logging(
    agent_factory_logger: logging.Logger,
):
    agent_factory_logger.setLevel(logging.DEBUG)


def get_agent_factory() -> SimpleAgentFactory:
    # Configure logging before we do anything else.
    # Factory logs need a place to live.
    agent_factory_logger = logging.getLogger("autogpt_agent_factory")
    configure_agent_factory_logging(
        agent_factory_logger,
    )
    return SimpleAgentFactory(agent_factory_logger)


class AgentFactoryMessageEmitter:
    def __init__(self, message_broker):
        self._message_broker = message_broker
        self._emitter = message_broker.get_emitter(
            # can get from user config
            channel_name="autogpt",
            sender_name="autogpt-agent-factory",
            sender_role=Role.AGENT_FACTORY,
        )

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


async def bootstrap_agent(
    message: Message,
) -> None:
    """Provision a new agent by getting an objective from the user and setting up agent resources."""
    # TODO: this could be an already running process we communicate with via the
    # message broker.  For now, we'll just do it in-process.
    agent_factory = get_agent_factory()

    message_content: dict = message.content
    message_broker: SimpleMessageBroker = message_content["message_broker"]
    user_configuration: dict = message_content["user_configuration"]
    user_objective: str = message_content["user_objective"]
    agent_factory_emitter = AgentFactoryMessageEmitter(message_broker)

    await agent_factory_emitter.send_confirmation_message()
    # Either need to do validation as we're building the configuration, or shortly
    # after. Probably should have systems do their own validation.
    configuration, configuration_errors = agent_factory.compile_configuration(
        user_configuration,
    )
    if configuration_errors:
        await agent_factory_emitter.send_configuration_error_message()
        return
    await agent_factory_emitter.send_configuration_success_message()

    objective_prompt = agent_factory.construct_objective_prompt_from_user_input(
        user_objective,
        configuration,
    )
    await agent_factory_emitter.send_user_objective_message(objective_prompt)

    model_response = await agent_factory.determine_agent_objective(
        objective_prompt,
        configuration,
    )
    await agent_factory_emitter.send_agent_objective_message(model_response.content)
    # Set the agents goals
    configuration.planner.update(model_response.content)

    # TODO: Provision memory backend. Waiting on interface to stabilize

    await message_broker.send_message(
        "agent_setup_complete",
        {"message": "Agent setup complete."},
    )
