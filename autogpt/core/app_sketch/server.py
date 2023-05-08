import logging

from autogpt.core.agent.base import Agent, AgentFactory
from autogpt.core.messaging.base import (
    Role,
    Message,
)


def configure_agent_factory_logging(
    agent_factory_logger: logging.Logger,
):
    agent_factory_logger.setLevel(logging.DEBUG)


def get_agent_factory():
    # Configure logging before we do anything else.
    # Factory logs need a place to live.
    agent_factory_logger = logging.getLogger("autogpt_agent_factory")
    configure_agent_factory_logging(
        agent_factory_logger,
    )
    return AgentFactory(agent_factory_logger)


def bootstrap_agent(
    message: Message,
):
    """Provision a new agent by getting an objective from the user and setting up agent resources."""
    # TODO: this could be an already running process we communicate with via the
    # message broker.  For now, we'll just do it in-process.
    agent_factory = get_agent_factory()

    message_content = message.content
    message_broker = message_content["message_broker"]
    user_configuration = message_content["user_configuration"]
    user_objective = message_content["user_objective"]

    agent_factory_emitter = message_broker.get_emitter(
        # can get from user config
        channel_name='autogpt',
        sender_name='autogpt-agent-factory',
        sender_role=Role.AGENT_FACTORY,
    )
    agent_factory_emitter.send_message(
        content={"message": "Startup request received, Setting up agent..."},
        message_type="log",
    )

    # Either need to do validation as we're building the configuration, or shortly
    # after.
    configuration, configuration_errors = agent_factory.compile_configuration(
        user_configuration,
    )
    if configuration_errors:
        agent_factory_emitter.send_message(
            content={"message": "Configuration errors encountered, aborting agent setup."},
            message_type="error",
        )
        return

    agent_factory_emitter.send_message(
        content={"message": "Agent configuration compiled. Constructing initial agent plan from user objective."},
        message_type="log",
    )

    agent_planner = agent_factory.get_system_class("planner", configuration)
    # TODO: is this a class method?  Or do we have the planner be partially initialized
    #  without access to any resources since this precedes Agent creation?
    objective_prompt = agent_planner.construct_objective_prompt_from_user_input(
        user_objective,
    )
    agent_factory_emitter.send_message(
        content={"message": "Translated user input into objective prompt.",
                 "objective_prompt": objective_prompt},
        message_type="log",
    )

    language_model = agent_factory.get_system_class("language_model", configuration)
    # TODO: is this a class method?  Or do we have the language model be
    #  partially initialized without access to any resources since this precedes
    #  Agent creation?
    model_response = language_model.construct_objective_from_prompt(objective_prompt)
    # This should be parsed into a standard format already
    agent_objective = model_response.content

    agent_factory_emitter.send_message(
        content={"message": "Translated objective prompt into objective.",
                 "objective": agent_objective},
        message_type="log",
    )

    budget_manager = agent_factory.get_system_class("budget_manager", configuration)
    # TODO: is this a class method? Or do we have the budget manager be partially
    #  initialized without access to any resources since this precedes Agent creation?

    # ...Update API budget, etc. ...

    # TODO: Set up workspace
    # TODO: Provision memory backend

    message_broker.send_message(
        "agent_setup_complete",
        {"message": "Agent setup complete."},
    )


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
