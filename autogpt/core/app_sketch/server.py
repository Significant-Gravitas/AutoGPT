import logging

from autogpt.core.agent.base import AgentFactory
from autogpt.core.messaging.base import Message


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

    message_broker.send_message(
        "agent_setup",
        {"message": "Startup request received, Setting up agent..."},
    )

    configuration, configuration_errors = agent_factory.compile_configuration(
        user_configuration,
    )



    configuration_errors = agent_factory.validate_configuration()
    agent_configuration, configuration_errors = AgentFactory.build_agent_configuration(
        user_configuration,
    )
    if configuration_errors:
        message_broker.send_message(
            "agent_setup",
            {"message": "Configuration errors encountered, aborting agent setup.",
             "errors": configuration_errors},
        )
        return
    else:
        message_broker.send_message(
            "agent_setup",
            {"message": "Configuration validated, continuing agent setup."},
        )






    # Application can use its own custom concrete planner class here if it
    # wants to supercede the default method of constructing an objective prompt.
    objective_prompt = Planner.construct_objective_prompt_from_user_input \
        (user_objective)
    # Application can use its own custom language model class here if it wants.


def bootstrap_agent_v1(
        file_based_configuration: dict,  # Need to figure out what's in here
        command_line_arguments: dict,  # Need to figure out what's in here
        application_logger: Logger,
):
    """Provision a new agent by getting an objective from the user and setting up agent resources."""
    # TODO: Need to resolve the plugin conversation to scope out how configuration
    #   namespace gets constructed, and how default values get set. Presumably we
    #   can look in the file-based config and command line arguments to collate
    #   agent system classes (using core systems if none are specified), and then
    #   use the agent system classes to construct the configuration namespace.
    # Any non-trivial configuration validation will happen here.
    # Pass in the logger so we can log warnings and errors.
    agent_configuration = build_agent_configuration(
        file_based_configuration,
        command_line_arguments,
        application_logger,
    )

    # Find out the user's objective for the new agent.
    user_objective = input(...)
    # Application can use its own custom concrete planner class here if it
    # wants to supercede the default method of constructing an objective prompt.
    # TODO: is this a static method of the planner, or do we init a planner with
    #   only part of its arguments (ie just the configuration) since other things
    #   like the workspace don't exist yet?
    objective_prompt = Planner.construct_objective_prompt_from_user_input \
        (user_objective)
    # Application can use its own custom language model class here if it wants.



def launch_agent():
    pass