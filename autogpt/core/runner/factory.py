import logging

from autogpt.core.agent.factory import AgentFactory
from autogpt.core.messaging.simple import Message, Role


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


async def bootstrap_agent(
    message: Message,
) -> None:
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
        channel_name="autogpt",
        sender_name="autogpt-agent-factory",
        sender_role=Role.AGENT_FACTORY,
    )
    await agent_factory_emitter.send_message(
        content={"message": "Startup request received, Setting up agent..."},
    )

    # Either need to do validation as we're building the configuration, or shortly
    # after.
    configuration, configuration_errors = agent_factory.compile_configuration(
        user_configuration,
    )
    if configuration_errors:
        await agent_factory_emitter.send_message(
            content={
                "message": "Configuration errors encountered, aborting agent setup."
            },
            message_type="error",
        )
        return

    await agent_factory_emitter.send_message(
        content={
            "message": "Agent configuration compiled. Constructing initial agent plan from user objective."
        },
        message_type="log",
    )

    agent_planner_class = agent_factory.get_system_class("planner", configuration)
    agent_planner = agent_planner_class(configuration)
    objective_prompt = agent_planner.construct_objective_prompt_from_user_input(
        user_objective,
    )
    await agent_factory_emitter.send_message(
        content={
            "message": "Translated user input into objective prompt.",
            "objective_prompt": objective_prompt,
        },
        message_type="log",
    )

    language_model_class = agent_factory.get_system_class(
        "language_model",
        configuration,
    )
    language_model = language_model_class(configuration)
    model_response = await language_model.construct_objective_from_prompt(
        objective_prompt
    )
    # This should be parsed into a standard format already
    agent_objective = model_response.content

    await agent_factory_emitter.send_message(
        content={
            "message": "Translated objective prompt into objective.",
            "objective": agent_objective,
        },
        message_type="log",
    )

    budget_manager_class = agent_factory.get_system_class(
        "budget_manager", configuration
    )
    budget_manager = budget_manager_class(configuration)
    budget_manager.update_resource_usage_and_cost("llm_budget", model_response)

    workspace = agent_factory.get_system_class("workspace", configuration)
    workspace_path = workspace.setup_workspace(configuration)
    # TODO: Provision memory backend. Waiting on interface to stabilize

    await message_broker.send_message(
        "agent_setup_complete",
        {"message": "Agent setup complete."},
    )
