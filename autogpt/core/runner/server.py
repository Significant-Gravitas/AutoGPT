import logging

from autogpt.core.agent.base import Agent
from autogpt.core.agent.factory import AgentFactory
from autogpt.core.messaging.simple import (
    Message,
    MessageMetadata,
    Role,
    SimpleMessageBroker,
)

##################################################
# Hacking stuff together for an in-process model #
##################################################


def get_message_broker() -> SimpleMessageBroker:
    message_channel_name = "autogpt"

    if hasattr(get_message_broker, "_MESSAGE_BROKER"):
        return get_message_broker._MESSAGE_BROKER

    message_broker = SimpleMessageBroker()
    message_broker.create_message_channel(message_channel_name)

    # def is_user_message(message: Message):
    #     metadata = message.metadata
    #     return (
    #         metadata.sender.role == Role.USER
    #     )
    #
    # def boostrap_filter(message: Message):
    #     metadata = message.metadata
    #     return (
    #         is_user_message(message)
    #         and metadata.additional_metadata[
    #             "instruction"] == "bootstrap_agent"
    #     )
    #
    # message_broker.register_listener(
    #     message_channel_name,
    #     listener=bootstrap_agent,
    #     message_filter=boostrap_filter,
    # )
    #
    # def launch_filter(message: Message):
    #     metadata = message.metadata
    #     return (
    #             is_user_message(message)
    #             and metadata.additional_metadata[
    #                 "instruction"] == "launch_agent"
    #     )
    #
    # message_broker.register_listener(
    #     message_channel_name,
    #     listener=launch_agent,
    #     message_filter=launch_filter,
    # )
    #
    # def is_server_message(message: Message):
    #     metadata = message.metadata
    #     return metadata.sender.role in [Role.AGENT, Role.AGENT_FACTORY]
    #
    # def log_filter(message: Message):
    #     metadata = message.metadata
    #     return (
    #             is_server_message(message)
    #             and metadata.additional_metadata["message_type"] == "log"
    #     )
    #
    # def error_filter(message: Message):
    #     metadata = message.metadata
    #     return (
    #             is_server_message(message)
    #             and metadata.additional_metadata["message_type"] == "error"
    #     )
    #
    # def error_callback(message: Message):
    #     raise RuntimeError(message.content["message"])
    #
    # message_broker.register_listener(
    #     message_channel_name,
    #     listener=error_callback,
    #     message_filter=error_filter,
    # )

    get_message_broker._MESSAGE_BROKER = message_broker

    return get_message_broker._MESSAGE_BROKER


class FakeApplicationServer:
    """The interface to the 'application server' process.

    This could be a restful API or something.

    """

    def __init__(self):
        self._message_broker = get_message_broker()
        self._user_emitter = self._message_broker.get_emitter(
            channel_name="autogpt",
            sender_name="autogpt-user",
            sender_role=Role.USER,
        )

    def list_agents(self, request):
        """List all agents."""
        pass

    def boostrap_new_agent(self, request):
        """Bootstrap a new agent."""
        self._user_emitter.send_message(**request.json)
        response = object()
        response.status_code = 200
        return response

    def launch_agent(self, request):
        """Launch an agent."""
        self._user_emitter.send_message(**request.json)
        response = object()
        response.status_code = 200
        return response

    def get_agent_plan(self, request):
        """Get the plan for an agent."""
        # TODO: need a clever hack here to get the agent plan since we'd have natural
        #  asynchrony here with a webserver.
        pass

    def give_agent_feedback(self, request):
        """Give feedback to an agent."""
        self._user_emitter.send_message(**request.json)
        response = object()
        response.status_code = 200
        return response


application_server = FakeApplicationServer()


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
        channel_name="autogpt",
        sender_name="autogpt-agent-factory",
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
            content={
                "message": "Configuration errors encountered, aborting agent setup."
            },
            message_type="error",
        )
        return

    agent_factory_emitter.send_message(
        content={
            "message": "Agent configuration compiled. Constructing initial agent plan from user objective."
        },
        message_type="log",
    )

    agent_planner = agent_factory.get_system_class("planner", configuration)
    # TODO: is this a class method?  Or do we have the planner be partially initialized
    #  without access to any resources since this precedes Agent creation?
    objective_prompt = agent_planner.construct_objective_prompt_from_user_input(
        user_objective,
    )
    agent_factory_emitter.send_message(
        content={
            "message": "Translated user input into objective prompt.",
            "objective_prompt": objective_prompt,
        },
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
