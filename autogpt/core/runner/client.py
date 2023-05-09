import logging

from autogpt.core.messaging.simple import Message, Role, SimpleMessageBroker
from autogpt.core.runner.server import bootstrap_agent, launch_agent


def configure_application_logging(
    application_logger: logging.Logger,
    user_configuration: dict,
):
    application_logger.setLevel(logging.DEBUG)


def start_message_broker(
    message_channel_name: str,
    application_name: str,
    application_logger: logging.Logger,
) -> SimpleMessageBroker:
    # This can take a lot of different forms. E.g. if I'm building an app that connects
    # to an existing agent on a server, I might just need to connect to a websocket.
    # If I'm launching a bunch of agents on a cluster, I might need to stand up a
    # message broker process like redis and launch an Agent factory process.
    # We're going to do some in-process setup for now by wiring up a simple message
    # broker and registering a listener for the user_objective message.
    message_broker = SimpleMessageBroker(application_logger)
    message_broker.create_message_channel(message_channel_name)

    # Some of this would happen on the "server" side, but hook here because we're
    # doing everything in-process.
    def is_user_message(message: Message):
        metadata = message.metadata
        return (
            metadata.sender.role == Role.USER
            and metadata.sender.name == application_name
        )

    def boostrap_filter(message: Message):
        metadata = message.metadata
        return (
            is_user_message(message)
            and metadata.additional_metadata["instruction"] == "bootstrap_agent"
        )

    message_broker.register_listener(
        message_channel_name,
        listener=bootstrap_agent,
        message_filter=boostrap_filter,
    )

    def launch_filter(message: Message):
        metadata = message.metadata
        return (
            is_user_message(message)
            and metadata.additional_metadata["instruction"] == "launch_agent"
        )

    message_broker.register_listener(
        message_channel_name,
        listener=launch_agent,
        message_filter=launch_filter,
    )

    def is_server_message(message: Message):
        metadata = message.metadata
        return metadata.sender.role in [Role.AGENT, Role.AGENT_FACTORY]

    def log_filter(message: Message):
        metadata = message.metadata
        return (
            is_server_message(message)
            and metadata.additional_metadata["message_type"] == "log"
        )

    message_broker.register_listener(
        message_channel_name,
        listener=lambda message: application_logger.info(message.content["message"]),
        message_filter=log_filter,
    )

    def error_filter(message: Message):
        metadata = message.metadata
        return (
            is_server_message(message)
            and metadata.additional_metadata["message_type"] == "error"
        )

    def error_callback(message: Message):
        raise RuntimeError(message.content["message"])

    message_broker.register_listener(
        message_channel_name,
        listener=error_callback,
        message_filter=error_filter,
    )

    return message_broker


def run_auto_gpt(
    user_configuration: dict,  # Need to figure out what's in here
):
    # Configure logging before we do anything else.
    # Application logs need a place to live.
    application_logger = logging.getLogger("autogpt")
    configure_application_logging(
        application_logger,
        user_configuration,
    )

    # Could make lots of channels here like logging, agent_status, etc.
    message_channel_name = "autogpt"
    application_name = "autogpt-client"
    message_broker = start_message_broker(
        message_channel_name,
        application_name,
        application_logger,
    )
    user_emitter = message_broker.get_emitter(
        channel_name=message_channel_name,
        sender_name=application_name,
        sender_role=Role.USER,
    )

    # This application either starts an existing agent or builds a new one.
    if user_configuration["agent_name"] is None:
        # Find out the user's objective for the new agent.
        user_objective = input(...)
        # Construct a message to send to the agent.  Real format TBD.
        user_objective_message = {
            "user_objective": user_objective,
            # These will need structures with some strongly-enforced fields to be
            # interpreted by the bootstrapping system.
            "user_configuration": user_configuration,
            # This might be like a websocket, (hostname, port) tuple, or
            # something else.
            "message_broker": message_broker,
        }
        user_emitter.send_message(
            content=user_objective_message,
            instruction="bootstrap_agent",
        )

    launch_agent_message = {
        "message_broker": message_broker,
        "agent": user_configuration["agent_name"],
    }
    user_emitter.send_message(
        content=launch_agent_message,
        instruction="launch_agent",
    )
