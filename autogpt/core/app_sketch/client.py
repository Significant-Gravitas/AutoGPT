import logging

from autogpt.core.app_sketch.server import bootstrap_agent, launch_agent
from autogpt.core.messaging.simple import SimpleMessageBroker


def configure_application_logging(
    application_logger: logging.Logger,
    file_based_configuration: dict,
    command_line_arguments: dict,
):
    application_logger.setLevel(logging.DEBUG)


def start_message_broker(application_logger) -> SimpleMessageBroker:
    # This can take a lot of different forms. E.g. if I'm building an app that connects
    # to an existing agent on a server, I might just need to connect to a websocket.
    # If I'm launching a bunch of agents on a cluster, I might need to stand up a
    # message broker process like redis and launch an Agent factory process.
    # We're going to do some in-process setup for now by wiring up a simple message
    # broker and registering a listener for the user_objective message.

    message_broker = SimpleMessageBroker(application_logger)

    # This would happen on the "server" side, but hook here because we're
    # doing everything in-process.
    # FIXME: I'm misuing message channels here. Message broker needs to be set up
    #   so that register_listener can subscribe to a channel and filter messages
    #   based on their metadata.
    message_broker.register_listener(
        "user_objective",
        bootstrap_agent,
    )
    message_broker.register_listener(
        "launch_agent",
        launch_agent,
    )

    message_broker.register_listener(
        "agent_setup",
        lambda message: application_logger.info(message.content["message"]),
    )
    message_broker.register_listener(
        "agent_setup_complete",
        lambda x: None,
    )

    return message_broker


def run_auto_gpt(
    file_based_configuration: dict,  # Need to figure out what's in here
    command_line_arguments: dict,  # Need to figure out what's in here
):
    # Configure logging before we do anything else.
    # Application logs need a place to live.
    application_logger = logging.getLogger("autogpt")
    configure_application_logging(
        application_logger,
        file_based_configuration,
        command_line_arguments,
    )

    message_broker = start_message_broker(application_logger)

    # This application either starts an existing agent or builds a new one.
    if command_line_arguments["agent_name"] is None:
        # Find out the user's objective for the new agent.
        user_objective = input(...)
        # Construct a message to send to the agent.  Real format TBD.
        user_objective_message = {
            # This might be like a websocket, (hostname, port) tuple, or something else.
            "message_broker": message_broker,
            # These will need structures with some strongly-enforced fields to be
            # interpreted by the bootstrapping system.
            # Command line arguments override file-based configuration.
            "user_configuration": file_based_configuration.update(
                command_line_arguments
            ),
            "user_objective": user_objective,
        }
        message_broker.send_message("user_objective", user_objective_message)

    launch_agent_message = {
        "message_broker": message_broker,
        "agent": command_line_arguments["agent_name"],
    }
    message_broker.send_message("launch_agent", launch_agent_message)
