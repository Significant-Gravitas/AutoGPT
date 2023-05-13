import logging
import requests
import json

# This could be a client library, or they could simply use requests to hit the API.
# For now, we'll do everything in-process.
from autogpt.core.runner.server import application_server


def configure_client_application_logging(
    application_logger: logging.Logger,
    user_configuration: dict,
):
    application_logger.setLevel(logging.DEBUG)


def make_request(content, **metadata):
    """Convert args to a json string."""
    request = object()
    request.json = {
        "content": content,
        "metadata": metadata,
    }
    return request


async def run_auto_gpt(
    user_configuration: dict,  # Need to figure out what's in here
):
    # Configure logging before we do anything else.
    # Application logs need a place to live.
    client_logger = logging.getLogger("autogpt_client_application")
    configure_client_application_logging(
        client_logger,
        user_configuration,
    )

    # This application either starts an existing agent or builds a new one.
    if user_configuration["agent_name"] is None:
        # Find out the user's objective for the new agent.
        user_objective = input("...")
        # Construct a message to send to the agent.  Real format TBD.
        user_objective_message = {
            "user_objective": user_objective,
            # These will need structures with some strongly-enforced fields to be
            # interpreted by the bootstrapping system.
            "user_configuration": user_configuration,
        }
        # Post to https endpoint here maybe instead
        response = await application_server.boostrap_new_agent(
            make_request(user_objective_message),
        )
        if response.status_code == 200:
            user_configuration["agent_name"] = response.json()["agent_name"]
            # Display some stuff to the user about the new agent.
        else:
            raise RuntimeError("Failed to bootstrap agent.")

    launch_agent_message = {
        "agent": user_configuration["agent_name"],
    }
    response = await application_server.launch_agent(
        make_request(launch_agent_message),
    )
    if response.status_code != 200:
        # Display some stuff to the user
        raise RuntimeError("Failed to launch agent.")

    # HACK: this should operate in a GET to first ask for the agent plan,
    #   then a POST to give feedback. That requires some asynchrony, so I'm
    #   doing both things in one call here for now.
    user_feedback = "Get started"

    while True:
        feedback_message = {
            "user_feedback": user_feedback,
        }
        response = await application_server.give_agent_feedback(
            make_request(feedback_message),
        )
        if response.status_code == 200:
            print(response.json["content"])
            # Display some stuff to the user
            user_feedback = input("...")

        else:
            raise RuntimeError("Main loop failed")


def run():
    body = json.dumps(
        {"ai_name": "HelloBot", "ai_role": "test", "ai_goals": ["goal1", "goal2"]}
    )

    header = {"Content-Type": "application/json", "openai_api_key": "asdf"}
    response = requests.post(
        "http://localhost:8080/api/v1/agents", data=body, headers=header
    )
    print(response.content.decode("utf-8"))
