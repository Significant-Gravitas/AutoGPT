import json
import logging

import click
import requests


async def run_auto_gpt(user_configuration: dict):
    """Run the Auto-GPT CLI client."""
    client_logger = get_client_logger()

    agent_name = (
        user_configuration.get("planning", {})
        .get("configuration", {})
        .get("agent_name", "")
    )

    if not agent_name:
        user_objective = click.prompt("What do you want Auto-GPT to do?")
        # Construct a message to send to the agent.  Real format TBD.
        user_objective_message = {
            "user_objective": user_objective,
            "user_configuration": user_configuration,
        }

    # This application either starts an existing agent or builds a new one.
    if user_configuration["agent_name"] is None:
        # Find out the user's objective for the new agent.
        user_objective = input("...")

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


def make_request(content, **metadata):
    """Convert args to a json string."""
    header = {"Content-Type": "application/json"}
    body = json.dumps(
        {"content": content, "metadata": metadata},
    )
    request = {
        "url": "http://localhost:8080/api/v1/agents",
    }

    return request


# def run():
#     body = json.dumps(
#         {"ai_name": "HelloBot", "ai_role": "test", "ai_goals": ["goal1", "goal2"]}
#     )
#
#     header = {"Content-Type": "application/json", "openai_api_key": "asdf"}
#     print("Sending: ", header, body)
#     response = requests.post(
#         "http://localhost:8080/api/v1/agents", data=body, headers=header
#     )
#     print(response.content.decode("utf-8"))
