"""
It is essential that Agent (the parent of PlannerAgent) provide a method to load an object from a dictionary/json object for webapps
"""
import json
from pathlib import Path
from time import sleep

import click
import requests

from autogpts.autogpt.autogpt.core.agents import PlannerAgent
from autogpts.autogpt.autogpt.core.runner.client_lib.logging import \
    get_client_logger

BASE_URL = "http://localhost:8080/api/v1"
MAX_ATTEMPTS = 3


def send_request_with_retry(url, method="GET", json_payload=None):
    for attempt in range(MAX_ATTEMPTS):
        response = requests.request(method, url, json=json_payload)
        if response.status_code == 200:
            return response.json()
        else:
            print(
                f"Request failed with status code {response.status_code}. Retrying..."
            )
            sleep(1)  # Wait for 1 second before retrying
    print(f"Maximum number of attempts reached. Unable to complete the request.")
    return None


input("Press Enter to Test all the end points : \n")


# GET /agents
print("\nGET /agents")
response = send_request_with_retry(f"{BASE_URL}/agents")
if response:
    print(" response:", response)

# POST /agent
print("\nPOST /agent")
response = send_request_with_retry(f"{BASE_URL}/agent", method="POST")
if response:
    print(" response:", response)
    agent_id = response.get("agent_id")

# GET /agent/{agent_id}
print(f"\nGET /agent/{agent_id}")
response = send_request_with_retry(f"{BASE_URL}/agent/{agent_id}")
if response:
    print(" response:", response)

# POST /agent/{agent_id}/start
start_request_body = {"message": "your message here", "start": True}
print(f"\nPOST /agent/{agent_id}/start")
response = send_request_with_retry(
    f"{BASE_URL}/agent/{agent_id}/start", method="POST", json_payload=start_request_body
)
if response:
    print(" response:", response)

# POST /agent/{agent_id}/message
message_request_body = {
    "message": "This is the tests message sent by the cli & so far it's working",
    "start": True,
}
print(f"\nPOST /agent/{agent_id}/message")
response = send_request_with_retry(
    f"{BASE_URL}/agent/{agent_id}/message",
    method="POST",
    json_payload=message_request_body,
)
if response:
    print(" response:", response)

# GET /agent/{agent_id}/messagehistory
print(f"\nGET /agent/{agent_id}/messagehistory")
response = send_request_with_retry(f"{BASE_URL}/agent/{agent_id}/messagehistory")
if response:
    print(" response:", response)

# GET /agent/{agent_id}/lastmessage
print(f"\nGET /agent/{agent_id}/lastmessage")
response = send_request_with_retry(f"{BASE_URL}/agent/{agent_id}/lastmessage")
if response:
    print(" response:", response)

input("\n\nPress Enter to start AutoGPT\n\n")


def run():
    """Run the Auto-GPT CLI client."""

    client_logger = get_client_logger()
    client_logger.debug("Getting agent settings")

    response = requests.get(f"{BASE_URL}/agents/")
    response_data = response.json()
    simple_agent_as_dict = response_data.get("agents")[0]

    # Todo : Needs to create a PlannerAgent from a dict
    exit(
        "Demo stops here as a method PlannerAgent.load_from_dict(simple_agent_as_dict) is needed"
    )
    agent = PlannerAgent.load_from_dict(simple_agent_as_dict)
    print("agent is loaded")

    response = requests.post(f"{BASE_URL}/agent/{agent_id}/start")
    print(response.json())

    user_input: str = ""
    while user_input.lower != "n":
        user_input = click.prompt(
            "Should the agent proceed with this ability?",
            default="y",
        )
        message_request_body = {"message": user_input, "start": True}  # or False
        response = requests.post(
            f"{BASE_URL}/agent/{agent_id}/message", json=message_request_body
        )
        print(f"\nPOST /agent/{agent_id}/message response:", response.json())
