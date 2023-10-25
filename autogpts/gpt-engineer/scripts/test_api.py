"""This is just a demo to test api.py."""

from time import sleep

import requests


def post_data(url, extra_arguments):
    """
    Make an HTTP POST request with extra_arguments as data.

    Args:
    - url (str): The URL to which the POST request should be sent.
    - extra_arguments (dict): A dictionary of data that needs to be sent in the POST request.

    Returns:
    - response: The response from the server.
    """

    response = requests.post(url, json=extra_arguments)
    return response


if __name__ == "__main__":
    URL_BASE = "http://127.0.0.1:8000"

    arguments = {
        "input": "We are writing snake in python. MVC components split \
        in separate files. Keyboard control.",  # our prompt
        "additional_input": {"improve_option": False},
    }

    # create a task
    response = post_data(f"{URL_BASE}/agent/tasks", arguments)
    print(response.json())
    task_id = response.json()["task_id"]

    sleep(1)  # this is not needed

    # execute the step for our task
    response = post_data(f"{URL_BASE}/agent/tasks/{task_id}/steps", {})
    print(response.json())
