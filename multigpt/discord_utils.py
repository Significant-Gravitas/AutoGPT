import json

import requests


def send_message(content, username):
    data = {"content": content, "username": username}

    webhook_url = None

    response = requests.post(webhook_url, data=json.dumps(data), headers={"Content-Type": "application/json"})

    if response.status_code != 204:
        raise ValueError(
            f"Request to webhook returned an error {response.status_code}, the response is:\n{response.text}")
