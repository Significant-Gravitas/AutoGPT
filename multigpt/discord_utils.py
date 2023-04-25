import json

import requests


def send_message(content, username, webhook_url, avatar_url):
    data = {"content": content, "username": username}
    data['avatar_url'] = avatar_url
    response = requests.post(webhook_url, data=json.dumps(data), headers={"Content-Type": "application/json"})

    if response.status_code != 204:
        raise ValueError(
            f"Request to webhook returned an error {response.status_code}, the response is:\n{response.text}")


def send_embed_message(content, system_name, username, webhook_url, avatar_url, avatar_url_system):
    data = {
        "content": "",
        "avatar_url": avatar_url_system,
        "username": system_name,
        "embeds": [
            {
                "type": "rich",
                "title": content,
                "description": "",
                "color": 0xed0707,
                "author": {
                    "name": username,
                    "icon_url": avatar_url
                }
            }
        ]
    }

    response = requests.post(webhook_url, data=json.dumps(data), headers={"Content-Type": "application/json"})

    if response.status_code != 204:
        raise ValueError(
            f"Request to webhook returned an error {response.status_code}, the response is:\n{response.text}")

