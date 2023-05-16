import json

import requests


def run():
    body = json.dumps(
        {"ai_name": "HelloBot", "ai_role": "test", "ai_goals": ["goal1", "goal2"]}
    )

    header = {"Content-Type": "application/json", "openai_api_key": "asdf"}
    print("Sending: ", header, body)
    response = requests.post(
        "http://localhost:8080/api/v1/agents", data=body, headers=header
    )
    print(response.content.decode("utf-8"))
