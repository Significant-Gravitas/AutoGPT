import json
import re


def replace_timestamp_in_request(request):
    # Check if the request body contains a JSON object

    try:
        if not request.body:
            return request
        body = json.loads(request.body)
    except ValueError:
        return request

    if "messages" not in body:
        return request

    for message in body["messages"]:
        if "content" in message and "role" in message and message["role"] == "system":
            timestamp_regex = re.compile(r"\w{3} \w{3} \d{2} \d{2}:\d{2}:\d{2} \d{4}")
            message["content"] = timestamp_regex.sub(
                "Tue Jan 01 00:00:00 2000", message["content"]
            )

    request.body = json.dumps(body)
    return request
