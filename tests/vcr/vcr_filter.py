import json
import os
import re
from typing import Any, Dict, List

from tests.conftest import PROXY

REPLACEMENTS: List[Dict[str, str]] = [
    {
        "regex": r"\w{3} \w{3} {1,2}\d{1,2} \d{2}:\d{2}:\d{2} \d{4}",
        "replacement": "Tue Jan  1 00:00:00 2000",
    },
    {
        "regex": r"<selenium.webdriver.chrome.webdriver.WebDriver[^>]*>",
        "replacement": "",
    },
]

ALLOWED_HOSTNAMES: List[str] = [
    "api.openai.com",
    "localhost:50337",
]

if PROXY:
    ALLOWED_HOSTNAMES.append(PROXY)
    ORIGINAL_URL = PROXY
else:
    ORIGINAL_URL = "no_ci"

NEW_URL = "api.openai.com"


def replace_message_content(content: str, replacements: List[Dict[str, str]]) -> str:
    for replacement in replacements:
        pattern = re.compile(replacement["regex"])
        content = pattern.sub(replacement["replacement"], content)

    return content


def replace_timestamp_in_request(request: Any) -> Any:
    try:
        if not request or not request.body:
            return request
        body = json.loads(request.body)
    except ValueError:
        return request

    if "messages" not in body:
        return request
    body[
        "max_tokens"
    ] = 0  # this field is inconsistent between requests and not used at the moment.
    for message in body["messages"]:
        if "content" in message and "role" in message:
            if message["role"] == "system":
                message["content"] = replace_message_content(
                    message["content"], REPLACEMENTS
                )

    request.body = json.dumps(body)
    return request


def before_record_response(response: Dict[str, Any]) -> Dict[str, Any]:
    if "Transfer-Encoding" in response["headers"]:
        del response["headers"]["Transfer-Encoding"]
    return response


def before_record_request(request: Any) -> Any:
    request = replace_request_hostname(request, ORIGINAL_URL, NEW_URL)

    filtered_request = filter_hostnames(request)
    filtered_request_without_dynamic_data = replace_timestamp_in_request(
        filtered_request
    )
    return filtered_request_without_dynamic_data


from urllib.parse import urlparse, urlunparse


def replace_request_hostname(request: Any, original_url: str, new_hostname: str) -> Any:
    parsed_url = urlparse(request.uri)

    if parsed_url.hostname in original_url:
        new_path = parsed_url.path.replace("/proxy_function", "")
        request.uri = urlunparse(
            parsed_url._replace(netloc=new_hostname, path=new_path, scheme="https")
        )

    return request


def filter_hostnames(request: Any) -> Any:
    # Add your implementation here for filtering hostnames
    if any(hostname in request.url for hostname in ALLOWED_HOSTNAMES):
        return request
    else:
        return None
