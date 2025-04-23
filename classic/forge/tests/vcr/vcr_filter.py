import contextlib
import json
import re
from io import BytesIO
from typing import Any

from vcr.request import Request

HOSTNAMES_TO_CACHE: list[str] = [
    "api.openai.com",
    "localhost:50337",
    "duckduckgo.com",
]

IGNORE_REQUEST_HEADERS: set[str | re.Pattern] = {
    "Authorization",
    "Cookie",
    "OpenAI-Organization",
    "X-OpenAI-Client-User-Agent",
    "User-Agent",
    re.compile(r"X-Stainless-[\w\-]+", re.IGNORECASE),
}

LLM_MESSAGE_REPLACEMENTS: list[dict[str, str]] = [
    {
        "regex": r"\w{3} \w{3} {1,2}\d{1,2} \d{2}:\d{2}:\d{2} \d{4}",
        "replacement": "Tue Jan  1 00:00:00 2000",
    },
    {
        "regex": r"<selenium.webdriver.chrome.webdriver.WebDriver[^>]*>",
        "replacement": "",
    },
]

OPENAI_URL = "api.openai.com"


def before_record_request(request: Request) -> Request | None:
    if not should_cache_request(request):
        return None

    request = filter_request_headers(request)
    request = freeze_request(request)
    return request


def should_cache_request(request: Request) -> bool:
    return any(hostname in request.url for hostname in HOSTNAMES_TO_CACHE)


def filter_request_headers(request: Request) -> Request:
    for header_name in list(request.headers):
        if any(
            (
                (type(ignore) is str and ignore.lower() == header_name.lower())
                or (isinstance(ignore, re.Pattern) and ignore.match(header_name))
            )
            for ignore in IGNORE_REQUEST_HEADERS
        ):
            del request.headers[header_name]
    return request


def freeze_request(request: Request) -> Request:
    if not request or not request.body:
        return request

    with contextlib.suppress(ValueError):
        request.body = freeze_request_body(
            json.loads(
                request.body.getvalue()
                if isinstance(request.body, BytesIO)
                else request.body
            )
        )

    return request


def freeze_request_body(body: dict) -> bytes:
    """Remove any dynamic items from the request body"""

    if "messages" not in body:
        return json.dumps(body, sort_keys=True).encode()

    if "max_tokens" in body:
        del body["max_tokens"]

    for message in body["messages"]:
        if "content" in message and "role" in message:
            if message["role"] == "system":
                message["content"] = replace_message_content(
                    message["content"], LLM_MESSAGE_REPLACEMENTS
                )

    return json.dumps(body, sort_keys=True).encode()


def replace_message_content(content: str, replacements: list[dict[str, str]]) -> str:
    for replacement in replacements:
        pattern = re.compile(replacement["regex"])
        content = pattern.sub(replacement["replacement"], content)

    return content


def before_record_response(response: dict[str, Any]) -> dict[str, Any]:
    if "Transfer-Encoding" in response["headers"]:
        del response["headers"]["Transfer-Encoding"]
    return response
