import ipaddress
import json
import socket
from enum import Enum
from urllib.parse import urlparse

import requests

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField
from backend.util.settings import Config


class HttpMethod(Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    OPTIONS = "OPTIONS"
    HEAD = "HEAD"


def validate_url(url: str) -> str:
    """
    To avoid SSRF attacks, the URL should not be a private IP address
    unless it is whitelisted in TRUST_ENDPOINTS_FOR_REQUESTS config.
    """
    if any(url.startswith(origin) for origin in Config().trust_endpoints_for_requests):
        return url

    parsed_url = urlparse(url)
    hostname = parsed_url.hostname

    if not hostname:
        raise ValueError(f"Invalid URL: Unable to determine hostname from {url}")

    try:
        host = socket.gethostbyname_ex(hostname)
        for ip in host[2]:
            ip_addr = ipaddress.ip_address(ip)
            if ip_addr.is_global:
                return url
        raise ValueError(
            f"Access to private or untrusted IP address at {hostname} is not allowed."
        )
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Invalid or unresolvable URL: {url}") from e


class SendWebRequestBlock(Block):
    class Input(BlockSchema):
        url: str = SchemaField(
            description="The URL to send the request to",
            placeholder="https://api.example.com",
        )
        method: HttpMethod = SchemaField(
            description="The HTTP method to use for the request",
            default=HttpMethod.POST,
        )
        headers: dict[str, str] = SchemaField(
            description="The headers to include in the request",
            default={},
        )
        body: object = SchemaField(
            description="The body of the request",
            default={},
        )

    class Output(BlockSchema):
        response: object = SchemaField(description="The response from the server")
        client_error: object = SchemaField(description="The error on 4xx status codes")
        server_error: object = SchemaField(description="The error on 5xx status codes")

    def __init__(self):
        super().__init__(
            id="6595ae1f-b924-42cb-9a41-551a0611c4b4",
            description="This block makes an HTTP request to the given URL.",
            categories={BlockCategory.OUTPUT},
            input_schema=SendWebRequestBlock.Input,
            output_schema=SendWebRequestBlock.Output,
        )

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        if isinstance(input_data.body, str):
            input_data.body = json.loads(input_data.body)

        validated_url = validate_url(input_data.url)

        response = requests.request(
            input_data.method.value,
            validated_url,
            headers=input_data.headers,
            json=input_data.body,
            allow_redirects=False,
        )
        if response.status_code // 100 == 2:
            yield "response", response.json()
        elif response.status_code // 100 == 4:
            yield "client_error", response.json()
        elif response.status_code // 100 == 5:
            yield "server_error", response.json()
        else:
            raise ValueError(f"Unexpected status code: {response.status_code}")
