import json
import logging
from enum import Enum
from typing import Any

from requests.exceptions import HTTPError, RequestException

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField
from backend.util.request import requests

logger = logging.getLogger(name=__name__)


class HttpMethod(Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    OPTIONS = "OPTIONS"
    HEAD = "HEAD"


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
        json_format: bool = SchemaField(
            title="JSON format",
            description="Whether to send and receive body as JSON",
            default=True,
        )
        body: Any = SchemaField(
            description="The body of the request",
            default=None,
        )

    class Output(BlockSchema):
        response: object = SchemaField(description="The response from the server")
        client_error: object = SchemaField(description="Errors on 4xx status codes")
        server_error: object = SchemaField(description="Errors on 5xx status codes")
        error: str = SchemaField(description="Errors for all other exceptions")

    def __init__(self):
        super().__init__(
            id="6595ae1f-b924-42cb-9a41-551a0611c4b4",
            description="This block makes an HTTP request to the given URL.",
            categories={BlockCategory.OUTPUT},
            input_schema=SendWebRequestBlock.Input,
            output_schema=SendWebRequestBlock.Output,
        )

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        body = input_data.body

        if input_data.json_format:
            if isinstance(body, str):
                try:
                    # Try to parse as JSON first
                    body = json.loads(body)
                except json.JSONDecodeError:
                    # If it's not valid JSON and just plain text,
                    # we should send it as plain text instead
                    input_data.json_format = False

        try:
            response = requests.request(
                input_data.method.value,
                input_data.url,
                headers=input_data.headers,
                json=body if input_data.json_format else None,
                data=body if not input_data.json_format else None,
            )
            result = response.json() if input_data.json_format else response.text
            yield "response", result

        except HTTPError as e:
            # Handle error responses
            try:
                result = e.response.json() if input_data.json_format else str(e)
            except json.JSONDecodeError:
                result = str(e)

            if 400 <= e.response.status_code < 500:
                yield "client_error", result
            elif 500 <= e.response.status_code < 600:
                yield "server_error", result
            else:
                error_msg = (
                    "Unexpected status code "
                    f"{e.response.status_code} '{e.response.reason}'"
                )
                logger.warning(error_msg)
                yield "error", error_msg

        except RequestException as e:
            # Handle other request-related exceptions
            yield "error", str(e)

        except Exception as e:
            # Catch any other unexpected exceptions
            yield "error", str(e)
