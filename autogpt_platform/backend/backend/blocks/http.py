import json
from enum import Enum

import requests

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField


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

        response = requests.request(
            input_data.method.value,
            input_data.url,
            headers=input_data.headers,
            json=input_data.body,
        )
        if response.status_code // 100 == 2:
            yield "response", response.json()
        elif response.status_code // 100 == 4:
            yield "client_error", response.json()
        elif response.status_code // 100 == 5:
            yield "server_error", response.json()
        else:
            raise ValueError(f"Unexpected status code: {response.status_code}")
