"""JSON encode and decode blocks for processing structured data in workflows."""

from typing import Any

import orjson

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import SchemaField
from backend.util.json import dumps, loads


class JSONEncoderBlock(Block):

    class Input(BlockSchemaInput):
        data: Any = SchemaField(
            description=(
                "The data structure/value (object, list, string, etc.) to encode "
                "into a JSON string."
            ),
            placeholder='e.g., {"key": "value"}',
        )

    class Output(BlockSchemaOutput):
        json_str: str = SchemaField(
            description="The JSON string representation of the input data."
        )

    def __init__(self):
        super().__init__(
            id="9a9d20c5-5b4d-4e94-8022-83b6cb72648a",
            description="Encodes any value or data structure into a JSON string.",
            categories={BlockCategory.DATA},
            input_schema=JSONEncoderBlock.Input,
            output_schema=JSONEncoderBlock.Output,
            test_input={"data": {"name": "AutoGPT", "active": True}},
            test_output=[
                ("json_str", '{"name":"AutoGPT","active":true}'),
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        try:
            json_str = dumps(input_data.data)
            yield "json_str", json_str
        except (orjson.JSONEncodeError, RecursionError, TypeError, ValueError) as e:
            raise ValueError(f"JSON Encoding Error: {str(e)}") from e


class JSONDecoderBlock(Block):

    class Input(BlockSchemaInput):
        json_str: str = SchemaField(
            description="The JSON string to decode.",
            placeholder='e.g., {"key": "value"}',
        )

    class Output(BlockSchemaOutput):
        data: Any = SchemaField(
            description="The value as decoded from the JSON string."
        )

    def __init__(self):
        super().__init__(
            id="2b935639-65bc-48fd-9f88-823cd706fcd9",
            description=(
                "Decodes a JSON string into the value or data structure, it "
                "represents, e.g. an object, list, string, or number."
            ),
            categories={BlockCategory.DATA},
            input_schema=JSONDecoderBlock.Input,
            output_schema=JSONDecoderBlock.Output,
            test_input={"json_str": '{"name":"AutoGPT","active":true}'},
            test_output=[
                ("data", {"name": "AutoGPT", "active": True}),
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        try:
            parsed_data = loads(input_data.json_str)
            yield "data", parsed_data
        except orjson.JSONDecodeError as e:
            raise ValueError(f"JSON Decoding Error: {str(e)}") from e
