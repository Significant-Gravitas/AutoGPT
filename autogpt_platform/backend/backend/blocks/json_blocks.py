"""JSON encode and decode blocks for processing structured data in workflows."""

from typing import Any

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
    """
    JSON Encoder Block.

    Converts Python data structures such as dictionaries, lists,
    strings, booleans, and numbers into a JSON-formatted string.
    """

    class Input(BlockSchemaInput):

        data: Any = SchemaField(
            description="The data structure (dictionary, list, string, etc.) to encode into a JSON string.",
            placeholder='e.g., {"key": "value"}',
        )

    class Output(BlockSchemaOutput):

        json_str: str = SchemaField(
            description="The resulting JSON string representation."
        )

    def __init__(self):
        super().__init__(
            id="9a9d20c5-5b4d-4e94-8022-83b6cb72648a",
            description="Encodes a Python object (dictionary, list, etc.) into a JSON string.",
            categories={BlockCategory.DATA},
            input_schema=JSONEncoderBlock.Input,
            output_schema=JSONEncoderBlock.Output,
            test_input={"data": {"name": "AutoGPT", "active": True}},
            test_output=[
                ("json_str", '{"name":"AutoGPT","active":true}'),
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        """
        Encode the provided Python object into a JSON string.

        Args:
            input_data (Input):
                Validated input containing the Python object to encode.

            **kwargs:
                Additional runtime arguments passed by the block framework.

        Yields:
            tuple[str, str]:
                - "json_str": Encoded JSON string on success.
        """
        try:
            json_str = dumps(input_data.data)
            yield "json_str", json_str
        except Exception as e:
            raise ValueError(f"JSON Encoding Error: {str(e)}") from e


class JSONDecoderBlock(Block):
    """
    JSON Decoder Block.

    Converts a JSON-formatted string into its corresponding
    Python object representation such as a dictionary or list.
    """

    class Input(BlockSchemaInput):
        """Input schema for the JSON decoder block."""

        json_str: str = SchemaField(
            description="The JSON string to parse and convert to a Python object.",
            placeholder='e.g., {"key": "value"}',
        )

    class Output(BlockSchemaOutput):
        """Output schema for the JSON decoder block."""

        data: Any = SchemaField(description="The decoded Python dictionary or list.")

    def __init__(self):
        super().__init__(
            id="2b935639-65bc-48fd-9f88-823cd706fcd9",
            description="Decodes a JSON string into a Python object (dictionary, list, etc.).",
            categories={BlockCategory.DATA},
            input_schema=JSONDecoderBlock.Input,
            output_schema=JSONDecoderBlock.Output,
            test_input={"json_str": '{"name":"AutoGPT","active":true}'},
            test_output=[
                ("data", {"name": "AutoGPT", "active": True}),
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        """
        Decode the provided JSON string into a Python object.

        Args:
            input_data (Input):
                Validated input containing the JSON string to decode.

            **kwargs:
                Additional runtime arguments passed by the block framework.

        Yields:
            tuple[str, Any]:
                - "data": Parsed Python object on success.
        """
        try:
            parsed_data = loads(input_data.json_str)
            yield "data", parsed_data
        except Exception as e:
            raise ValueError(f"JSON Decoding Error: {str(e)}") from e
