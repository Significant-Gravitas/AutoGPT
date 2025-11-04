import codecs
import uuid

from backend.data.block import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import SchemaField


class TextEncoderBlock(Block):
    class Input(BlockSchemaInput):
        text: str = SchemaField(
            description="A string to be encoded with escape sequences",
            placeholder='Your text with newlines and "quotes" to be escaped',
        )

    class Output(BlockSchemaOutput):
        encoded_text: str = SchemaField(
            description="The encoded text with escape sequences added"
        )

    def __init__(self):
        super().__init__(
            id="3681f9ef-9558-54fe-95d8-81e768034342",  # Fixed UUID4 format
            description="Encodes a string by adding escape sequences for special characters",
            categories={BlockCategory.TEXT},
            input_schema=TextEncoderBlock.Input,
            output_schema=TextEncoderBlock.Output,
            test_input={
                "text": """Hello
World!
This is a "quoted" string."""
            },
            test_output=[
                (
                    "encoded_text",
                    """Hello\\nWorld!\\nThis is a \\"quoted\\" string.""",
                )
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        encoded_text = codecs.encode(input_data.text, "unicode_escape").decode("ascii")
        yield "encoded_text", encoded_text

