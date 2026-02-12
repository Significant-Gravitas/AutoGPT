"""Text encoding block for converting special characters to escape sequences."""

import codecs

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import SchemaField


class TextEncoderBlock(Block):
    """
    Encodes a string by converting special characters into escape sequences.

    This block is the inverse of TextDecoderBlock. It takes text containing
    special characters (like newlines, tabs, etc.) and converts them into
    their escape sequence representations (e.g., newline becomes \\n).
    """

    class Input(BlockSchemaInput):
        """Input schema for TextEncoderBlock."""

        text: str = SchemaField(
            description="A string containing special characters to be encoded",
            placeholder="Your text with newlines and quotes to encode",
        )

    class Output(BlockSchemaOutput):
        """Output schema for TextEncoderBlock."""

        encoded_text: str = SchemaField(
            description="The encoded text with special characters converted to escape sequences"
        )
        error: str = SchemaField(description="Error message if encoding fails")

    def __init__(self):
        super().__init__(
            id="5185f32e-4b65-4ecf-8fbb-873f003f09d6",
            description="Encodes a string by converting special characters into escape sequences",
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
                    """Hello\\nWorld!\\nThis is a "quoted" string.""",
                )
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        """
        Encode the input text by converting special characters to escape sequences.

        Args:
            input_data: The input containing the text to encode.
            **kwargs: Additional keyword arguments (unused).

        Yields:
            The encoded text with escape sequences, or an error message if encoding fails.
        """
        try:
            encoded_text = codecs.encode(input_data.text, "unicode_escape").decode(
                "utf-8"
            )
            yield "encoded_text", encoded_text
        except Exception as e:
            yield "error", f"Encoding error: {str(e)}"
