import codecs

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField


class TextDecoderBlock(Block):
    class Input(BlockSchema):
        text: str = SchemaField(
            description="A string containing escaped characters to be decoded",
            placeholder='Your entire text block with \\n and \\" escaped characters',
        )

    class Output(BlockSchema):
        decoded_text: str = SchemaField(
            description="The decoded text with escape sequences processed"
        )

    def __init__(self):
        super().__init__(
            id="2570e8fe-8447-43ed-84c7-70d657923231",
            description="Decodes a string containing escape sequences into actual text",
            categories={BlockCategory.TEXT},
            input_schema=TextDecoderBlock.Input,
            output_schema=TextDecoderBlock.Output,
            test_input={"text": """Hello\nWorld!\nThis is a \"quoted\" string."""},
            test_output=[
                (
                    "decoded_text",
                    """Hello
World!
This is a "quoted" string.""",
                )
            ],
        )

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        decoded_text = codecs.decode(input_data.text, "unicode_escape")
        yield "decoded_text", decoded_text
