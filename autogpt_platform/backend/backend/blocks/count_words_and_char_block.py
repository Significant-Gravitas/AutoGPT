from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField


class WordCharacterCountBlock(Block):
    class Input(BlockSchema):
        text: str = SchemaField(
            description="Input text to count words and characters",
            placeholder="Enter your text here",
            advanced=False,
        )

    class Output(BlockSchema):
        word_count: int = SchemaField(description="Number of words in the input text")
        character_count: int = SchemaField(
            description="Number of characters in the input text"
        )
        error: str = SchemaField(
            description="Error message if the counting operation failed"
        )

    def __init__(self):
        super().__init__(
            id="ab2a782d-22cf-4587-8a70-55b59b3f9f90",
            description="Counts the number of words and characters in a given text.",
            categories={BlockCategory.TEXT},
            input_schema=WordCharacterCountBlock.Input,
            output_schema=WordCharacterCountBlock.Output,
            test_input={"text": "Hello, how are you?"},
            test_output=[("word_count", 4), ("character_count", 19)],
        )

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        try:
            text = input_data.text
            word_count = len(text.split())
            character_count = len(text)

            yield "word_count", word_count
            yield "character_count", character_count

        except Exception as e:
            yield "error", str(e)
