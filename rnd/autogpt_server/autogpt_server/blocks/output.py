from pathlib import Path

from autogpt_server.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from autogpt_server.data.model import SchemaField


class FileWriterBlock(Block):
    class Input(BlockSchema):
        text: str = SchemaField(
            description="The text content to write to the file.",
            placeholder="Hello, world!",
        )
        output_path: str = SchemaField(
            description="The path where the file should be written.",
            placeholder="/path/to/output/file.txt",
        )

    class Output(BlockSchema):
        status: str = SchemaField(description="Status of the file writing operation.")
        file_path: str = SchemaField(description="The path of the written file.")
        error: str = SchemaField(description="Error message if the operation failed.")

    def __init__(self):
        super().__init__(
            id="6f7b2dcb-4a78-4e3f-b0f1-88132e1b89df",  # Replace with a proper UUID
            description="Writes the given text to a file at the specified output path.",
            categories={BlockCategory.OUTPUT},
            input_schema=FileWriterBlock.Input,
            output_schema=FileWriterBlock.Output,
            test_input={"text": "Hello, world!", "output_path": "/tmp/test_output.txt"},
            test_output=[("status", "success"), ("file_path", "/tmp/test_output.txt")],
        )

    def run(self, input_data: Input) -> BlockOutput:
        try:
            output_path = Path(input_data.output_path)

            # Ensure the directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write the text to the file
            with open(output_path, "w") as file:
                file.write(input_data.text)

            yield "status", "success"
            yield "file_path", str(output_path)
        except Exception as e:
            yield "error", f"Failed to write file: {str(e)}"
