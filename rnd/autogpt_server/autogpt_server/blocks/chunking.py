import re
from typing import List, Optional

from pydantic import BaseModel, Field

from autogpt_server.data.block import Block, BlockOutput, BlockSchema

class ChunkingConfig(BaseModel):
    chunk_size: int = Field(default=1000, description="Maximum number of characters per chunk")
    overlap: int = Field(default=100, description="Number of characters to overlap between chunks")
    split_on: Optional[str] = Field(default=None, description="Regular expression to split on (e.g., '\n\n' for paragraphs)")

class ChunkingBlock(Block):
    class Input(BlockSchema):
        text: str = Field(description="Text to be chunked")
        config: ChunkingConfig = Field(description="Chunking configuration")

    class Output(BlockSchema):
        chunks: List[str] = Field(description="List of text chunks")

    def __init__(self):
        super().__init__(
            id="7d9e8f3a-2b5c-4e1d-9f3a-2b5c4e1d9f3a",
            input_schema=ChunkingBlock.Input,
            output_schema=ChunkingBlock.Output,
            test_input={
                "text": "This is a long piece of text that needs to be chunked. " * 20,
                "config": {
                    "chunk_size": 100,
                    "overlap": 20,
                    "split_on": None
                }
            },
            test_output=("chunks", [
                "This is a long piece of text that needs to be chunked. This is a long piece of text that needs to be chunked. ",
                "to be chunked. This is a long piece of text that needs to be chunked. This is a long piece of text that needs ",
                "text that needs to be chunked. This is a long piece of text that needs to be chunked. This is a long piece of ",
                "of text that needs to be chunked. This is a long piece of text that needs to be chunked. This is a long piece ",
                "piece of text that needs to be chunked. This is a long piece of text that needs to be chunked. This is a long "
            ]),
        )

    def chunk_text(self, text: str, config: ChunkingConfig) -> List[str]:
        if config.split_on:
            # Split on the specified pattern
            segments = re.split(config.split_on, text)
            chunks = []
            current_chunk = ""
            for segment in segments:
                if len(current_chunk) + len(segment) > config.chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = segment
                else:
                    current_chunk += (" " if current_chunk else "") + segment
            if current_chunk:
                chunks.append(current_chunk.strip())
        else:
            chunks = []
            start = 0
            while start < len(text):
                end = start + config.chunk_size
                chunk = text[start:end]
                chunks.append(chunk)
                start = end - config.overlap

        return chunks

    def run(self, input_data: Input) -> BlockOutput:
        chunks = self.chunk_text(input_data.text, input_data.config)
        yield "chunks", chunks