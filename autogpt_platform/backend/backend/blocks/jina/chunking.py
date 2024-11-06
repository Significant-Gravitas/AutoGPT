from backend.blocks.jina._auth import (
    JinaCredentials,
    JinaCredentialsField,
    JinaCredentialsInput,
)
from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField
from backend.util.request import requests


class JinaChunkingBlock(Block):
    class Input(BlockSchema):
        texts: list = SchemaField(description="List of texts to chunk")

        credentials: JinaCredentialsInput = JinaCredentialsField()
        max_chunk_length: int = SchemaField(
            description="Maximum length of each chunk", default=1000
        )
        return_tokens: bool = SchemaField(
            description="Whether to return token information", default=False
        )

    class Output(BlockSchema):
        chunks: list = SchemaField(description="List of chunked texts")
        tokens: list = SchemaField(
            description="List of token information for each chunk", optional=True
        )

    def __init__(self):
        super().__init__(
            id="806fb15e-830f-4796-8692-557d300ff43c",
            description="Chunks texts using Jina AI's segmentation service",
            categories={BlockCategory.AI, BlockCategory.TEXT},
            input_schema=JinaChunkingBlock.Input,
            output_schema=JinaChunkingBlock.Output,
        )

    def run(
        self, input_data: Input, *, credentials: JinaCredentials, **kwargs
    ) -> BlockOutput:
        url = "https://segment.jina.ai/"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {credentials.api_key.get_secret_value()}",
        }

        all_chunks = []
        all_tokens = []

        for text in input_data.texts:
            data = {
                "content": text,
                "return_tokens": str(input_data.return_tokens).lower(),
                "return_chunks": "true",
                "max_chunk_length": str(input_data.max_chunk_length),
            }

            response = requests.post(url, headers=headers, json=data)
            result = response.json()

            all_chunks.extend(result.get("chunks", []))
            if input_data.return_tokens:
                all_tokens.extend(result.get("tokens", []))

        yield "chunks", all_chunks
        if input_data.return_tokens:
            yield "tokens", all_tokens
