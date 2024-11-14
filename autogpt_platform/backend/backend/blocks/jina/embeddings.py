from backend.blocks.jina._auth import (
    JinaCredentials,
    JinaCredentialsField,
    JinaCredentialsInput,
)
from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField
from backend.util.request import requests


class JinaEmbeddingBlock(Block):
    class Input(BlockSchema):
        texts: list = SchemaField(description="List of texts to embed")
        credentials: JinaCredentialsInput = JinaCredentialsField()
        model: str = SchemaField(
            description="Jina embedding model to use",
            default="jina-embeddings-v2-base-en",
        )

    class Output(BlockSchema):
        embeddings: list = SchemaField(description="List of embeddings")

    def __init__(self):
        super().__init__(
            id="7c56b3ab-62e7-43a2-a2dc-4ec4245660b6",
            description="Generates embeddings using Jina AI",
            categories={BlockCategory.AI},
            input_schema=JinaEmbeddingBlock.Input,
            output_schema=JinaEmbeddingBlock.Output,
        )

    def run(
        self, input_data: Input, *, credentials: JinaCredentials, **kwargs
    ) -> BlockOutput:
        url = "https://api.jina.ai/v1/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {credentials.api_key.get_secret_value()}",
        }
        data = {"input": input_data.texts, "model": input_data.model}
        response = requests.post(url, headers=headers, json=data)
        embeddings = [e["embedding"] for e in response.json()["data"]]
        yield "embeddings", embeddings
