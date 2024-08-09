import logging
from enum import Enum
from typing import List

import openai
from pydantic import BaseModel, Field

from autogpt_server.data.block import Block, BlockOutput, BlockSchema
from autogpt_server.util import json

logger = logging.getLogger(__name__)

class EmbeddingModel(str, Enum):
    ada_002 = "text-embedding-ada-002"

class EmbeddingConfig(BaseModel):
    model: EmbeddingModel
    api_key: str

class EmbeddingBlock(Block):
    class Input(BlockSchema):
        config: EmbeddingConfig
        texts: List[str] = Field(description="List of texts to create embeddings for")

    class Output(BlockSchema):
        embeddings: List[List[float]]
        error: str

    def __init__(self):
        super().__init__(
            id="8f7e9a1c-3b7a-4b0f-9f1a-1c3b7a4b0f9f",
            input_schema=EmbeddingBlock.Input,
            output_schema=EmbeddingBlock.Output,
            test_input={
                "config": {
                    "model": "text-embedding-ada-002",
                    "api_key": "fake-api-key",
                },
                "texts": ["Hello, world!", "AutoGPT is amazing"],
            },
            test_output=("embeddings", [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            test_mock={"create_embeddings": lambda *args, **kwargs: [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]},
        )

    @staticmethod
    def create_embeddings(api_key: str, model: EmbeddingModel, texts: List[str]) -> List[List[float]]:
        openai.api_key = api_key
        response = openai.embeddings.create(
            model=model,
            input=texts,
        )
        return [embedding.embedding for embedding in response.data]

    def run(self, input_data: Input) -> BlockOutput:
        try:
            embeddings = self.create_embeddings(
                input_data.config.api_key,
                input_data.config.model,
                input_data.texts
            )
            yield "embeddings", embeddings
        except Exception as e:
            error_message = f"Error creating embeddings: {str(e)}"
            logger.error(error_message)
            yield "error", error_message