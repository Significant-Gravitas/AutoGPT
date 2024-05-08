import logging
from typing import Any, Sequence, overload

import numpy as np
from forge.config.config import Config
from forge.llm.providers import EmbeddingModelProvider

logger = logging.getLogger(__name__)

Embedding = list[float] | list[np.float32] | np.ndarray[Any, np.dtype[np.float32]]
"""Embedding vector"""

TText = Sequence[int]
"""Tokenized text"""


@overload
async def get_embedding(
    input: str | TText, config: Config, embedding_provider: EmbeddingModelProvider
) -> Embedding:
    ...


@overload
async def get_embedding(
    input: list[str] | list[TText],
    config: Config,
    embedding_provider: EmbeddingModelProvider,
) -> list[Embedding]:
    ...


async def get_embedding(
    input: str | TText | list[str] | list[TText],
    config: Config,
    embedding_provider: EmbeddingModelProvider,
) -> Embedding | list[Embedding]:
    """Get an embedding from the ada model.

    Args:
        input: Input text to get embeddings for, encoded as a string or array of tokens.
            Multiple inputs may be given as a list of strings or token arrays.
        embedding_provider: The provider to create embeddings.

    Returns:
        List[float]: The embedding.
    """
    multiple = isinstance(input, list) and all(not isinstance(i, int) for i in input)

    if isinstance(input, str):
        input = input.replace("\n", " ")
    elif multiple and isinstance(input[0], str):
        input = [text.replace("\n", " ") for text in input]

    model = config.embedding_model

    logger.debug(
        f"Getting embedding{f's for {len(input)} inputs' if multiple else ''}"
        f" with model '{model}'"
    )

    if not multiple:
        return (
            await embedding_provider.create_embedding(
                text=input,
                model_name=model,
                embedding_parser=lambda e: e,
            )
        ).embedding
    else:
        embeddings = []
        for text in input:
            result = await embedding_provider.create_embedding(
                text=text,
                model_name=model,
                embedding_parser=lambda e: e,
            )
            embeddings.append(result.embedding)
        return embeddings
