from typing import Any, overload

import numpy as np

from autogpt.config import Config
from autogpt.llm.base import TText
from autogpt.llm.providers import openai as iopenai
from autogpt.logs import logger

Embedding = list[np.float32] | np.ndarray[Any, np.dtype[np.float32]]
"""Embedding vector"""


@overload
def get_embedding(input: str | TText) -> Embedding:
    ...


@overload
def get_embedding(input: list[str] | list[TText]) -> list[Embedding]:
    ...


def get_embedding(
    input: str | TText | list[str] | list[TText], config: Config
) -> Embedding | list[Embedding]:
    """Get an embedding from the ada model.

    Args:
        input: Input text to get embeddings for, encoded as a string or array of tokens.
            Multiple inputs may be given as a list of strings or token arrays.

    Returns:
        List[float]: The embedding.
    """
    multiple = isinstance(input, list) and all(not isinstance(i, int) for i in input)

    if isinstance(input, str):
        input = input.replace("\n", " ")
    elif multiple and isinstance(input[0], str):
        input = [text.replace("\n", " ") for text in input]

    model = config.embedding_model
    kwargs = {"model": model}
    kwargs.update(config.get_openai_credentials(model))

    logger.debug(
        f"Getting embedding{f's for {len(input)} inputs' if multiple else ''}"
        f" with model '{model}'"
        + (f" via Azure deployment '{kwargs['engine']}'" if config.use_azure else "")
    )
    if config.use_azure:
        breakpoint()

    embeddings = iopenai.create_embedding(
        input,
        **kwargs,
    ).data

    if not multiple:
        return embeddings[0]["embedding"]

    embeddings = sorted(embeddings, key=lambda x: x["index"])
    return [d["embedding"] for d in embeddings]
