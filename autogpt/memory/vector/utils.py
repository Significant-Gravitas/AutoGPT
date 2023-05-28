from typing import Any, overload

import numpy as np
import openai

from autogpt.config import Config
from autogpt.llm.utils import metered, retry_openai_api
from autogpt.logs import logger

Embedding = list[np.float32] | np.ndarray[Any, np.dtype[np.float32]]
"""Embedding vector"""
TText = list[int]
"""Token array representing text"""


@overload
def get_embedding(input: str | TText) -> Embedding:
    ...


@overload
def get_embedding(input: list[str] | list[TText]) -> list[Embedding]:
    ...


@metered
@retry_openai_api()
def get_embedding(
    input: str | TText | list[str] | list[TText],
) -> Embedding | list[Embedding]:
    """Get an embedding from the ada model.

    Args:
        input: Input text to get embeddings for, encoded as a string or array of tokens.
            Multiple inputs may be given as a list of strings or token arrays.

    Returns:
        List[float]: The embedding.
    """
    cfg = Config()
    multiple = isinstance(input, list) and all(not isinstance(i, int) for i in input)

    if isinstance(input, str):
        input = input.replace("\n", " ")
    elif multiple and isinstance(input[0], str):
        input = [text.replace("\n", " ") for text in input]

    model = cfg.embedding_model
    if cfg.use_azure:
        kwargs = {"engine": cfg.get_azure_deployment_id_for_model(model)}
    else:
        kwargs = {"model": model}

    logger.debug(
        f"Getting embedding{f's for {len(input)} inputs' if multiple else ''}"
        f" with model '{model}'"
        + (f" via Azure deployment '{kwargs['engine']}'" if cfg.use_azure else "")
    )

    embeddings = openai.Embedding.create(
        input=input,
        api_key=cfg.openai_api_key,
        **kwargs,
    ).data

    if not multiple:
        return embeddings[0]["embedding"]

    embeddings = sorted(embeddings, key=lambda x: x["index"])
    return [d["embedding"] for d in embeddings]
