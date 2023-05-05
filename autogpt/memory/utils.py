import numpy as np
import openai

from autogpt.config import Config
from autogpt.llm.llm_utils import metered, retry_openai_api

Embedding = np.ndarray[np.float32]
"""Embedding vector"""
TText = list[int]
"""Token array representing text"""


@metered
@retry_openai_api()
def get_embedding(
    input: str | TText | list[str | TText],
) -> Embedding | list[Embedding]:
    """Get an embedding from the ada model.

    Args:
        input: Input text to get embeddings for, encoded as a string or array of tokens.
            Multiple inputs may be given as a list of strings or token arrays.

    Returns:
        List[float]: The embedding.
    """
    cfg = Config()
    multiple = isinstance(input, list) and not isinstance(input[0], int)

    if isinstance(input, str):
        input = input.replace("\n", " ")
    elif multiple and isinstance(input[0], str):
        input = [text.replace("\n", " ") for text in input]

    model = cfg.embedding_model
    if cfg.use_azure:
        kwargs = {"engine": cfg.get_azure_deployment_id_for_model(model)}
    else:
        kwargs = {"model": model}

    embeddings = openai.Embedding.create(
        input=input,
        api_key=cfg.openai_api_key,
        **kwargs,
    ).data

    if not multiple:
        return embeddings[0]["embedding"]

    embeddings = sorted(embeddings, key=lambda x: x["index"])
    return [d["embedding"] for d in embeddings]
