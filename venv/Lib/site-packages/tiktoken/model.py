from __future__ import annotations

from .core import Encoding
from .registry import get_encoding

# TODO: these will likely be replaced by an API endpoint
MODEL_PREFIX_TO_ENCODING: dict[str, str] = {
    # chat
    "gpt-4-": "cl100k_base",  # e.g., gpt-4-0314, etc., plus gpt-4-32k
    "gpt-3.5-turbo-": "cl100k_base",  # e.g, gpt-3.5-turbo-0301, -0401, etc.
}

MODEL_TO_ENCODING: dict[str, str] = {
    # chat
    "gpt-4": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    # text
    "text-davinci-003": "p50k_base",
    "text-davinci-002": "p50k_base",
    "text-davinci-001": "r50k_base",
    "text-curie-001": "r50k_base",
    "text-babbage-001": "r50k_base",
    "text-ada-001": "r50k_base",
    "davinci": "r50k_base",
    "curie": "r50k_base",
    "babbage": "r50k_base",
    "ada": "r50k_base",
    # code
    "code-davinci-002": "p50k_base",
    "code-davinci-001": "p50k_base",
    "code-cushman-002": "p50k_base",
    "code-cushman-001": "p50k_base",
    "davinci-codex": "p50k_base",
    "cushman-codex": "p50k_base",
    # edit
    "text-davinci-edit-001": "p50k_edit",
    "code-davinci-edit-001": "p50k_edit",
    # embeddings
    "text-embedding-ada-002": "cl100k_base",
    # old embeddings
    "text-similarity-davinci-001": "r50k_base",
    "text-similarity-curie-001": "r50k_base",
    "text-similarity-babbage-001": "r50k_base",
    "text-similarity-ada-001": "r50k_base",
    "text-search-davinci-doc-001": "r50k_base",
    "text-search-curie-doc-001": "r50k_base",
    "text-search-babbage-doc-001": "r50k_base",
    "text-search-ada-doc-001": "r50k_base",
    "code-search-babbage-code-001": "r50k_base",
    "code-search-ada-code-001": "r50k_base",
    # open source
    "gpt2": "gpt2",
}


def encoding_for_model(model_name: str) -> Encoding:
    """Returns the encoding used by a model."""
    encoding_name = None
    if model_name in MODEL_TO_ENCODING:
        encoding_name = MODEL_TO_ENCODING[model_name]
    else:
        # Check if the model matches a known prefix
        # Prefix matching avoids needing library updates for every model version release
        # Note that this can match on non-existent models (e.g., gpt-3.5-turbo-FAKE)
        for model_prefix, model_encoding_name in MODEL_PREFIX_TO_ENCODING.items():
            if model_name.startswith(model_prefix):
                return get_encoding(model_encoding_name)

    if encoding_name is None:
        raise KeyError(
            f"Could not automatically map {model_name} to a tokeniser. "
            "Please use `tiktok.get_encoding` to explicitly get the tokeniser you expect."
        ) from None

    return get_encoding(encoding_name)
