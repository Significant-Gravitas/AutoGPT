"""OpenAI Embeddings payloads for deterministic memory searches."""

import json
from typing import Any

EMBEDDING_DIMENSIONS = 1536


class EmbeddingError(ValueError):
    pass


def response(body: dict[str, Any]) -> dict[str, Any]:
    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise EmbeddingError("model must be a non-empty string")

    count = _input_count(body.get("input"))
    dimensions = body.get("dimensions", EMBEDDING_DIMENSIONS)
    if (
        not isinstance(dimensions, int)
        or isinstance(dimensions, bool)
        or dimensions < 1
    ):
        raise EmbeddingError("dimensions must be a positive integer")

    tokens = max(1, len(json.dumps(body.get("input"))) // 4)
    vector = [0.1] * dimensions
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": vector, "index": index}
            for index in range(count)
        ],
        "model": model,
        "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
    }


def error(message_text: str) -> dict[str, Any]:
    return {
        "error": {
            "message": f"scripted-model: {message_text}",
            "type": "invalid_request_error",
            "param": None,
            "code": "scripted_model",
        }
    }


def _input_count(value: Any) -> int:
    if isinstance(value, str):
        return 1
    if not isinstance(value, list) or not value:
        raise EmbeddingError("input must be a string or a non-empty array")
    if all(isinstance(item, str) for item in value):
        return len(value)
    if all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        return 1
    if all(
        isinstance(item, list)
        and item
        and all(
            isinstance(token, int) and not isinstance(token, bool) for token in item
        )
        for item in value
    ):
        return len(value)
    raise EmbeddingError("input array has unsupported values")
