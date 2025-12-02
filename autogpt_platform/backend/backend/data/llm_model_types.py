from typing import NamedTuple


class ModelMetadata(NamedTuple):
    provider: str
    context_window: int
    max_output_tokens: int | None
