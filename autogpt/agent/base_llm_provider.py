"""The base class for all LLM providers."""
import abc


class LLMProvider(abc.ABC):
    """The base class for all LLM providers."""

    def __init__(self, url: str, api_key: str) -> None:
        self.api_key = api_key
        self.url = url
