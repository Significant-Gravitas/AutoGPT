"""The base class for all LLM providers."""
from autogpt.agent.LLM.base_llm_provider import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    """The base class for all LLM providers."""

    def __init__(self, url: str, api_key: str) -> None:
        self.api_key = api_key
        self.url = url
