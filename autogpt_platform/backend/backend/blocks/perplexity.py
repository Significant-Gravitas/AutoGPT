# flake8: noqa: E501
import logging
from enum import Enum
from typing import Any, Literal

import openai
from pydantic import SecretStr

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import (
    APIKeyCredentials,
    CredentialsField,
    CredentialsMetaInput,
    NodeExecutionStats,
    SchemaField,
)
from backend.integrations.providers import ProviderName
from backend.util.logging import TruncatedLogger

logger = TruncatedLogger(logging.getLogger(__name__), "[Perplexity-Block]")


class PerplexityModel(str, Enum):
    """Perplexity sonar models available via OpenRouter"""

    SONAR = "perplexity/sonar"
    SONAR_PRO = "perplexity/sonar-pro"
    SONAR_DEEP_RESEARCH = "perplexity/sonar-deep-research"


PerplexityCredentials = CredentialsMetaInput[
    Literal[ProviderName.OPEN_ROUTER], Literal["api_key"]
]

TEST_CREDENTIALS = APIKeyCredentials(
    id="test-perplexity-creds",
    provider="open_router",
    api_key=SecretStr("mock-openrouter-api-key"),
    title="Mock OpenRouter API key",
    expires_at=None,
)
TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}


def PerplexityCredentialsField() -> PerplexityCredentials:
    return CredentialsField(
        description="OpenRouter API key for accessing Perplexity models.",
    )


class PerplexityBlock(Block):
    class Input(BlockSchema):
        prompt: str = SchemaField(
            description="The query to send to the Perplexity model.",
            placeholder="Enter your query here...",
        )
        model: PerplexityModel = SchemaField(
            title="Perplexity Model",
            default=PerplexityModel.SONAR,
            description="The Perplexity sonar model to use.",
            advanced=False,
        )
        credentials: PerplexityCredentials = PerplexityCredentialsField()
        system_prompt: str = SchemaField(
            title="System Prompt",
            default="",
            description="Optional system prompt to provide context to the model.",
            advanced=True,
        )
        max_tokens: int | None = SchemaField(
            advanced=True,
            default=None,
            description="The maximum number of tokens to generate.",
        )

    class Output(BlockSchema):
        response: str = SchemaField(
            description="The response from the Perplexity model."
        )
        annotations: list[dict[str, Any]] = SchemaField(
            description="List of URL citations and annotations from the response."
        )
        error: str = SchemaField(description="Error message if the API call failed.")

    def __init__(self):
        super().__init__(
            id="c8a5f2e9-8b3d-4a7e-9f6c-1d5e3c9b7a4f",
            description="Query Perplexity's sonar models with real-time web search capabilities and receive annotated responses with source citations.",
            categories={BlockCategory.AI, BlockCategory.SEARCH},
            input_schema=PerplexityBlock.Input,
            output_schema=PerplexityBlock.Output,
            test_input={
                "prompt": "What is the weather today?",
                "model": PerplexityModel.SONAR,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("response", "The weather varies by location..."),
                ("annotations", list),
            ],
            test_mock={
                "call_perplexity": lambda *args, **kwargs: {
                    "response": "The weather varies by location...",
                    "annotations": [
                        {
                            "type": "url_citation",
                            "url_citation": {
                                "title": "weather.com",
                                "url": "https://weather.com",
                            },
                        }
                    ],
                }
            },
        )
        self.execution_stats = NodeExecutionStats()

    async def call_perplexity(
        self,
        credentials: APIKeyCredentials,
        model: PerplexityModel,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Call Perplexity via OpenRouter and extract annotations."""
        client = openai.AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=credentials.api_key.get_secret_value(),
        )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://agpt.co",
                    "X-Title": "AutoGPT",
                },
                model=model.value,
                messages=messages,
                max_tokens=max_tokens,
            )

            if not response.choices:
                raise ValueError("No response from Perplexity via OpenRouter.")

            # Extract the response content
            response_content = response.choices[0].message.content or ""

            # Extract annotations if present in the message
            annotations = []
            if hasattr(response.choices[0].message, "annotations"):
                # If annotations are directly available
                annotations = response.choices[0].message.annotations
            else:
                # Check if there's a raw response with annotations
                raw = getattr(response.choices[0].message, "_raw_response", None)
                if isinstance(raw, dict) and "annotations" in raw:
                    annotations = raw["annotations"]

            if not annotations and hasattr(response, "model_extra"):
                # Check model_extra for annotations
                model_extra = response.model_extra
                if isinstance(model_extra, dict):
                    # Check in choices
                    if "choices" in model_extra and len(model_extra["choices"]) > 0:
                        choice = model_extra["choices"][0]
                        if "message" in choice and "annotations" in choice["message"]:
                            annotations = choice["message"]["annotations"]

            # Also check the raw response object for annotations
            if not annotations:
                raw = getattr(response, "_raw_response", None)
                if isinstance(raw, dict):
                    # Check various possible locations for annotations
                    if "annotations" in raw:
                        annotations = raw["annotations"]
                    elif "choices" in raw and len(raw["choices"]) > 0:
                        choice = raw["choices"][0]
                        if "message" in choice and "annotations" in choice["message"]:
                            annotations = choice["message"]["annotations"]

            # Update execution stats
            if response.usage:
                self.execution_stats.input_token_count = response.usage.prompt_tokens
                self.execution_stats.output_token_count = (
                    response.usage.completion_tokens
                )

            return {"response": response_content, "annotations": annotations or []}

        except Exception as e:
            logger.error(f"Error calling Perplexity: {e}")
            raise

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        logger.debug(f"Running Perplexity block with model: {input_data.model}")

        try:
            result = await self.call_perplexity(
                credentials=credentials,
                model=input_data.model,
                prompt=input_data.prompt,
                system_prompt=input_data.system_prompt,
                max_tokens=input_data.max_tokens,
            )

            yield "response", result["response"]
            yield "annotations", result["annotations"]

        except Exception as e:
            error_msg = f"Error calling Perplexity: {str(e)}"
            logger.error(error_msg)
            yield "error", error_msg
