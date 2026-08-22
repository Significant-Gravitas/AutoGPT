import logging
from enum import Enum
from typing import Any, Literal

import openai
from pydantic import SecretStr, field_validator

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.block import BlockInput
from backend.data.model import (
    APIKeyCredentials,
    CredentialsField,
    CredentialsMetaInput,
    NodeExecutionStats,
    SchemaField,
)
from backend.integrations.providers import ProviderName
from backend.util.exceptions import BlockExecutionError
from backend.util.logging import TruncatedLogger

logger = TruncatedLogger(logging.getLogger(__name__), "[DeepSeek-Block]")

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekModel(str, Enum):
    """DeepSeek chat and reasoning models."""

    CHAT = "deepseek-chat"
    REASONER = "deepseek-reasoner"


def _sanitize_deepseek_model(value: Any) -> DeepSeekModel:
    """Return a valid DeepSeekModel, falling back to CHAT for invalid values."""
    if isinstance(value, DeepSeekModel):
        return value
    try:
        return DeepSeekModel(value)
    except ValueError:
        logger.warning(
            f"Invalid DeepSeekModel '{value}', "
            f"falling back to {DeepSeekModel.CHAT.value}"
        )
        return DeepSeekModel.CHAT


DeepSeekCredentials = CredentialsMetaInput[
    Literal[ProviderName.DEEPSEEK], Literal["api_key"]
]

TEST_CREDENTIALS = APIKeyCredentials(
    id="test-deepseek-creds",
    provider="deepseek",
    api_key=SecretStr("mock-deepseek-api-key"),
    title="Mock DeepSeek API key",
    expires_at=None,
)

TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}


def DeepSeekCredentialsField() -> DeepSeekCredentials:
    return CredentialsField(
        description="DeepSeek API key for accessing DeepSeek models.",
    )


class DeepSeekBlock(Block):
    class Input(BlockSchemaInput):
        prompt: str = SchemaField(
            description="The prompt or query to send to the DeepSeek model.",
            placeholder="Enter your prompt here...",
        )
        model: DeepSeekModel = SchemaField(
            title="DeepSeek Model",
            default=DeepSeekModel.CHAT,
            description="The DeepSeek model to use (deepseek-chat or deepseek-reasoner).",
            advanced=False,
        )
        system_prompt: str = SchemaField(
            title="System Prompt",
            default="",
            description="Optional system prompt to provide context to the model.",
            advanced=True,
        )
        temperature: float | None = SchemaField(
            title="Temperature",
            default=None,
            ge=0.0,
            le=2.0,
            description="Sampling temperature between 0 and 2. Ignored by deepseek-reasoner.",
            advanced=True,
        )
        max_tokens: int | None = SchemaField(
            title="Max Tokens",
            default=None,
            ge=0,
            description="The maximum number of tokens to generate.",
            advanced=True,
        )
        json_mode: bool = SchemaField(
            title="JSON Mode",
            default=False,
            description="Enforce JSON object output format.",
            advanced=True,
        )
        stream: bool = SchemaField(
            title="Stream Response",
            default=False,
            description="Whether to stream the response from the API.",
            advanced=True,
        )
        credentials: DeepSeekCredentials = DeepSeekCredentialsField()

        @field_validator("model", mode="before")
        @classmethod
        def fallback_invalid_model(cls, v: Any) -> DeepSeekModel:
            """Fall back to CHAT if model value is invalid."""
            return _sanitize_deepseek_model(v)

        @classmethod
        def validate_data(
            cls,
            data: BlockInput,
            exclude_fields: set[str] | None = None,
        ) -> str | None:
            """Sanitize the model field before JSON schema validation."""
            model_value = data.get("model")
            if model_value is not None:
                data["model"] = _sanitize_deepseek_model(model_value).value
            return super().validate_data(data, exclude_fields=exclude_fields)

    class Output(BlockSchemaOutput):
        response: str = SchemaField(
            description="The text response from the DeepSeek model."
        )
        reasoning_content: str = SchemaField(
            default="",
            description="Chain-of-thought reasoning content from deepseek-reasoner models.",
        )

    def __init__(self):
        super().__init__(
            id="d3e9f5a1-7c2b-4e8a-9f1d-6b3a4c5e7f8a",
            description="Execute chat and reasoning prompts with DeepSeek AI models (DeepSeek-V3 and DeepSeek-R1) and extract reasoning tokens.",
            categories={BlockCategory.AI, BlockCategory.TEXT},
            input_schema=DeepSeekBlock.Input,
            output_schema=DeepSeekBlock.Output,
            test_input={
                "prompt": "Explain quantum computing in one sentence.",
                "model": DeepSeekModel.CHAT,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("response", "Quantum computing leverages qubits for computation."),
                ("reasoning_content", ""),
            ],
            test_mock={
                "call_deepseek": lambda *args, **kwargs: {
                    "response": "Quantum computing leverages qubits for computation.",
                    "reasoning_content": "",
                }
            },
        )
        self.execution_stats = NodeExecutionStats()

    async def call_deepseek(
        self,
        credentials: APIKeyCredentials,
        model: DeepSeekModel,
        prompt: str,
        system_prompt: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Call DeepSeek API and extract response and reasoning content."""
        client = openai.AsyncOpenAI(
            base_url=DEEPSEEK_BASE_URL,
            api_key=credentials.api_key.get_secret_value(),
        )

        # DeepSeek-reasoner does not support custom temperature
        extra_kwargs: dict[str, Any] = {}
        if temperature is not None and model != DeepSeekModel.REASONER:
            extra_kwargs["temperature"] = temperature

        if json_mode:
            extra_kwargs["response_format"] = {"type": "json_object"}
            # Ensure "json" is present in the prompt or system_prompt to satisfy API constraint
            if "json" not in f"{system_prompt} {prompt}".lower():
                if system_prompt:
                    system_prompt = f"{system_prompt}\nRespond with a valid JSON object."
                else:
                    system_prompt = "Respond with a valid JSON object."

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            if stream:
                stream_response = await client.chat.completions.create(
                    model=model.value,
                    messages=messages,
                    max_tokens=max_tokens,
                    stream=True,
                    stream_options={"include_usage": True},
                    **extra_kwargs,
                )

                collected_content: list[str] = []
                collected_reasoning: list[str] = []
                prompt_tokens = 0
                completion_tokens = 0

                async for chunk in stream_response:
                    if chunk.usage:
                        prompt_tokens = chunk.usage.prompt_tokens
                        completion_tokens = chunk.usage.completion_tokens

                    if not chunk.choices:
                        continue

                    delta = chunk.choices[0].delta
                    if delta.content:
                        collected_content.append(delta.content)

                    # Extract reasoning_content from delta if present
                    reasoning_delta = getattr(delta, "reasoning_content", None)
                    if not reasoning_delta and hasattr(delta, "model_extra"):
                        model_extra = delta.model_extra
                        if isinstance(model_extra, dict):
                            reasoning_delta = model_extra.get("reasoning_content")
                    if reasoning_delta:
                        collected_reasoning.append(reasoning_delta)

                response_text = "".join(collected_content)
                reasoning_text = "".join(collected_reasoning)

                self.execution_stats.input_token_count = prompt_tokens
                self.execution_stats.output_token_count = completion_tokens

                return {
                    "response": response_text,
                    "reasoning_content": reasoning_text,
                }

            # Non-streaming call
            response = await client.chat.completions.create(
                model=model.value,
                messages=messages,
                max_tokens=max_tokens,
                stream=False,
                **extra_kwargs,
            )

            if not response.choices:
                raise ValueError("No choices returned from DeepSeek API.")

            choice = response.choices[0]
            message = choice.message
            response_content = message.content or ""

            # Extract reasoning_content from message
            reasoning_content = getattr(message, "reasoning_content", "") or ""
            if not reasoning_content and hasattr(message, "model_extra"):
                model_extra = message.model_extra
                if isinstance(model_extra, dict):
                    reasoning_content = model_extra.get("reasoning_content", "") or ""

            # Update execution stats
            self.execution_stats.input_token_count = 0
            self.execution_stats.output_token_count = 0
            if response.usage:
                self.execution_stats.input_token_count = response.usage.prompt_tokens
                self.execution_stats.output_token_count = (
                    response.usage.completion_tokens
                )

            return {
                "response": response_content,
                "reasoning_content": str(reasoning_content),
            }

        except Exception as e:
            logger.error(f"Error calling DeepSeek API: {e}")
            raise

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        logger.debug(f"Running DeepSeek block with model: {input_data.model}")

        try:
            result = await self.call_deepseek(
                credentials=credentials,
                model=input_data.model,
                prompt=input_data.prompt,
                system_prompt=input_data.system_prompt,
                temperature=input_data.temperature,
                max_tokens=input_data.max_tokens,
                json_mode=input_data.json_mode,
                stream=input_data.stream,
            )

            yield "response", result["response"]
            yield "reasoning_content", result.get("reasoning_content", "")

        except Exception as e:
            error_msg = f"Error calling DeepSeek: {str(e)}"
            logger.error(error_msg)
            raise BlockExecutionError(error_msg) from e
