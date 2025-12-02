from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from openai import AsyncOpenAI
from openai.types.responses import Response as OpenAIResponse
from pydantic import SecretStr

from backend.data.block import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import (
    APIKeyCredentials,
    CredentialsField,
    CredentialsMetaInput,
    NodeExecutionStats,
    SchemaField,
)
from backend.integrations.providers import ProviderName


@dataclass
class CodexCallResult:
    """Structured response returned by Codex invocations."""

    response: str
    reasoning: str
    response_id: str


class CodexModel(str, Enum):
    """Codex-capable OpenAI models."""

    GPT5_1_CODEX = "gpt-5.1-codex"


class CodexReasoningEffort(str, Enum):
    """Configuration for the Responses API reasoning effort."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


CodexCredentials = CredentialsMetaInput[
    Literal[ProviderName.OPENAI], Literal["api_key"]
]

TEST_CREDENTIALS = APIKeyCredentials(
    id="e2fcb203-3f2d-4ad4-a344-8df3bc7db36b",
    provider="openai",
    api_key=SecretStr("mock-openai-api-key"),
    title="Mock OpenAI API key",
    expires_at=None,
)
TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}


def CodexCredentialsField() -> CodexCredentials:
    return CredentialsField(
        description="OpenAI API key with access to Codex models (Responses API).",
    )


class CodeGenerationBlock(Block):
    """Block that talks to Codex models via the OpenAI Responses API."""

    class Input(BlockSchemaInput):
        prompt: str = SchemaField(
            description="Primary coding request passed to the Codex model.",
            placeholder="Generate a Python function that reverses a list.",
        )
        system_prompt: str = SchemaField(
            title="System Prompt",
            default=(
                "You are Codex, an elite software engineer. "
                "Favor concise, working code and highlight important caveats."
            ),
            description="Optional instructions injected via the Responses API instructions field.",
            advanced=True,
        )
        model: CodexModel = SchemaField(
            title="Codex Model",
            default=CodexModel.GPT5_1_CODEX,
            description="Codex-optimized model served via the Responses API.",
            advanced=False,
        )
        reasoning_effort: CodexReasoningEffort = SchemaField(
            title="Reasoning Effort",
            default=CodexReasoningEffort.MEDIUM,
            description="Controls the Responses API reasoning budget. Select 'none' to skip reasoning configs.",
            advanced=True,
        )
        max_output_tokens: int | None = SchemaField(
            title="Max Output Tokens",
            default=2048,
            description="Upper bound for generated tokens (hard limit 128,000). Leave blank to let OpenAI decide.",
            advanced=True,
        )
        credentials: CodexCredentials = CodexCredentialsField()

    class Output(BlockSchemaOutput):
        response: str = SchemaField(
            description="Code-focused response returned by the Codex model."
        )
        reasoning: str = SchemaField(
            description="Reasoning summary returned by the model, if available.",
            default="",
        )
        response_id: str = SchemaField(
            description="ID of the Responses API call for auditing/debugging.",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="86a2a099-30df-47b4-b7e4-34ae5f83e0d5",
            description="Generate or refactor code using OpenAI's Codex (Responses API).",
            categories={BlockCategory.AI, BlockCategory.DEVELOPER_TOOLS},
            input_schema=CodeGenerationBlock.Input,
            output_schema=CodeGenerationBlock.Output,
            test_input=[
                {
                    "prompt": "Write a TypeScript function that deduplicates an array.",
                    "credentials": TEST_CREDENTIALS_INPUT,
                }
            ],
            test_output=[
                ("response", str),
                ("reasoning", str),
                ("response_id", str),
            ],
            test_mock={
                "call_codex": lambda *_args, **_kwargs: CodexCallResult(
                    response="function dedupe<T>(items: T[]): T[] { return [...new Set(items)]; }",
                    reasoning="Used Set to remove duplicates in O(n).",
                    response_id="resp_test",
                )
            },
            test_credentials=TEST_CREDENTIALS,
        )
        self.execution_stats = NodeExecutionStats()

    async def call_codex(
        self,
        *,
        credentials: APIKeyCredentials,
        model: CodexModel,
        prompt: str,
        system_prompt: str,
        max_output_tokens: int | None,
        reasoning_effort: CodexReasoningEffort,
    ) -> CodexCallResult:
        """Invoke the OpenAI Responses API."""
        client = AsyncOpenAI(api_key=credentials.api_key.get_secret_value())

        request_payload: dict[str, Any] = {
            "model": model.value,
            "input": prompt,
        }
        if system_prompt:
            request_payload["instructions"] = system_prompt
        if max_output_tokens is not None:
            request_payload["max_output_tokens"] = max_output_tokens
        if reasoning_effort != CodexReasoningEffort.NONE:
            request_payload["reasoning"] = {"effort": reasoning_effort.value}

        response = await client.responses.create(**request_payload)
        if not isinstance(response, OpenAIResponse):
            raise TypeError(f"Expected OpenAIResponse, got {type(response).__name__}")

        # Extract data directly from typed response
        text_output = response.output_text or ""
        reasoning_summary = (
            str(response.reasoning.summary)
            if response.reasoning and response.reasoning.summary
            else ""
        )
        response_id = response.id or ""

        # Update usage stats
        self.execution_stats.input_token_count = (
            response.usage.input_tokens if response.usage else 0
        )
        self.execution_stats.output_token_count = (
            response.usage.output_tokens if response.usage else 0
        )
        self.execution_stats.llm_call_count += 1

        return CodexCallResult(
            response=text_output,
            reasoning=reasoning_summary,
            response_id=response_id,
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        **_kwargs,
    ) -> BlockOutput:
        result = await self.call_codex(
            credentials=credentials,
            model=input_data.model,
            prompt=input_data.prompt,
            system_prompt=input_data.system_prompt,
            max_output_tokens=input_data.max_output_tokens,
            reasoning_effort=input_data.reasoning_effort,
        )

        yield "response", result.response
        yield "reasoning", result.reasoning
        yield "response_id", result.response_id
