import logging
from enum import Enum
from typing import Any, Literal

from openai import AsyncOpenAI
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
from backend.util.logging import TruncatedLogger

logger = TruncatedLogger(logging.getLogger(__name__), "[Codex-Block]")


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


class CodexBlock(Block):
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
            description="Upper bound for generated tokens. Leave blank to let OpenAI decide.",
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
            input_schema=CodexBlock.Input,
            output_schema=CodexBlock.Output,
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
                "call_codex": lambda *args, **kwargs: {
                    "response": "function dedupe<T>(items: T[]): T[] { return [...new Set(items)]; }",
                    "reasoning": "Used Set to remove duplicates in O(n).",
                    "response_id": "resp_test",
                }
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
    ) -> dict[str, Any]:
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

        text_output = self._extract_output_text(response)
        reasoning_summary = self._extract_reasoning_summary(response)
        response_id = getattr(response, "id", "")

        usage = getattr(response, "usage", None)
        if usage:
            input_tokens = getattr(usage, "input_tokens", None)
            output_tokens = getattr(usage, "output_tokens", None)
            self.execution_stats.input_token_count = (
                input_tokens if isinstance(input_tokens, int) else (usage.get("input_tokens") if isinstance(usage, dict) else 0)
            )
            self.execution_stats.output_token_count = (
                output_tokens if isinstance(output_tokens, int) else (usage.get("output_tokens") if isinstance(usage, dict) else 0)
            )
        self.execution_stats.llm_call_count += 1

        return {
            "response": text_output,
            "reasoning": reasoning_summary,
            "response_id": response_id,
        }

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            result = await self.call_codex(
                credentials=credentials,
                model=input_data.model,
                prompt=input_data.prompt,
                system_prompt=input_data.system_prompt,
                max_output_tokens=input_data.max_output_tokens,
                reasoning_effort=input_data.reasoning_effort,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            error_msg = f"Codex request failed: {exc}"
            logger.error(error_msg)
            yield "error", error_msg
            return

        yield "response", result["response"]
        yield "reasoning", result.get("reasoning", "")
        yield "response_id", result.get("response_id", "")

    @staticmethod
    def _extract_output_text(response: Any) -> str:
        """Best-effort extraction of the aggregated text output."""
        if not response:
            return ""

        output_text = getattr(response, "output_text", None)
        if output_text:
            if isinstance(output_text, list):
                return "".join(str(part) for part in output_text)
            return str(output_text)

        aggregated_parts: list[str] = []
        output_items = getattr(response, "output", None) or []
        for item in output_items:
            content_list = getattr(item, "content", None) or []
            for content in content_list:
                text_value = getattr(content, "text", None)
                if text_value:
                    aggregated_parts.append(text_value)
                elif isinstance(content, dict) and content.get("text"):
                    aggregated_parts.append(str(content["text"]))

        if aggregated_parts:
            return "\n".join(aggregated_parts)

        raw_response = getattr(response, "_raw_response", None)
        if isinstance(raw_response, dict):
            try:
                first_text = raw_response["output"][0]["content"][0]["text"]
                return str(first_text)
            except (KeyError, IndexError, TypeError):
                return ""

        return ""

    @staticmethod
    def _extract_reasoning_summary(response: Any) -> str:
        reasoning_block = getattr(response, "reasoning", None)
        if not reasoning_block:
            return ""

        if isinstance(reasoning_block, dict):
            return str(reasoning_block.get("summary") or "")

        summary = getattr(reasoning_block, "summary", None)
        if isinstance(summary, list):
            return "\n".join(str(part) for part in summary)

        return str(summary or "")

