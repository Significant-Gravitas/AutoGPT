from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Literal

from openai import AsyncOpenAI
from openai.types.responses import Response as OpenAIResponse
from pydantic import SecretStr

from backend.blocks._base import (
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
    OAuth2Credentials,
    SchemaField,
)
from backend.integrations.codex.models import (
    CodexInvocationRequest,
    CodexInvocationResult,
)
from backend.integrations.codex.models import (
    CodexReasoningEffort as CodexRuntimeReasoningEffort,
)
from backend.integrations.codex.transport import get_codex_transport
from backend.integrations.credential_lease import CredentialLease
from backend.integrations.providers import ProviderName


@dataclass
class CodexCallResult:
    """Structured response returned by Codex invocations."""

    response: str
    reasoning: str
    response_id: str


class CodexModel(str, Enum):
    """Codex-capable OpenAI models."""

    GPT5_6_SOL = "gpt-5.6-sol"
    GPT5_6_TERRA = "gpt-5.6-terra"
    GPT5_6_LUNA = "gpt-5.6-luna"
    GPT5_3_CODEX = "gpt-5.3-codex"
    GPT5_1_CODEX = "gpt-5.1-codex"


class CodexReasoningEffort(str, Enum):
    """Reasoning effort supported by either Codex execution transport."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"
    ULTRA = "ultra"


class CodexExecutionTransport(str, Enum):
    OPENAI_API = "openai_api"
    CODEX_APP_SERVER = "codex_app_server"


_APP_SERVER_EFFORT: dict[
    CodexReasoningEffort,
    CodexRuntimeReasoningEffort | None,
] = {
    CodexReasoningEffort.NONE: None,
    CodexReasoningEffort.LOW: "low",
    CodexReasoningEffort.MEDIUM: "medium",
    CodexReasoningEffort.HIGH: "high",
    CodexReasoningEffort.XHIGH: "xhigh",
    CodexReasoningEffort.MAX: "max",
    CodexReasoningEffort.ULTRA: "ultra",
}


def _app_server_effort(
    effort: CodexReasoningEffort,
) -> CodexRuntimeReasoningEffort | None:
    return _APP_SERVER_EFFORT[effort]


CodexCredentials = CredentialsMetaInput[
    Literal[ProviderName.OPENAI, ProviderName.CODEX],
    Literal["api_key", "oauth2"],
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
        description="OpenAI API key or connected ChatGPT plan for Codex.",
        discriminator="transport",
        discriminator_mapping={
            CodexExecutionTransport.OPENAI_API.value: ProviderName.OPENAI,
            CodexExecutionTransport.CODEX_APP_SERVER.value: ProviderName.CODEX,
        },
        discriminator_type_mapping={
            CodexExecutionTransport.OPENAI_API.value: ["api_key"],
            CodexExecutionTransport.CODEX_APP_SERVER.value: ["oauth2"],
        },
    )


class CodeGenerationBlock(Block):
    """Block that talks to Codex through an API key or ChatGPT connection."""

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
            description=(
                "Optional instructions passed to the selected Codex transport."
            ),
            advanced=True,
        )
        transport: CodexExecutionTransport = SchemaField(
            title="Transport",
            default=CodexExecutionTransport.OPENAI_API,
            description=(
                "Use an OpenAI API key or your connected ChatGPT plan through "
                "Codex App Server."
            ),
            advanced=False,
        )
        model: CodexModel = SchemaField(
            title="Codex Model",
            default=CodexModel.GPT5_3_CODEX,
            description=(
                "OpenAI API transport only. Codex App Server selects the current "
                "subscription model from its live model catalog."
            ),
            advanced=False,
        )
        reasoning_effort: CodexReasoningEffort = SchemaField(
            title="Reasoning Effort",
            default=CodexReasoningEffort.MEDIUM,
            description=(
                "Controls the selected transport's reasoning effort. OpenAI API "
                "does not support 'ultra'; select 'none' to omit reasoning config."
            ),
            advanced=True,
        )
        max_output_tokens: int | None = SchemaField(
            title="Max Output Tokens",
            default=2048,
            description=(
                "OpenAI API transport only: upper bound for generated tokens "
                "(hard limit 128,000). Codex App Server uses its model and plan limits."
            ),
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
            description="Transport response ID for auditing and debugging.",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="86a2a099-30df-47b4-b7e4-34ae5f83e0d5",
            description=(
                "Generate or refactor code using an OpenAI API key or a connected "
                "ChatGPT plan through Codex App Server."
            ),
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

    _MODEL_USD_PER_1M: ClassVar[dict[CodexModel, tuple[float, float]]] = {
        CodexModel.GPT5_6_SOL: (5.0, 30.0),
        CodexModel.GPT5_6_TERRA: (2.5, 15.0),
        CodexModel.GPT5_6_LUNA: (1.0, 6.0),
        CodexModel.GPT5_3_CODEX: (1.75, 14.0),
        CodexModel.GPT5_1_CODEX: (1.25, 10.0),
    }

    @classmethod
    def _compute_token_usd(
        cls,
        model: CodexModel,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        input_rate, output_rate = cls._MODEL_USD_PER_1M[model]
        return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000

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
        input_tokens = response.usage.input_tokens if response.usage else 0
        output_tokens = response.usage.output_tokens if response.usage else 0
        self.execution_stats.input_token_count = input_tokens
        self.execution_stats.output_token_count = output_tokens
        self.execution_stats.llm_call_count += 1
        self.execution_stats.provider_cost = self._compute_token_usd(
            model, input_tokens, output_tokens
        )
        self.execution_stats.provider_cost_type = "cost_usd"

        return CodexCallResult(
            response=text_output,
            reasoning=reasoning_summary,
            response_id=response_id,
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials | OAuth2Credentials,
        credential_leases: dict[str, CredentialLease] | None = None,
        **_kwargs,
    ) -> BlockOutput:
        if input_data.transport == CodexExecutionTransport.OPENAI_API:
            result = await self._run_openai_api(input_data, credentials)
        else:
            result = await self._run_codex_app_server(
                input_data,
                credentials,
                credential_leases or {},
            )

        yield "response", result.response
        yield "reasoning", result.reasoning
        yield "response_id", result.response_id

    async def _run_openai_api(
        self,
        input_data: Input,
        credentials: APIKeyCredentials | OAuth2Credentials,
    ) -> CodexCallResult:
        if credentials.type != "api_key" or credentials.provider != "openai":
            raise ValueError("OpenAI API transport requires an OpenAI API key")
        if input_data.reasoning_effort == CodexReasoningEffort.ULTRA:
            raise ValueError(
                "OpenAI API transport does not support 'ultra' reasoning effort"
            )
        return await self.call_codex(
            credentials=credentials,
            model=input_data.model,
            prompt=input_data.prompt,
            system_prompt=input_data.system_prompt,
            max_output_tokens=input_data.max_output_tokens,
            reasoning_effort=input_data.reasoning_effort,
        )

    async def _run_codex_app_server(
        self,
        input_data: Input,
        credentials: APIKeyCredentials | OAuth2Credentials,
        credential_leases: dict[str, CredentialLease],
    ) -> CodexCallResult:
        if credentials.type != "oauth2" or credentials.provider != "codex":
            raise ValueError(
                "Codex App Server transport requires connected ChatGPT credentials"
            )
        lease = credential_leases.get("credentials")
        if lease is None or lease.credentials.id != credentials.id:
            raise ValueError("Codex App Server transport requires a credential lease")
        response = await get_codex_transport().invoke(
            lease=lease,
            request=CodexInvocationRequest(
                prompt=input_data.prompt,
                instructions=input_data.system_prompt or None,
                model=None,
                effort=_app_server_effort(input_data.reasoning_effort),
            ),
        )
        self._record_subscription_usage(response)
        return CodexCallResult(
            response=response.final_response,
            reasoning=response.reasoning_summary or "",
            response_id=response.response_id,
        )

    def _record_subscription_usage(
        self,
        response: CodexInvocationResult,
    ) -> None:
        self.execution_stats.llm_call_count += 1
        self.execution_stats.billing_mode = "user_subscription"
        self.execution_stats.auth_provider = "codex"
        self.execution_stats.execution_path = "codex_app_server"
        self.execution_stats.resolved_model = response.resolved_model
        usage = response.usage
        if usage is None:
            return

        self.execution_stats.input_token_count = usage.input_tokens
        self.execution_stats.cache_read_token_count = usage.cached_input_tokens
        self.execution_stats.output_token_count = usage.output_tokens
        self.execution_stats.provider_cost = usage.total_tokens
        self.execution_stats.provider_cost_type = "tokens"
