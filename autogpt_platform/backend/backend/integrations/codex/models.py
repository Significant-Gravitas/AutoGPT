from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from backend.integrations.codex.auth_bundle import CodexAuthBundleV1

CodexReasoningEffort = Literal[
    "none",
    "low",
    "medium",
    "high",
    "xhigh",
]


class CodexDeviceCodeDetails(BaseModel):
    model_config = ConfigDict(extra="forbid")

    login_id: str
    verification_url: str
    user_code: str


class CodexAccountSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    connected: bool
    requires_openai_auth: bool
    account_type: str | None = None
    email: str | None = None
    plan_type: str | None = None


class CodexRateLimitWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    used_percent: int
    window_duration_mins: int | None = None
    resets_at: int | None = None


class CodexRateLimitsSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_type: str | None = None
    limit_id: str | None = None
    limit_name: str | None = None
    rate_limit_reached_type: str | None = None
    primary: CodexRateLimitWindow | None = None
    secondary: CodexRateLimitWindow | None = None
    has_credits: bool | None = None
    unlimited_credits: bool | None = None
    bucket_ids: list[str] = Field(default_factory=list)


class CodexTokenUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_output_tokens: int
    total_tokens: int


class CodexInvocationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str
    instructions: str | None = None
    model: str | None = None
    effort: CodexReasoningEffort | None = None
    output_schema: dict[str, object] | None = None
    timeout_seconds: float | None = Field(default=None, gt=0)


class CodexInvocationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    response_id: str
    final_response: str
    reasoning_summary: str | None = None
    status: str
    duration_ms: int | None = None
    usage: CodexTokenUsage | None = None


class CodexDynamicToolSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,128}$")
    description: str
    input_schema: dict[str, object]


class CodexDynamicToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str
    turn_id: str
    call_id: str
    namespace: str | None = None
    tool: str
    arguments: object


class CodexDynamicToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str
    success: bool = True


class CodexLoginCompletion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bundle: CodexAuthBundleV1
    account: CodexAccountSnapshot | None = None
    rate_limits: CodexRateLimitsSnapshot | None = None
