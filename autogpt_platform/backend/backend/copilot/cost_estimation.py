"""CoPilot pre-execution cost estimation."""

from typing import Literal

from pydantic import BaseModel, Field

from backend.copilot.baseline.service import _resolve_baseline_model
from backend.copilot.config import ChatConfig, CopilotLlmModel, CopilotMode
from backend.copilot.executor.processor import (
    resolve_effective_mode,
    resolve_use_sdk_for_mode,
)
from backend.copilot.model import ChatSession
from backend.copilot.sdk.service import _resolve_model_and_multiplier
from backend.util.feature_flag import Flag, get_feature_flag_value
from backend.util.prompt import estimate_token_count, estimate_token_count_str

config = ChatConfig()


class CoPilotTurnCostEstimate(BaseModel):
    """Estimated token usage and USD cost for a CoPilot turn."""

    resolved_path: Literal["baseline", "sdk"]
    resolved_model: str
    estimated_llm_calls: int = Field(ge=1)
    estimated_prompt_tokens: int = Field(ge=0)
    estimated_completion_tokens: int = Field(ge=0)
    estimated_total_tokens: int = Field(ge=0)
    estimated_cost_usd: float = Field(ge=0)
    confirmation_threshold_usd: float = Field(ge=0)
    threshold_source: Literal["launchdarkly", "default"]
    requires_confirmation: bool
    rationale: str
    approval_token: str | None = None


async def estimate_copilot_turn_cost(
    *,
    session_id: str,
    user_id: str | None,
    session: ChatSession,
    message: str,
    is_user_message: bool,
    context: dict[str, str] | None,
    file_ids: list[str] | None,
    mode: CopilotMode | None,
    model: CopilotLlmModel | None,
) -> CoPilotTurnCostEstimate:
    """Estimate cost before executing a CoPilot turn."""
    resolved_path, resolved_model = await _resolve_path_and_model(
        session_id=session_id,
        user_id=user_id,
        mode=mode,
        model=model,
    )
    message_tokens = max(1, estimate_token_count_str(message))
    history_tokens = _estimate_history_tokens(session)
    file_count = len(file_ids or [])
    llm_calls = _estimate_llm_calls(
        resolved_path=resolved_path,
        message_tokens=message_tokens,
        history_tokens=history_tokens,
        file_count=file_count,
        has_context=context is not None,
        is_user_message=is_user_message,
    )
    prompt_tokens, completion_tokens = _estimate_token_usage(
        resolved_path=resolved_path,
        message_tokens=message_tokens,
        history_tokens=history_tokens,
        llm_calls=llm_calls,
        file_count=file_count,
        has_context=context is not None,
        resolved_model=resolved_model,
    )
    estimated_cost = _estimate_cost_usd(
        model_name=resolved_model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    threshold, source = await resolve_cost_confirmation_threshold(user_id)
    requires_confirmation = threshold > 0 and estimated_cost >= threshold
    rationale = _build_rationale(
        resolved_path=resolved_path,
        llm_calls=llm_calls,
        message_tokens=message_tokens,
        history_tokens=history_tokens,
        file_count=file_count,
        has_context=context is not None,
    )
    return CoPilotTurnCostEstimate(
        resolved_path=resolved_path,
        resolved_model=resolved_model,
        estimated_llm_calls=llm_calls,
        estimated_prompt_tokens=prompt_tokens,
        estimated_completion_tokens=completion_tokens,
        estimated_total_tokens=prompt_tokens + completion_tokens,
        estimated_cost_usd=round(estimated_cost, 6),
        confirmation_threshold_usd=threshold,
        threshold_source=source,
        requires_confirmation=requires_confirmation,
        rationale=rationale,
    )


async def resolve_cost_confirmation_threshold(
    user_id: str | None,
) -> tuple[float, Literal["launchdarkly", "default"]]:
    """Resolve threshold from LaunchDarkly with config fallback."""
    fallback = max(0.0, float(config.copilot_cost_confirmation_threshold_usd))
    if not user_id:
        return fallback, "default"
    raw_value = await get_feature_flag_value(
        Flag.COPILOT_COST_CONFIRMATION_THRESHOLD_USD.value,
        user_id,
        fallback,
    )
    try:
        threshold = max(0.0, float(raw_value))
    except (TypeError, ValueError):
        return fallback, "default"
    if threshold == fallback:
        return threshold, "default"
    return threshold, "launchdarkly"


async def _resolve_path_and_model(
    *,
    session_id: str,
    user_id: str | None,
    mode: CopilotMode | None,
    model: CopilotLlmModel | None,
) -> tuple[Literal["baseline", "sdk"], str]:
    if config.test_mode:
        return "baseline", config.fast_model
    effective_mode = await resolve_effective_mode(mode, user_id)
    use_sdk = await resolve_use_sdk_for_mode(
        effective_mode,
        user_id,
        use_claude_code_subscription=config.use_claude_code_subscription,
        config_default=config.use_claude_agent_sdk,
    )
    if use_sdk:
        sdk_model, _ = await _resolve_model_and_multiplier(model, session_id)
        return "sdk", sdk_model or config.model
    return "baseline", _resolve_baseline_model(effective_mode)


def _estimate_history_tokens(session: ChatSession) -> int:
    messages = [
        {"role": m.role, "content": m.content or ""}
        for m in session.messages[-30:]
        if m.role in {"system", "user", "assistant", "tool"}
    ]
    if not messages:
        return 0
    return estimate_token_count(messages)


def _estimate_llm_calls(
    *,
    resolved_path: Literal["baseline", "sdk"],
    message_tokens: int,
    history_tokens: int,
    file_count: int,
    has_context: bool,
    is_user_message: bool,
) -> int:
    calls = 2 if resolved_path == "sdk" else 1
    if has_context:
        calls += 1
    if file_count > 0:
        calls += 1 + min(2, file_count // 3)
    if message_tokens > 800:
        calls += 1
    if message_tokens > 2_000:
        calls += 1
    if history_tokens > 8_000:
        calls += 1
    if history_tokens > 20_000:
        calls += 1
    if not is_user_message:
        calls = max(1, calls - 1)
    return max(1, min(calls, 12))


def _estimate_token_usage(
    *,
    resolved_path: Literal["baseline", "sdk"],
    message_tokens: int,
    history_tokens: int,
    llm_calls: int,
    file_count: int,
    has_context: bool,
    resolved_model: str,
) -> tuple[int, int]:
    context_tokens = min(history_tokens, 15_000)
    per_call_prompt = max(400, message_tokens + context_tokens)
    if resolved_path == "baseline":
        per_call_prompt = int(per_call_prompt * 0.8)
    completion_per_call = 550 if resolved_path == "sdk" else 320
    if has_context:
        completion_per_call += 80
    if file_count > 0:
        completion_per_call += 100
    if "opus" in resolved_model.lower():
        completion_per_call = int(completion_per_call * 1.2)
    prompt_total = per_call_prompt * llm_calls
    completion_total = completion_per_call * llm_calls
    return prompt_total, completion_total


def _estimate_cost_usd(
    *,
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    input_rate, output_rate = _model_pricing_per_million(model_name)
    input_cost = (prompt_tokens / 1_000_000) * input_rate
    output_cost = (completion_tokens / 1_000_000) * output_rate
    return input_cost + output_cost


def _model_pricing_per_million(model_name: str) -> tuple[float, float]:
    lower_name = model_name.lower()
    if "opus" in lower_name:
        return 15.0, 75.0
    if "sonnet" in lower_name:
        return 3.0, 15.0
    return 3.0, 15.0


def _build_rationale(
    *,
    resolved_path: Literal["baseline", "sdk"],
    llm_calls: int,
    message_tokens: int,
    history_tokens: int,
    file_count: int,
    has_context: bool,
) -> str:
    reasons: list[str] = [f"{resolved_path} path estimated at {llm_calls} model calls"]
    if message_tokens > 800:
        reasons.append("large prompt")
    if history_tokens > 8_000:
        reasons.append("long chat history")
    if file_count > 0:
        reasons.append(f"{file_count} attached files")
    if has_context:
        reasons.append("extra context included")
    return ", ".join(reasons)
