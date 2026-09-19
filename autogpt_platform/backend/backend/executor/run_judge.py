"""TypeSafe Jev run judge: typed post-run verdicts for a graph execution.

The existing activity-status generator asks a small chat model for a prose
summary *and* a free-form ``correctness_score``. This module replaces the
numeric half with typed judgments from Jev (TypeSafe's System One model): one
stateless request carrying the full execution evidence as ``state`` and six
named questions. Jev returns, per question, a choice (or score level) with a
probability over every option and a confidence (top-two margin). No free text
is generated here; the prose still comes from the chat model, optionally
conditioned on these verdicts (``RUN_JUDGE_MODE=primary``).

Transparency contract: every ``JudgeResult`` carries the verbatim request and
response bodies, the TypeSafe request ID, latency and token usage, so the exact
evidence and rubric that produced a verdict can be inspected later.

The deterministic terminal failures the current judge already short-circuits
(insufficient balance, entitlement denied) never reach Jev; they produce a
fixed ``JudgeResult`` with ``source="deterministic"`` instead.
"""

import asyncio
import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

from backend.blocks.typesafe._client import JevCallResult, call_jev
from backend.data.execution import ExecutionStatus
from backend.data.model import GraphExecutionStats
from backend.executor.run_judge_questions import (
    DELIVERED,
    DELIVERED_OPTIONS,
    ERRORS_VS_OUTCOME,
    ERRORS_VS_OUTCOME_OPTIONS,
    EXTERNAL_SIDE_EFFECTS,
    EXTERNAL_SIDE_EFFECTS_OPTIONS,
    FAILURE_CAUSE,
    FAILURE_CAUSE_OPTIONS,
    OUTPUT_QUALITY,
    OUTPUT_QUALITY_LEVELS,
    QUESTION_KEYS,
    USER_ACTION_NEEDED,
    USER_ACTION_NEEDED_OPTIONS,
    build_questions,
)
from backend.util.exceptions import ExecutionFailureReason
from backend.util.settings import Settings

logger = logging.getLogger(__name__)

JUDGE_VERSION = "run_judge/1"


class JudgeResult(BaseModel):
    """Persisted under ``GraphExecutionStats.judge`` (as a plain dict)."""

    version: str = JUDGE_VERSION
    source: Literal["jev", "deterministic"]
    mode: Literal["shadow", "primary"]
    answers: dict[str, dict[str, Any]] = Field(default_factory=dict)
    derived_correctness_score: float | None = None
    deterministic_reason: str | None = None
    request_id: str = ""
    latency_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    request: str | None = None
    response: str | None = None
    truncated: bool = False
    truncation_note: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and self.derived_correctness_score is not None

    def choice(self, key: str) -> str | None:
        answer = self.answers.get(key) or {}
        value = answer.get("choice")
        return str(value) if value is not None else None

    def to_stats(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def derive_correctness_score(answers: dict[str, dict[str, Any]]) -> float | None:
    """``P(delivered) + 0.5 * P(partially_delivered)``, clamped to [0, 1].

    Returns None when the ``delivered`` answer or its probabilities are absent,
    so a malformed response never silently becomes a score of 0.
    """
    delivered = answers.get(DELIVERED) or {}
    probabilities = delivered.get("probabilities")
    if not isinstance(probabilities, dict):
        return None
    try:
        full = float(probabilities.get("delivered", 0.0))
        partial = float(probabilities.get("partially_delivered", 0.0))
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, full + 0.5 * partial))


def _certain_choice(choice: str, options: dict[str, str]) -> dict[str, Any]:
    return {
        "type": "choice",
        "choice": choice,
        "probabilities": {name: (1.0 if name == choice else 0.0) for name in options},
        "confidence": 1.0,
    }


def deterministic_judgment(
    execution_stats: GraphExecutionStats,
    execution_status: ExecutionStatus | None,
    mode: Literal["shadow", "primary"],
) -> JudgeResult | None:
    """Fixed verdicts for failures the platform already classified.

    Mirrors ``activity_status_generator._get_deterministic_failure_response``:
    only FAILED runs with a structured ``failure_reason`` qualify. These never
    call Jev.
    """
    if execution_status != ExecutionStatus.FAILED:
        return None
    reason = execution_stats.failure_reason
    if reason == ExecutionFailureReason.INSUFFICIENT_BALANCE:
        action = "add_credits"
    elif reason == ExecutionFailureReason.ENTITLEMENT_REQUIRED:
        action = "add_credits"
    else:
        return None
    levels = {str(i): level for i, level in enumerate(OUTPUT_QUALITY_LEVELS)}
    answers = {
        DELIVERED: _certain_choice("not_delivered", DELIVERED_OPTIONS),
        ERRORS_VS_OUTCOME: _certain_choice(
            "errors_caused_failure", ERRORS_VS_OUTCOME_OPTIONS
        ),
        FAILURE_CAUSE: _certain_choice("not_applicable", FAILURE_CAUSE_OPTIONS),
        USER_ACTION_NEEDED: _certain_choice(action, USER_ACTION_NEEDED_OPTIONS),
        EXTERNAL_SIDE_EFFECTS: _certain_choice("none", EXTERNAL_SIDE_EFFECTS_OPTIONS),
        OUTPUT_QUALITY: {
            "type": "score",
            "score": 0.0,
            "legend": levels,
            "probabilities": {key: (1.0 if key == "0" else 0.0) for key in levels},
            "confidence": 1.0,
        },
    }
    return JudgeResult(
        source="deterministic",
        mode=mode,
        answers=answers,
        derived_correctness_score=0.0,
        deterministic_reason=reason.value,
    )


def _from_call(
    result: JevCallResult, mode: Literal["shadow", "primary"]
) -> JudgeResult:
    error = result.error
    missing = [key for key in QUESTION_KEYS if key not in result.answers]
    if not error and missing:
        error = f"Jev response is missing answers for: {', '.join(missing)}."
    derived = derive_correctness_score(result.answers) if not error else None
    if not error and derived is None:
        error = "Jev 'delivered' answer has no usable probabilities."
    return JudgeResult(
        source="jev",
        mode=mode,
        answers=result.answers,
        derived_correctness_score=derived,
        request_id=result.request_id,
        latency_ms=result.latency_ms,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        request=result.request,
        response=result.response,
        truncated=result.truncated,
        truncation_note=result.truncation_note,
        error=error,
    )


async def judge_execution(
    execution_data: dict[str, Any],
    execution_stats: GraphExecutionStats,
    execution_status: ExecutionStatus | None,
    *,
    api_key: str,
    mode: Literal["shadow", "primary"] = "shadow",
    timeout_seconds: float = 10.0,
) -> JudgeResult:
    """Ask Jev the six run-judge questions about one execution.

    ``execution_data`` is the summary from
    ``activity_status_generator._build_execution_summary`` and is sent as the
    Jev ``state`` verbatim (compact JSON, truncated with a note if it exceeds
    the client's byte budget). Never raises for API/transport failures: the
    returned ``JudgeResult.error`` explains them. Raises ``ValueError`` when
    ``api_key`` is empty.
    """
    deterministic = deterministic_judgment(execution_stats, execution_status, mode)
    if deterministic is not None:
        return deterministic
    if not api_key:
        raise ValueError(
            "TYPESAFE_API_KEY is not configured; run judge cannot call Jev."
        )
    questions = build_questions()
    try:
        call = await asyncio.wait_for(
            call_jev(api_key, execution_data, questions),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        return JudgeResult(
            source="jev",
            mode=mode,
            error=(
                f"Jev call exceeded the run judge timeout of {timeout_seconds:g}s; "
                "no verdicts were recorded."
            ),
        )
    return _from_call(call, mode)


async def judge_execution_safely(
    execution_data: dict[str, Any],
    execution_stats: GraphExecutionStats,
    execution_status: ExecutionStatus | None,
    *,
    graph_exec_id: str,
    settings: Settings | None = None,
) -> JudgeResult | None:
    """Settings-driven wrapper that never raises and returns None when off.

    Used by the activity-status generator. In ``shadow`` and ``primary`` modes
    a Jev failure is logged and surfaced as ``JudgeResult.error``; any
    unexpected exception is logged and swallowed so the judge can never fail
    the run or the existing summary.
    """
    settings = settings or Settings()
    mode = settings.config.run_judge_mode
    if mode == "off":
        return None
    deterministic = deterministic_judgment(execution_stats, execution_status, mode)
    if deterministic is not None:
        return deterministic
    api_key = settings.secrets.typesafe_api_key
    if not api_key:
        logger.debug(
            "run_judge: RUN_JUDGE_MODE=%s but TYPESAFE_API_KEY is empty; skipping %s",
            mode,
            graph_exec_id,
        )
        return None
    try:
        result = await judge_execution(
            execution_data,
            execution_stats,
            execution_status,
            api_key=api_key,
            mode=mode,
            timeout_seconds=settings.config.run_judge_timeout_seconds,
        )
    except Exception:
        logger.exception("run_judge: unexpected failure judging %s", graph_exec_id)
        return None
    if result.error:
        logger.warning(
            "run_judge: %s verdicts unavailable for %s: %s",
            mode,
            graph_exec_id,
            result.error,
        )
    else:
        logger.info(
            "run_judge: %s %s delivered=%s derived_correctness=%.2f "
            "latency_ms=%.0f request_id=%s",
            mode,
            graph_exec_id,
            result.choice(DELIVERED),
            result.derived_correctness_score or 0.0,
            result.latency_ms or 0.0,
            result.request_id,
        )
    return result


def verdicts_block(result: JudgeResult) -> str:
    """Short ``Verdicts:`` block for the summary prompt in primary mode."""
    lines = ["Verdicts (typed judgments with probabilities from a separate judge):"]
    for key in QUESTION_KEYS:
        answer = result.answers.get(key)
        if not answer:
            continue
        probabilities = answer.get("probabilities") or {}
        if answer.get("type") == "score" or "score" in answer:
            legend = answer.get("legend") or {}
            level = str(int(round(float(answer.get("score", 0)))))
            label = str(legend.get(level, level)).split(":", 1)[0]
            lines.append(
                f"- {key}: {label} (level {level} of {len(legend) - 1 if legend else '?'}, "
                f"confidence {float(answer.get('confidence', 0)):.2f})"
            )
            continue
        top = sorted(
            ((str(name), float(p)) for name, p in probabilities.items()),
            key=lambda item: item[1],
            reverse=True,
        )[:3]
        spread = ", ".join(f"{name} {p:.2f}" for name, p in top)
        lines.append(f"- {key}: {answer.get('choice')} ({spread})")
    if result.derived_correctness_score is not None:
        lines.append(
            f"- derived correctness score: {result.derived_correctness_score:.2f}"
        )
    return "\n".join(lines)
