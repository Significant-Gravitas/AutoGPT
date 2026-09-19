"""Tests for the TypeSafe Jev run judge."""

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from typesafe_sdk import Choice, Score
from typesafe_sdk._core.json import serialize

from backend.blocks.typesafe._client import JevCallResult
from backend.data.execution import ExecutionStatus
from backend.data.model import GraphExecutionStats
from backend.executor import run_judge
from backend.executor.run_judge import (
    JudgeResult,
    build_questions,
    derive_correctness_score,
    deterministic_judgment,
    judge_execution,
    judge_execution_safely,
    verdicts_block,
)
from backend.executor.run_judge_questions import (
    DELIVERED_OPTIONS,
    ERRORS_VS_OUTCOME_OPTIONS,
    EXTERNAL_SIDE_EFFECTS_OPTIONS,
    FAILURE_CAUSE_OPTIONS,
    OUTPUT_QUALITY_LEVELS,
    QUESTION_KEYS,
    USER_ACTION_NEEDED_OPTIONS,
)
from backend.util.exceptions import ExecutionFailureReason
from backend.util.settings import Settings

EXECUTION_DATA: dict[str, Any] = {
    "graph_info": {"name": "Summarizer", "description": "Summarize a URL"},
    "nodes": [
        {
            "node_id": "abc",
            "block_name": "AgentOutputBlock",
            "is_graph_output": True,
            "recent_outputs": [{"output_data": {"output": ["A summary."]}}],
        }
    ],
    "overall_status": {"graph_execution_status": "COMPLETED", "graph_error": None},
}


def _choice(choice: str, options: dict[str, str], p: float = 0.9) -> dict[str, Any]:
    rest = (1.0 - p) / (len(options) - 1)
    return {
        "type": "choice",
        "choice": choice,
        "probabilities": {k: (p if k == choice else rest) for k in options},
        "confidence": p - rest,
    }


def _answers() -> dict[str, dict[str, Any]]:
    return {
        "delivered": {
            "type": "choice",
            "choice": "delivered",
            "probabilities": {
                "delivered": 0.7,
                "partially_delivered": 0.2,
                "not_delivered": 0.05,
                "cannot_tell_from_evidence": 0.05,
            },
            "confidence": 0.5,
        },
        "errors_vs_outcome": _choice("no_errors", ERRORS_VS_OUTCOME_OPTIONS),
        "failure_cause": _choice("not_applicable", FAILURE_CAUSE_OPTIONS),
        "user_action_needed": _choice("none", USER_ACTION_NEEDED_OPTIONS),
        "external_side_effects": _choice("none", EXTERNAL_SIDE_EFFECTS_OPTIONS),
        "output_quality": {
            "type": "score",
            "score": 3.0,
            "legend": {str(i): level for i, level in enumerate(OUTPUT_QUALITY_LEVELS)},
            "probabilities": {"0": 0.0, "1": 0.05, "2": 0.15, "3": 0.6, "4": 0.2},
            "confidence": 0.4,
        },
    }


def _call_result(
    answers: dict[str, dict[str, Any]] | None = None, error: str = ""
) -> JevCallResult:
    return JevCallResult(
        answers=answers if answers is not None else {},
        request=json.dumps({"state": json.dumps(EXECUTION_DATA)}),
        response=None if error else json.dumps({"answers": answers}),
        latency_ms=123.4,
        input_tokens=None if error else 900,
        output_tokens=None if error else 60,
        request_id="req-judge-1",
        truncated=False,
        truncation_note="",
        error=error,
    )


def _settings(mode: str = "shadow", key: str = "test-key") -> Settings:
    # Environment (.env) outranks constructor kwargs for these settings
    # models, so set the fields after construction.
    settings = Settings()
    settings.config.run_judge_mode = mode
    settings.config.run_judge_timeout_seconds = 5
    settings.secrets.typesafe_api_key = key
    return settings


class TestQuestionSet:
    def test_question_keys_and_types(self):
        questions = build_questions()
        assert tuple(questions) == QUESTION_KEYS
        for key in QUESTION_KEYS[:-1]:
            assert isinstance(questions[key], Choice)
        assert isinstance(questions["output_quality"], Score)

    def test_choice_options_match_spec(self):
        assert list(DELIVERED_OPTIONS) == [
            "delivered",
            "partially_delivered",
            "not_delivered",
            "cannot_tell_from_evidence",
        ]
        assert list(ERRORS_VS_OUTCOME_OPTIONS) == [
            "no_errors",
            "errors_recovered_outcome_unaffected",
            "errors_degraded_outcome",
            "errors_caused_failure",
        ]
        assert list(FAILURE_CAUSE_OPTIONS) == [
            "not_applicable",
            "bad_user_input",
            "missing_credential_or_integration",
            "external_service_failure",
            "agent_design_or_wiring",
            "platform_bug",
        ]
        assert list(USER_ACTION_NEEDED_OPTIONS) == [
            "none",
            "fix_the_input",
            "connect_an_integration",
            "add_credits",
            "rebuild_the_agent",
        ]
        assert list(EXTERNAL_SIDE_EFFECTS_OPTIONS) == [
            "none",
            "intended_only",
            "unintended_or_repeated",
        ]
        assert [level.split(":", 1)[0] for level in OUTPUT_QUALITY_LEVELS] == [
            "unusable",
            "poor",
            "acceptable",
            "good",
            "excellent",
        ]

    def test_questions_reference_evidence_fields(self):
        questions = build_questions()
        wire = serialize(questions).decode("utf-8")
        for field in (
            "graph_info.description",
            "is_graph_output",
            "recent_errors",
            "graph_error",
            "input_output_data",
        ):
            assert field in wire
        assert "if the run delivered, answer not_applicable" in wire.lower()
        assert "cannot_tell_from_evidence" in wire


class TestDerivedScore:
    def test_formula(self):
        assert derive_correctness_score(_answers()) == pytest.approx(0.8)

    def test_missing_probabilities_is_none(self):
        assert derive_correctness_score({}) is None
        assert derive_correctness_score({"delivered": {"choice": "x"}}) is None

    def test_clamped(self):
        answers = {
            "delivered": {
                "probabilities": {"delivered": 1.0, "partially_delivered": 1.0}
            }
        }
        assert derive_correctness_score(answers) == 1.0


class TestDeterministicShortCircuit:
    @pytest.mark.parametrize(
        "reason,action",
        [
            (ExecutionFailureReason.INSUFFICIENT_BALANCE, "add_credits"),
            (ExecutionFailureReason.ENTITLEMENT_REQUIRED, "add_credits"),
        ],
    )
    def test_failed_with_reason(self, reason, action):
        stats = GraphExecutionStats(failure_reason=reason)
        result = deterministic_judgment(stats, ExecutionStatus.FAILED, "shadow")
        assert result is not None
        assert result.source == "deterministic"
        assert result.deterministic_reason == reason.value
        assert result.derived_correctness_score == 0.0
        assert result.choice("delivered") == "not_delivered"
        assert result.choice("user_action_needed") == action
        assert set(result.answers) == set(QUESTION_KEYS)
        assert result.ok

    def test_completed_or_unclassified_is_none(self):
        stats = GraphExecutionStats(
            failure_reason=ExecutionFailureReason.INSUFFICIENT_BALANCE
        )
        assert (
            deterministic_judgment(stats, ExecutionStatus.COMPLETED, "shadow") is None
        )
        assert (
            deterministic_judgment(
                GraphExecutionStats(), ExecutionStatus.FAILED, "shadow"
            )
            is None
        )

    async def test_short_circuit_never_calls_jev(self):
        stats = GraphExecutionStats(
            failure_reason=ExecutionFailureReason.ENTITLEMENT_REQUIRED
        )
        with patch.object(run_judge, "call_jev", AsyncMock()) as call:
            result = await judge_execution(
                {}, stats, ExecutionStatus.FAILED, api_key="", mode="primary"
            )
        assert result.source == "deterministic"
        call.assert_not_called()


class TestJudgeExecution:
    async def test_sends_evidence_as_state_and_maps_answers(self):
        call = AsyncMock(return_value=_call_result(_answers()))
        stats = GraphExecutionStats(node_count=1)
        with patch.object(run_judge, "call_jev", call):
            result = await judge_execution(
                EXECUTION_DATA,
                stats,
                ExecutionStatus.COMPLETED,
                api_key="k",
                mode="shadow",
            )
        api_key, state, questions = call.await_args.args
        assert api_key == "k"
        assert state is EXECUTION_DATA
        assert tuple(questions) == QUESTION_KEYS
        assert result.source == "jev"
        assert result.ok
        assert result.derived_correctness_score == pytest.approx(0.8)
        assert result.request_id == "req-judge-1"
        assert result.latency_ms == 123.4
        assert result.input_tokens == 900 and result.output_tokens == 60
        assert result.request and result.response
        record = result.to_stats()
        assert record["answers"]["delivered"]["probabilities"]["delivered"] == 0.7
        assert record["derived_correctness_score"] == pytest.approx(0.8)

    async def test_api_error_is_reported_not_raised(self):
        call = AsyncMock(
            return_value=_call_result(error="Jev API request failed (HTTP 500).")
        )
        with patch.object(run_judge, "call_jev", call):
            result = await judge_execution(
                EXECUTION_DATA,
                GraphExecutionStats(),
                ExecutionStatus.COMPLETED,
                api_key="k",
            )
        assert not result.ok
        assert "HTTP 500" in result.error
        assert result.derived_correctness_score is None
        assert result.request  # verbatim request is still kept

    async def test_missing_answer_is_an_error(self):
        answers = _answers()
        del answers["output_quality"]
        with patch.object(
            run_judge, "call_jev", AsyncMock(return_value=_call_result(answers))
        ):
            result = await judge_execution(
                EXECUTION_DATA,
                GraphExecutionStats(),
                ExecutionStatus.COMPLETED,
                api_key="k",
            )
        assert "output_quality" in result.error
        assert result.derived_correctness_score is None

    async def test_timeout_is_bounded(self):
        async def slow(*_args, **_kwargs):
            await asyncio.sleep(5)
            return _call_result(_answers())

        with patch.object(run_judge, "call_jev", slow):
            result = await judge_execution(
                EXECUTION_DATA,
                GraphExecutionStats(),
                ExecutionStatus.COMPLETED,
                api_key="k",
                timeout_seconds=0.05,
            )
        assert "timeout" in result.error
        assert result.derived_correctness_score is None

    async def test_empty_key_raises(self):
        with pytest.raises(ValueError):
            await judge_execution(
                EXECUTION_DATA,
                GraphExecutionStats(),
                ExecutionStatus.COMPLETED,
                api_key="",
            )


class TestJudgeExecutionSafely:
    async def test_off_mode_skips(self):
        with patch.object(run_judge, "call_jev", AsyncMock()) as call:
            result = await judge_execution_safely(
                EXECUTION_DATA,
                GraphExecutionStats(),
                ExecutionStatus.COMPLETED,
                graph_exec_id="e",
                settings=_settings("off"),
            )
        assert result is None
        call.assert_not_called()

    async def test_missing_key_skips(self):
        with patch.object(run_judge, "call_jev", AsyncMock()) as call:
            result = await judge_execution_safely(
                EXECUTION_DATA,
                GraphExecutionStats(),
                ExecutionStatus.COMPLETED,
                graph_exec_id="e",
                settings=_settings("shadow", key=""),
            )
        assert result is None
        call.assert_not_called()

    async def test_deterministic_without_key(self):
        stats = GraphExecutionStats(
            failure_reason=ExecutionFailureReason.INSUFFICIENT_BALANCE
        )
        result = await judge_execution_safely(
            {},
            stats,
            ExecutionStatus.FAILED,
            graph_exec_id="e",
            settings=_settings(key=""),
        )
        assert result is not None and result.source == "deterministic"

    async def test_unexpected_exception_is_swallowed(self):
        with patch.object(
            run_judge, "call_jev", AsyncMock(side_effect=RuntimeError("boom"))
        ):
            result = await judge_execution_safely(
                EXECUTION_DATA,
                GraphExecutionStats(),
                ExecutionStatus.COMPLETED,
                graph_exec_id="e",
                settings=_settings("shadow"),
            )
        assert result is None

    async def test_shadow_mode_records_verdicts(self):
        with patch.object(
            run_judge, "call_jev", AsyncMock(return_value=_call_result(_answers()))
        ):
            result = await judge_execution_safely(
                EXECUTION_DATA,
                GraphExecutionStats(),
                ExecutionStatus.COMPLETED,
                graph_exec_id="e",
                settings=_settings("shadow"),
            )
        assert result is not None and result.mode == "shadow" and result.ok


class TestVerdictsBlock:
    def test_lists_every_answer_and_derived_score(self):
        result = JudgeResult(
            source="jev",
            mode="primary",
            answers=_answers(),
            derived_correctness_score=0.8,
        )
        block = verdicts_block(result)
        assert block.startswith("Verdicts")
        for key in QUESTION_KEYS:
            assert f"- {key}:" in block
        assert "delivered: delivered (delivered 0.70, partially_delivered 0.20" in block
        assert "output_quality: good (level 3 of 4" in block
        assert "derived correctness score: 0.80" in block
