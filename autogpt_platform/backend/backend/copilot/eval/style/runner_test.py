"""The runner without a model: prompt set per expert, the gate's two
directions with a stubbed judge, the fingerprint skip, and the summary shape."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from .assembly import load_fixtures, load_rubric, roster_experts
from .models import (
    DimensionJudgement,
    Judgement,
    ReferencePrompt,
    ScoredResponse,
    Usage,
)
from .runner import (
    RunOptions,
    evaluate_gate,
    plan_jobs,
    run,
    separation,
    summarize_expert,
    wrong_spec_rows,
)
from .scorer import response_score

_RUNNER = "backend.copilot.eval.style.runner"


def _judgement(score: int) -> Judgement:
    rubric = load_rubric()
    return Judgement(
        scores={
            d.key: DimensionJudgement(score=score, evidence="q")
            for d in rubric.dimensions
        }
    )


def test_response_score_spans_the_scale():
    rubric = load_rubric()
    assert response_score(_judgement(1), rubric) == 0.0
    assert response_score(_judgement(5), rubric) == 100.0
    assert response_score(_judgement(3), rubric) == 50.0


def test_plan_assembles_each_experts_own_prompt_set():
    experts = roster_experts()
    jobs = plan_jobs(experts, load_fixtures(), RunOptions(repeats=2, control=3))
    for expert in experts:
        own = [j for j in jobs if j.expert.name == expert.name and j.arm == "expert"]
        assert len(own) == 60
        assert all(j.prompt.id.startswith(expert.name.lower()) for j in own)
        controls = [
            j for j in jobs if j.expert.name == expert.name and j.arm == "no_suffix"
        ]
        assert len(controls) == 3
        assert all(j.prompt.kind != "briefing_lede" for j in controls)


def test_plan_filters_kinds():
    jobs = plan_jobs(
        roster_experts(["Max"]), load_fixtures(), RunOptions(kinds=["failure"])
    )
    assert {j.prompt.kind for j in jobs} == {"failure"}
    assert len(jobs) == 6


def _row(
    expert: str, score: float | None, *, arm="expert", repeat=0, error=None
) -> ScoredResponse:
    return ScoredResponse(
        expert=expert,
        spec_expert=expert,
        arm=arm,
        kind="reply_draft",
        prompt_id=f"{expert.lower()}-reply-01",
        repeat=repeat,
        model="m",
        response="hi",
        score=score,
        error=error,
    )


def test_gate_fails_below_threshold_and_passes_above():
    below = summarize_expert("Maria", [_row("Maria", 65.0), _row("Maria", 68.0)])
    above = summarize_expert("Max", [_row("Max", 80.0), _row("Max", 90.0)])
    outcome = evaluate_gate([below, above], threshold=70)
    assert not outcome.passed
    assert outcome.failing == ["Maria"]
    assert "Maria: mean 66.5" in outcome.reason
    assert evaluate_gate([above], threshold=70).passed
    assert not evaluate_gate([above], threshold=85.01).passed


def test_gate_fails_an_expert_with_too_many_errors():
    rows = [_row("Max", 90.0)] + [_row("Max", None, error="boom") for _ in range(4)]
    summary = summarize_expert("Max", rows)
    assert summary.errors == 4
    assert not evaluate_gate([summary], threshold=70).passed


def test_gate_fails_an_expert_with_no_scores():
    empty = summarize_expert("Frankie", [])
    assert not evaluate_gate([empty], threshold=1).passed


def test_summary_reports_distribution_and_repeats():
    rows = [
        _row("Max", 60.0, repeat=0),
        _row("Max", 80.0, repeat=0),
        _row("Max", 100.0, repeat=1),
    ]
    summary = summarize_expert("Max", rows)
    assert summary.scores.n == 3
    assert summary.scores.mean == 80.0
    assert summary.scores.min == 60.0
    assert summary.by_repeat == [70.0, 100.0]
    assert summary.below_60 == 0
    assert summary.by_kind["reply_draft"].n == 3


def test_wrong_spec_rows_pair_each_response_with_every_other_expert():
    experts = roster_experts()
    rows = [_row("Maria", 80.0), _row("Max", None, error="x")]
    crossed = wrong_spec_rows(rows, experts)
    assert {(r.expert, r.spec_expert) for r in crossed} == {
        ("Maria", "Max"),
        ("Maria", "Frankie"),
    }
    assert all(r.arm == "wrong_spec" and r.response == "hi" for r in crossed)


def test_separation_compares_own_spec_with_the_controls():
    rows = [
        _row("Maria", 85.0),
        _row("Maria", 75.0),
        _row("Maria", 40.0, arm="wrong_spec"),
        _row("Maria", 50.0, arm="no_suffix"),
    ]
    sep = separation(rows)
    assert sep is not None
    assert sep.right_spec_mean == 80.0
    assert sep.wrong_spec_mean == 40.0
    assert sep.no_suffix_mean == 50.0
    assert sep.gap == 30.0
    assert sep.separated
    assert separation([_row("Maria", 85.0)]) is None


@pytest.fixture
def stubbed_models(tmp_path: Path):
    """Generation and judging without a model, the router pinned to a name."""
    from .assembly import RoutedModel
    from .generation import Turn

    routed = RoutedModel(
        mode="thinking",
        slug="anthropic/claude-x",
        transport_slug="anthropic/claude-x",
        source="env",
    )
    usage = Usage(model="m", input_tokens=10, output_tokens=5, cost_usd=0.001)
    turn = Turn(text="text", truncated=False, usage=usage, tool_calls=["memory_search"])
    judge = AsyncMock(return_value=(_judgement(4), usage))
    with (
        patch(f"{_RUNNER}.resolve_chat_model", AsyncMock(return_value=routed)),
        patch(f"{_RUNNER}.chat_client", MagicMock()),
        patch(f"{_RUNNER}.generate_turn", AsyncMock(return_value=turn)),
        patch(f"{_RUNNER}.generate_lede", AsyncMock(return_value=("lede", usage))),
        patch(f"{_RUNNER}.judge_response", judge),
        patch(f"{_RUNNER}.save_gate") as save_gate,
    ):
        yield judge, save_gate


@pytest.mark.asyncio
async def test_run_scores_every_prompt_and_gates(stubbed_models, tmp_path: Path):
    judge, save_gate = stubbed_models
    out = tmp_path / "r.json"
    result, code = await run(
        RunOptions(experts=["Max"], out=out, gate=True, force=True)
    )
    assert result is not None
    assert code == 0
    assert result.gate is not None and result.gate.passed
    assert result.experts[0].scores.n == 30
    assert result.experts[0].scores.mean == 75.0
    assert result.experts[0].tool_calls == 27
    assert judge.await_count == 30
    assert result.cost_usd == pytest.approx(0.06)
    save_gate.assert_called_once()
    written = json.loads(out.read_text())
    assert written["fingerprint"] == result.fingerprint
    assert len(written["responses"]) == 30


@pytest.mark.asyncio
async def test_run_fails_the_gate_below_threshold(stubbed_models, tmp_path: Path):
    judge, save_gate = stubbed_models
    judge.return_value = (_judgement(2), Usage(model="m"))
    result, code = await run(
        RunOptions(experts=["Max"], out=tmp_path / "r.json", gate=True, force=True)
    )
    assert result is not None and result.gate is not None
    assert code == 1
    assert result.gate.failing == ["Max"]
    save_gate.assert_not_called()


@pytest.mark.asyncio
async def test_gate_skips_while_fingerprint_unchanged(stubbed_models, tmp_path: Path):
    from .assembly import load_gate

    judge, _ = stubbed_models
    gate = load_gate()
    with (
        patch(f"{_RUNNER}.fingerprint", return_value="same"),
        patch(
            f"{_RUNNER}.load_gate",
            return_value=gate.model_copy(update={"last_gated_fingerprint": "same"}),
        ),
    ):
        result, code = await run(
            RunOptions(experts=["Max"], out=tmp_path / "r.json", gate=True)
        )
        assert (result, code) == (None, 0)
        assert judge.await_count == 0
        result, code = await run(
            RunOptions(experts=["Max"], out=tmp_path / "r.json", gate=True, force=True)
        )
        assert result is not None and code == 0


@pytest.mark.asyncio
async def test_run_without_gate_writes_no_verdict(stubbed_models, tmp_path: Path):
    _, save_gate = stubbed_models
    result, code = await run(
        RunOptions(experts=["Max"], kinds=["failure"], out=tmp_path / "r.json")
    )
    assert result is not None and code == 0
    assert result.gate is None
    assert result.experts[0].scores.n == 6
    save_gate.assert_not_called()


def test_reference_prompt_rejects_mismatched_inputs():
    with pytest.raises(ValueError):
        ReferencePrompt(id="x", kind="failure", prompt="")
    with pytest.raises(ValueError):
        ReferencePrompt(id="x", kind="briefing_lede", prompt="p")
